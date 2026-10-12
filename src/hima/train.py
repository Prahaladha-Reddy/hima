"""The training run.

Start it with:   bash scripts/run.sh <config.yaml> <GPU list>

What one step does, on every GPU at the same time:
  1. take a batch of clips (sound picture + the pieces of their text)
  2. the ears turn the picture into frames
  3. two readers look at the frames, and each gets a loss:
       the writer loss - the writer must predict each next piece of the true text
       the vote loss   - the vote layer must name the pieces from the frames alone
     the two are added, with the weights from the config
  4. backward() finds, for every weight, which way to move it; the GPUs average their answers
  5. the optimizer moves the weights a small step

The model has four parts, and each has its own speed (learning rate):
  ears      already trained by their makers: slow, and they WAIT at first
  writer    already trained on text by its makers: slow, and it waits too
  bridge    new: fast, from the first step
  vote      new: fast, from the first step
While the new parts know nothing, their signals are noise. The trained parts wait until that noise is gone.

The file is long because a real run must also: start slowly, save its work, pick up after a stop,
measure itself on the test pack, and say clearly when something is wrong.
"""
import faulthandler
import json
import math
import os
import shutil
import sys
import time

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .data import ClipStream, load_shard_list
from .ears import hide_parts, load_feature_maker
from .model import build_model
from .evaluate import score_on_test_pack
from .text import PieceTable


def learning_rate_scale(step, start, warmup, end):
    """A number between 0 and 1 that the base learning rate is multiplied by.

    before `start`:            0     (this part of the model does not move yet)
    the next `warmup` steps:   rises in a straight line from 0 to 1
    after that:                falls along a smooth curve to 0.1 at the last step
    """
    if step < start:
        return 0.0
    if step < start + warmup:
        return (step - start + 1) / warmup
    done = (step - start - warmup) / max(1, end - start - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, done)))


def newest_checkpoint(run_dir):
    if not os.path.isdir(run_dir):
        return None
    steps = [int(d.split("_")[1]) for d in os.listdir(run_dir) if d.startswith("step_") and os.path.exists("%s/%s/model.pt" % (run_dir, d))]
    return "%s/step_%d" % (run_dir, max(steps)) if steps else None


def save_checkpoint(run_dir, step, model, optimizer, cfg, scores, keep):
    """Writes into a temporary folder first and renames it at the end.
    A stop in the middle of a save then leaves no half-written checkpoint that looks whole."""
    final, tmp = "%s/step_%d" % (run_dir, step), "%s/.tmp_step_%d" % (run_dir, step)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    torch.save(model.state_dict(), tmp + "/model.pt")
    torch.save({"optimizer": optimizer.state_dict(), "step": step}, tmp + "/optim.pt")
    shutil.copy(cfg["pieces"], tmp + "/pieces.json")
    with open(tmp + "/config.json", "w") as f:
        json.dump(cfg, f, indent=1)
    with open(tmp + "/commit.txt", "w") as f:
        f.write(os.environ.get("HIMA_COMMIT", "unknown") + "\n")
    if scores:
        with open(tmp + "/scores.json", "w") as f:
            json.dump(scores, f, indent=1, ensure_ascii=False)
    os.replace(tmp, final)
    old = sorted(int(d.split("_")[1]) for d in os.listdir(run_dir) if d.startswith("step_"))
    for s in old[:-keep]:
        shutil.rmtree("%s/step_%d" % (run_dir, s), ignore_errors=True)


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    data_cfg, tr = cfg["data"], cfg["train"]

    # ---- which copy am I?  torchrun starts one copy of this program for each GPU and tells each its number.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    many = world > 1
    if many:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    first = rank == 0                                               # only the first copy prints, tests and saves
    torch.manual_seed(tr["seed"] + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    faulthandler.enable()                                           # a crash inside a library still prints where it was

    run_dir = "%s/%s" % (cfg["out_dir"], cfg["run"])
    if first:
        os.makedirs(run_dir, exist_ok=True)

    # ---- the model
    pieces = PieceTable.load(cfg["pieces"])
    model = build_model(cfg, pieces.size).to(device)
    # Four groups of weights, each with its own top speed and its own number of steps to wait.
    groups = [("ears", list(model.ears.parameters()), tr["lr_ears"], tr["ears_wait_steps"])]
    if model.vote is not None:
        groups.append(("vote", list(model.vote.parameters()), tr["lr_vote"], 0))
    if model.head is not None:
        groups.append(("bridge", list(model.head.bridge.parameters()) + [model.head.now_write], tr["lr_bridge"], 0))
        groups.append(("writer", list(model.head.writer.parameters()), tr["lr_writer"], tr["writer_wait_steps"]))
    optimizer = torch.optim.AdamW([{"params": weights, "lr": 0.0} for _, weights, _, _ in groups],
                                  betas=(0.9, 0.98), weight_decay=tr["weight_decay"])

    # ---- pick up after a stop
    step = 0
    last = newest_checkpoint(run_dir)
    if last and tr.get("resume", True):
        model.load_state_dict(torch.load(last + "/model.pt", map_location=device))
        state = torch.load(last + "/optim.pt", map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        if first:
            print("picked up at step %d from %s" % (step, last), flush=True)

    core = model                                                    # the model itself, for saving and testing
    if many:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ---- the data
    with open(data_cfg["tables"]) as f:
        table_specs = yaml.safe_load(f)
    extra = []
    if data_cfg.get("heldout_keys"):
        with open(data_cfg["heldout_keys"]) as f:
            extra = [line.strip() for line in f if line.strip()]
    stream = ClipStream(load_shard_list(cfg["shards"]), table_specs, cfg["pieces"], cfg["model_dir"], data_cfg,
                        rank=rank, world=world, seed=tr["seed"] + step, train=True, extra_held_out=extra)
    loader = DataLoader(stream, batch_size=None, num_workers=data_cfg["workers"], prefetch_factor=4,
                        persistent_workers=True, pin_memory=True)
    feature_maker = load_feature_maker(cfg["model_dir"]) if first else None

    if first:
        sizes = " | ".join("%s %.1fM weights" % (name, sum(p.numel() for p in weights) / 1e6) for name, weights, _, _ in groups)
        print("run %s | GPUs %d | %s | writer loss x %s + vote loss x %s | steps %d -> %d"
              % (cfg["run"], world, sizes, tr["writer_weight"], tr["vote_weight"], step, tr["max_steps"]), flush=True)
        log = open(run_dir + "/log.jsonl", "a")

    # ---- the loop
    model.train()
    seen_seconds, seen_clips, loss_sum, vote_sum, writer_sum, loss_n, skipped, dropped = 0.0, 0, 0.0, 0.0, 0.0, 0, 0, {}
    wait_time, t_log, t_last = 0.0, time.time(), time.time()
    scores = None
    for batch in loader:
        if step >= tr["max_steps"]:
            break
        faulthandler.dump_traceback_later(tr["stuck_seconds"], repeat=True)     # if one step takes this long, print where every copy is
        wait_time += time.time() - t_last

        # the learning rates of this step
        for k, (name, weights, top, wait) in enumerate(groups):
            optimizer.param_groups[k]["lr"] = top * learning_rate_scale(step, wait, tr["warmup_steps"], tr["max_steps"])
            if wait > 0 and step == wait:
                # While this part waited, the optimizer still collected gradient averages for it.
                # Those came from new parts that knew nothing, so we throw them away before the part starts to move.
                for p in weights:
                    optimizer.state.pop(p, None)

        picture, mask = batch["picture"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)
        votes, text_ids, lengths = batch["votes"].to(device), batch["text_ids"].to(device), batch["lengths"].to(device)
        if tr["hide_parts"]:
            picture = hide_parts(picture, mask)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            vote_part, writer_part = model(picture, mask, votes, text_ids, lengths)
        per_clip = tr["writer_weight"] * writer_part + tr["vote_weight"] * vote_part
        good = torch.isfinite(per_clip)
        # A clip with a broken loss gets loss 0. We do NOT skip the step: with many GPUs every copy must
        # call backward() the same number of times, or all of them wait for ever.
        loss = torch.where(good, per_clip, torch.zeros_like(per_clip)).sum() / good.sum().clamp(min=1)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(core.parameters(), tr["clip_norm"])
        if torch.isfinite(norm):
            optimizer.step()
        else:
            skipped += 1                                            # the averaged gradient is the same on every GPU, so all skip together
        step += 1

        # ---- book-keeping (first copy only)
        seen_seconds += batch["seconds"]
        seen_clips += len(batch["langs"])
        loss_sum += float(loss.detach())
        vote_sum += float(vote_part[good].mean()) if bool(good.any()) else 0.0
        writer_sum += float(writer_part[good].mean()) if bool(good.any()) else 0.0
        loss_n += 1
        for k, v in batch["dropped"].items():
            dropped[k] = dropped.get(k, 0) + v
        if first and step % tr["log_every"] == 0:
            spent = time.time() - t_log
            row = {"step": step, "loss": round(loss_sum / loss_n, 4), "vote_loss": round(vote_sum / loss_n, 4), "writer_loss": round(writer_sum / loss_n, 4),
                   "lr": {name: optimizer.param_groups[k]["lr"] for k, (name, _, _, _) in enumerate(groups)}, "grad_norm": round(float(norm), 3),
                   "hours_seen_all_gpus": round(seen_seconds * world / 3600, 2), "clips_per_step": round(seen_clips / loss_n, 1),
                   "times_real_time": round(seen_seconds * world / spent), "seconds_per_step": round(spent / loss_n, 3),
                   "waiting_for_data": round(wait_time / spent, 3), "skipped_steps": skipped, "dropped": dropped,
                   "gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            seen_seconds, seen_clips, loss_sum, vote_sum, writer_sum, loss_n, wait_time, t_log, dropped = 0.0, 0, 0.0, 0.0, 0.0, 0, 0.0, time.time(), {}

        if first and cfg.get("test_pack") and (step % tr["test_every"] == 0 or step == tr["max_steps"]):
            scores = score_on_test_pack(core, cfg["test_pack"], pieces, feature_maker, device)
            scores["step"] = step
            for reader in ("writer", "vote"):
                if reader in scores:
                    print("TEST step %d %s: letters wrong %.1f%% | %s" % (step, reader, 100 * scores[reader]["all"]["letters_wrong"],
                          " ".join("%s %.0f%%" % (l, 100 * v["letters_wrong"]) for l, v in scores[reader]["by_lang"].items())), flush=True)
            with open(run_dir + "/scores.jsonl", "a") as f:
                f.write(json.dumps(scores, ensure_ascii=False) + "\n")
        if first and (step % tr["save_every"] == 0 or step == tr["max_steps"]):
            save_checkpoint(run_dir, step, core, optimizer, cfg, scores, tr["keep"])
            print("saved step %d" % step, flush=True)
        t_last = time.time()

    faulthandler.cancel_dump_traceback_later()
    if first:
        print("DONE step %d" % step, flush=True)
    if many:
        dist.destroy_process_group()
    os._exit(0)                                                     # the data processes never end by themselves


if __name__ == "__main__":
    main()
