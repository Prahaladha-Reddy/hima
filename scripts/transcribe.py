"""Write the text of one sound file with a Hima checkpoint from Hugging Face.

    python scripts/transcribe.py my_clip.wav
    python scripts/transcribe.py my_clip.wav --step 140000      # a chosen checkpoint (default: the newest)
    python scripts/transcribe.py my_clip.wav --checkpoint DIR   # a folder that holds model.pt, config.json, pieces.json

Needs: torch, torchaudio, transformers (5.13 or newer), soundfile, huggingface_hub. Run it from the top folder of this repo.
The first run downloads the checkpoint (about 2.4 GB) and two small settings files.
Clips are best between 1 and 30 seconds, 16 kHz. Other rates are resampled.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import soundfile as sf
import torch
import transformers.models.qwen3_asr.modeling_qwen3_asr as qwen_code
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from hima.ears import load_feature_maker, make_sound_picture, read_votes   # noqa: E402
from hima.model import Hima                                                # noqa: E402
from hima.text import PieceTable                                          # noqa: E402

REPO = "Prahaladha/hima"
RUN = "full"
EARS = "Qwen/Qwen3-ASR-1.7B-hf"        # only its settings and its sound-picture maker are used; the weights come from the checkpoint


def newest_step():
    """The newest checkpoint of the run. The small file config.json at the top of the model repo names it; if that file cannot be read, the list of files is used."""
    try:
        with open(hf_hub_download(REPO, "config.json"), encoding="utf-8") as f:
            return int(json.load(f)["newest_step"])
    except Exception:
        pass
    steps = []
    for name in HfApi().list_repo_files(REPO):
        m = re.fullmatch(RUN + r"/step_(\d+)/model\.pt", name)
        if m:
            steps.append(int(m.group(1)))
    if not steps:
        sys.exit("no checkpoint found in " + REPO)
    return max(steps)


def fetch(step):
    folder = "%s/step_%d" % (RUN, step)
    return {f: hf_hub_download(REPO, folder + "/" + f) for f in ("config.json", "pieces.json", "model.pt")}


def read_sound(path):
    sound, rate = sf.read(path, dtype="float32", always_2d=True)
    sound = sound.mean(axis=1)                                   # one channel
    if rate != 16000:
        import torchaudio
        sound = torchaudio.functional.resample(torch.from_numpy(sound), rate, 16000).numpy()
    if len(sound) < 16000:
        print("warning: the clip is shorter than 1 second", file=sys.stderr)
    if len(sound) > 30 * 16000:
        print("warning: the clip is longer than 30 seconds; the model was trained on clips up to 30 seconds", file=sys.stderr)
    return np.ascontiguousarray(sound)


def build(files):
    with open(files["config.json"], encoding="utf-8") as f:
        cfg = json.load(f)
    tr = cfg["train"]
    pieces = PieceTable.load(files["pieces.json"])
    ears = qwen_code.Qwen3ASREncoder(AutoConfig.from_pretrained(EARS).audio_config)
    writer = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(cfg["writer"]), dtype=torch.float32)
    model = Hima(ears, pieces.size, writer, tr["vote_hidden"], tr["dropout"], tr["bridge_hidden"])
    model.load_state_dict(torch.load(files["model.pt"], map_location="cpu"))      # strict: every weight must be found
    return model.eval(), pieces


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sound", help="a sound file (wav, flac, ...)")
    ap.add_argument("--step", type=int, help="checkpoint step (default: the newest)")
    ap.add_argument("--checkpoint", help="a local folder with model.pt, config.json and pieces.json")
    ap.add_argument("--votes", action="store_true", help="also print what the vote layer reads (a helper of training)")
    args = ap.parse_args()

    if args.checkpoint:
        files = {f: os.path.join(args.checkpoint, f) for f in ("config.json", "pieces.json", "model.pt")}
    else:
        step = args.step or newest_step()
        print("checkpoint: %s/step_%d" % (RUN, step), file=sys.stderr)
        files = fetch(step)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, pieces = build(files)
    model.to(device)
    feature_maker = load_feature_maker(EARS)

    picture, mask = make_sound_picture(feature_maker, [read_sound(args.sound)])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        frames, counts = model.hear(picture.to(device), mask.to(device))
        ids = model.head.write(frames, counts)[0]
        print(pieces.decode_ids(ids))
        if args.votes:
            print("votes:", pieces.decode(read_votes(model.vote(frames, counts), counts)[0]))


if __name__ == "__main__":
    main()
