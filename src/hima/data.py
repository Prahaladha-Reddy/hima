"""Clips for training: read from shards, mixed, cut into batches, turned into tensors.

A SHARD is two files:
  an audio file   - a tar with thousands of short sound files
  a table         - one row for each sound file: its text, its length, and a key (who speaks, or which talk)

The shards are too large to copy, so we read each tar as a stream, from first byte to last.
One process does this for a few shards at the same time, and mixes their clips.

Rules that this file enforces:
  - a clip whose key is "held out" is never given to training (it belongs to the test)
  - every training process (each GPU has several) reads its own shards, never the shards of another
  - the stream never ends: when all shards are read, they are shuffled and read again.
    Training counts steps, not passes. No GPU can run out of batches while another still has some.
"""
import io
import json
import random
import tarfile
import time
import urllib.request
import zlib

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .ears import load_feature_maker, make_sound_picture
from .text import PieceTable, clean_text, looks_like_a_loop, texts_differ

RATE = 16000


# ---------------------------------------------------------------- reading files

class Stream:
    """One file, read from start to end. The file is a path, or a web link.
    If a web connection drops in the middle, we open it again at the same byte and go on.
    From the web we take 4 MB at a time and hand out small parts of it: many small web reads are slow."""
    CHUNK = 4 << 20

    def __init__(self, where, tries=6):
        self.where, self.tries = where, tries
        self.position, self.handle = 0, None                      # position = bytes taken from the file so far
        self.chunk, self.used = b"", 0                             # the part we hold, and how much of it was handed out
        self.is_web = where.startswith("http://") or where.startswith("https://")

    def _open(self):
        if not self.is_web:
            self.handle = open(self.where, "rb")
            self.handle.seek(self.position)
            return
        headers = {"Range": "bytes=%d-" % self.position} if self.position else {}
        self.handle = urllib.request.urlopen(urllib.request.Request(self.where, headers=headers), timeout=60)

    def _take(self):
        """the next part of the file, or b"" at its end"""
        for attempt in range(self.tries):
            try:
                if self.handle is None:
                    self._open()
                data = self.handle.read(self.CHUNK)
                self.position += len(data)
                return data
            except Exception:
                self.handle = None
                time.sleep(2.0 * (attempt + 1))
        raise IOError("cannot read " + self.where.split("?")[0])

    def read(self, size=-1):
        parts, need = [], size if size is not None and size >= 0 else None
        while need is None or need > 0:
            if self.used >= len(self.chunk):
                self.chunk, self.used = self._take(), 0
                if not self.chunk:
                    break
            part = self.chunk[self.used:] if need is None else self.chunk[self.used:self.used + need]
            self.used += len(part)
            parts.append(part)
            if need is not None:
                need -= len(part)
        return b"".join(parts)

    def close(self):
        if self.handle is not None:
            self.handle.close()


def read_all(where):
    stream = Stream(where)
    try:
        return stream.read()
    finally:
        stream.close()


def load_shard_list(path):
    """One line for each shard: {"lang": "hi", "kind": "a", "audio": <path or link>, "table": <path or link>, "hours": 25.7}"""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_held_out(key, percent, extra_keys):
    """True if every clip with this key belongs to the test.
    The rule needs no list: the key itself decides, the same way on every machine, forever."""
    if key in extra_keys:
        return True
    return zlib.crc32(str(key).encode("utf-8")) % 100 < percent


def read_table(where, spec):
    """The table of one shard -> {name of the sound file: (text, key, seconds)}

    spec says which columns to use:  {"id": ..., "text": ..., "key": ..., "seconds": ..., "second_texts": [...], "max_differ": 0.2}
    If "second_texts" is given, a clip is kept only if at least one of those other texts agrees with its text.
    Two texts that were written in different ways and still agree are much more likely to be right.
    """
    import pyarrow.parquet as pq
    second = list(spec.get("second_texts") or [])
    wanted = [spec["id"], spec["text"], spec["key"], spec["seconds"]] + second
    table = pq.read_table(io.BytesIO(read_all(where)))
    table = table.select([c for c in wanted if c in table.column_names])
    rows, dropped = {}, {"no_text": 0, "texts_differ": 0, "loop_text": 0}
    for r in table.to_pylist():
        text = clean_text(r.get(spec["text"]) or "")
        if not text or "redacted" in text:
            dropped["no_text"] += 1
            continue
        if looks_like_a_loop(text):
            dropped["loop_text"] += 1
            continue
        if second:
            others = [r.get(c) for c in second if r.get(c)]
            if not others or min(texts_differ(text, o) for o in others) > spec.get("max_differ", 0.2):
                dropped["texts_differ"] += 1
                continue
        rows[str(r[spec["id"]])] = (text, str(r[spec["key"]]), float(r.get(spec["seconds"]) or 0.0))
    return rows, dropped


def decode_sound(raw):
    """bytes of one sound file -> 1-D numbers at 16 kHz"""
    import soundfile as sf
    sound, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    sound = sound.mean(axis=1)                                   # two channels -> one
    if rate != RATE:
        import soxr
        sound = soxr.resample(sound, rate, RATE).astype(np.float32)
    return sound


def enough_frames(seconds, votes):
    """The vote loss needs at least one frame for each piece, and one more between two equal pieces."""
    frames = int(seconds * 12.5)
    repeats = sum(1 for a, b in zip(votes, votes[1:]) if a == b)
    return frames >= len(votes) + repeats


# ---------------------------------------------------------------- the stream of clips

class ClipStream(IterableDataset):
    def __init__(self, shards, table_specs, pieces_path, model_dir, cfg, rank=0, world=1, seed=0, train=True, extra_held_out=()):
        self.shards, self.table_specs = shards, table_specs
        self.pieces_path, self.model_dir, self.cfg = pieces_path, model_dir, cfg
        self.rank, self.world, self.seed, self.train = rank, world, seed, train
        self.extra_held_out = set(extra_held_out)

    # ---- one shard -> its clips
    def clips_of_shard(self, shard, pieces, counts):
        try:
            rows, dropped = read_table(shard["table"], self.table_specs[shard["kind"]])
        except Exception:
            counts["shard_failed"] += 1
            return
        for k, v in dropped.items():
            counts[k] += v
        stream = Stream(shard["audio"])
        try:
            tar = tarfile.open(fileobj=stream, mode="r|")
            for member in tar:
                if not member.isfile():
                    continue
                name = member.name
                row = rows.get(name) or rows.get(name.rsplit(".", 1)[0])
                if row is None:
                    continue
                text, key, seconds = row
                held = is_held_out(key, self.cfg["heldout_percent"], self.extra_held_out)
                if held == self.train:                               # training skips the test clips; the test reader skips the rest
                    counts["held_out"] += self.train
                    continue
                if not (self.cfg["min_seconds"] <= seconds <= self.cfg["max_seconds"]):
                    counts["length"] += 1
                    continue
                ids, votes = pieces.encode_both(text)
                if not votes or not enough_frames(seconds, votes):
                    counts["too_dense"] += 1
                    continue
                try:
                    sound = decode_sound(tar.extractfile(member).read())
                except Exception:
                    counts["bad_sound"] += 1
                    continue
                yield {"sound": sound, "votes": votes, "ids": ids, "lang": shard["lang"], "seconds": len(sound) / RATE, "text": text, "name": name}
        except Exception:
            counts["shard_failed"] += 1
        finally:
            stream.close()

    # ---- which shards belong to this process
    def my_shards(self, slot, slots):
        """Shards of each language are dealt out like cards: shard i goes to process i % slots."""
        by_lang = {}
        for lang in sorted({s["lang"] for s in self.shards}):
            mine = [s for s in self.shards if s["lang"] == lang]
            random.Random(self.seed).shuffle(mine)                   # the same order in every process
            mine = mine[slot::slots]
            if mine:
                by_lang[lang] = mine
        return by_lang

    # ---- many shards -> one endless, mixed stream
    def mixed_clips(self, by_lang, pieces, counts, rng):
        """Keeps a few shards open, takes a clip from each in turn, and shuffles in a buffer.

        Which language is opened next is drawn by chance. A language with H hours gets weight H ** power.
        power = 1 would follow the hours (large languages drown the small ones); power = 0 would give
        every language the same share. We use a value between the two.
        """
        langs = sorted(by_lang)
        weights = [sum(s.get("hours", 1.0) for s in by_lang[l]) ** self.cfg["language_power"] for l in langs]
        order = {l: [] for l in langs}

        def next_shard():
            lang = rng.choices(langs, weights)[0]
            if not order[lang]:                                      # all read: shuffle and start again
                order[lang] = list(by_lang[lang])
                rng.shuffle(order[lang])
            return self.clips_of_shard(order[lang].pop(), pieces, counts)

        open_shards = [next_shard() for _ in range(self.cfg["open_shards"])]
        buffer = []
        while True:
            for i in range(len(open_shards)):
                clip = next(open_shards[i], None)
                if clip is None:
                    open_shards[i] = next_shard()
                    continue
                buffer.append(clip)
                if len(buffer) >= self.cfg["shuffle_clips"]:
                    j = rng.randrange(len(buffer))
                    buffer[j], buffer[-1] = buffer[-1], buffer[j]
                    yield buffer.pop()

    # ---- clips -> batches of tensors
    def batches(self, clips, feature_maker, counts, rng):
        """Takes a pool of clips, sorts it by length, and cuts it into batches.
        Clips of like length sit together, so little of a batch is padding.
        A batch holds at most `batch_seconds` of sound, padding counted, and at most `batch_clips` clips."""
        pool = []
        for clip in clips:
            pool.append(clip)
            if len(pool) < self.cfg["pool_clips"]:
                continue
            pool.sort(key=lambda c: len(c["sound"]))
            cut, batch = [], []
            for clip in pool:
                longest = len(clip["sound"]) / RATE                  # sorted, so the new clip is the longest
                if batch and ((len(batch) + 1) * longest > self.cfg["batch_seconds"] or len(batch) >= self.cfg["batch_clips"]):
                    cut.append(batch)
                    batch = []
                batch.append(clip)
            if batch:
                cut.append(batch)
            rng.shuffle(cut)
            for batch in cut:
                yield collate(batch, feature_maker, dict(counts))
                for k in counts:
                    counts[k] = 0
            pool = []

    def __iter__(self):
        worker = get_worker_info()
        workers = worker.num_workers if worker else 1
        slot = self.rank * workers + (worker.id if worker else 0)
        rng = random.Random(self.seed * 100003 + slot)
        torch.set_num_threads(1)
        pieces = PieceTable.load(self.pieces_path)
        feature_maker = load_feature_maker(self.model_dir)
        counts = {k: 0 for k in ("no_text", "texts_differ", "loop_text", "held_out", "length", "too_dense", "bad_sound", "shard_failed")}
        by_lang = self.my_shards(slot, self.world * workers)
        if not by_lang:
            raise RuntimeError("process %d got no shard: use fewer workers or more shards" % slot)
        clips = self.mixed_clips(by_lang, pieces, counts, rng)
        return self.batches(clips, feature_maker, counts, rng)


def collate(batch, feature_maker, counts=None):
    """list of clips -> the tensors that one training step needs"""
    picture, mask = make_sound_picture(feature_maker, [c["sound"] for c in batch])
    lengths = torch.tensor([len(c["votes"]) for c in batch])
    votes = torch.zeros(len(batch), int(lengths.max()), dtype=torch.long)      # the pieces as vote numbers (our small table)
    text_ids = torch.zeros(len(batch), int(lengths.max()), dtype=torch.long)   # the same pieces as the writer's own numbers
    for i, c in enumerate(batch):
        votes[i, :len(c["votes"])] = torch.tensor(c["votes"])
        text_ids[i, :len(c["ids"])] = torch.tensor(c["ids"])
    return {"picture": picture, "mask": mask, "votes": votes, "text_ids": text_ids, "lengths": lengths,
            "seconds": float(sum(c["seconds"] for c in batch)), "langs": [c["lang"] for c in batch],
            "texts": [c["text"] for c in batch], "dropped": counts or {}}


# ---------------------------------------------------------------- the test pack

def test_batches(pack_dir, pieces, feature_maker, batch_seconds=240):
    """The test pack is a folder made once by scripts/build_test_pack.py: clips.jsonl and one sound file for each clip.
    Read in the same order every time, longest first, so that a memory problem shows at once."""
    import soundfile as sf
    with open(pack_dir + "/clips.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    rows.sort(key=lambda r: -r["seconds"])
    batch, longest = [], 0.0
    for r in rows:
        if batch and (len(batch) + 1) * longest > batch_seconds:
            yield collate(batch, feature_maker)
            batch = []
        if not batch:
            longest = r["seconds"]
        sound, _ = sf.read("%s/audio/%s" % (pack_dir, r["file"]), dtype="float32")
        ids, votes = pieces.encode_both(r["text"])
        batch.append({"sound": sound, "votes": votes, "ids": ids, "lang": r["lang"], "seconds": r["seconds"], "text": r["text"], "name": r["file"]})
    if batch:
        yield collate(batch, feature_maker)
