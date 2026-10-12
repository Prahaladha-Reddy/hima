"""The score: letters wrong on the test pack, for each language.

The test pack holds clips of speakers that training never sees (see data.py, is_held_out).
Each reader of the frames writes its text, and we compare it, letter by letter, with the text that came with the clip.
  "vote"   - the vote layer: best piece of each frame, repeats merged, empty frames dropped
  "writer" - the writer writes by itself, always taking its most likely next piece
"""
import torch

from .data import test_batches
from .ears import read_votes
from .text import letters_wrong


class Tally:
    """Adds up wrong letters and letters, for each language."""

    def __init__(self):
        self.wrong, self.letters, self.clips, self.shown = {}, {}, {}, []

    def add(self, lang, reference, written, examples):
        w, n = letters_wrong(reference, written)
        self.wrong[lang] = self.wrong.get(lang, 0) + w
        self.letters[lang] = self.letters.get(lang, 0) + n
        self.clips[lang] = self.clips.get(lang, 0) + 1
        if sum(1 for s in self.shown if s["lang"] == lang) < examples:
            self.shown.append({"lang": lang, "reference": reference, "written": written})

    def result(self):
        by_lang = {l: {"letters_wrong": self.wrong[l] / max(1, self.letters[l]), "clips": self.clips[l]} for l in sorted(self.clips)}
        return {"all": {"letters_wrong": sum(self.wrong.values()) / max(1, sum(self.letters.values())), "clips": sum(self.clips.values())},
                "by_lang": by_lang, "examples": self.shown}


@torch.no_grad()
def score_on_test_pack(model, pack_dir, pieces, feature_maker, device, examples=2):
    """-> {"vote": {...}, "writer": {...}}, each with "all", "by_lang" and a few examples. A reader that is not built is left out."""
    was_training = model.training
    model.eval()
    tallies = {}
    if model.vote is not None:
        tallies["vote"] = Tally()
    if model.head is not None:
        tallies["writer"] = Tally()
    for batch in test_batches(pack_dir, pieces, feature_maker):
        picture, mask = batch["picture"].to(device), batch["mask"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            frames, counts = model.hear(picture, mask)
            if model.vote is not None:
                for i, votes in enumerate(read_votes(model.vote(frames, counts), counts)):
                    tallies["vote"].add(batch["langs"][i], batch["texts"][i], pieces.decode(votes), examples)
            if model.head is not None:
                for i, ids in enumerate(model.head.write(frames, counts)):
                    tallies["writer"].add(batch["langs"][i], batch["texts"][i], pieces.decode_ids(ids), examples)
    model.train(was_training)
    return {name: t.result() for name, t in tallies.items()}
