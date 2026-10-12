"""Text for stage 1: one form for every clip, and the table of pieces that the vote layer chooses from.

Three things live here:
  clean_text   - turns any stored text into the one form that the model learns to write
  PieceTable   - text -> piece numbers for the vote loss, and piece numbers -> text
  letters_wrong - the score: how many letters differ between two texts
"""
import json
import re
import unicodedata

# Two marks that are invisible but change how Indian scripts join letters. We must keep them.
JOINERS = {"‌", "‍"}
# A sound tag such as <noise> or [laugh] is not speech. We take it out.
TAG = re.compile(r"<[^>]*>|\[[^\]]*\]")


def clean_text(text):
    """The one form of text: no marks, no tags, small Latin letters, single spaces.

    'Namaste, ji! [laugh]'  ->  'namaste ji'
    Letters of Indian scripts are not changed. Digits stay as they are.
    """
    text = unicodedata.normalize("NFC", text or "")
    text = TAG.sub(" ", text)
    out = []
    for i, ch in enumerate(text):
        kind = unicodedata.category(ch)[0]        # L letter, M vowel sign, N digit, P mark, S symbol, Z space, C control
        if ch in JOINERS:
            out.append(ch)
        elif ch in "'’" and 0 < i < len(text) - 1 and text[i - 1].isalpha() and text[i + 1].isalpha():
            out.append("'")                        # the mark inside "don't" stays
        elif kind in "LMN":
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).lower().split())


def looks_like_a_loop(text):
    """True for a text that is stuck on a word: "and then then then then then".

    Text that a machine wrote sometimes ends in such a loop. A clip with a loop as its text teaches the
    writer to loop too, so we drop it. Two signs: one word four times in a row, or a long text made of very few words.
    """
    words = text.split()
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        if run >= 4:
            return True
    return len(words) >= 12 and len(set(words)) / len(words) < 0.25


class PieceTable:
    """The pieces that the vote layer can name.

    The writer of the full model (stage 2) cuts text into pieces with its own tokenizer.
    We use the same cut here, so that the ears learn the pieces that the writer will use.
    The writer's table is very large, so we keep only the pieces that our text really uses.

    Vote number 0 = "no piece at this frame" (the blank of the vote loss).
    Vote number 1 = a piece that is not in our table.
    Vote numbers 2 and up = our pieces.
    """
    BLANK, UNKNOWN, FIRST = 0, 1, 2

    def __init__(self, tokenizer_name, piece_ids):
        from transformers import AutoTokenizer
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.piece_ids = list(piece_ids)                                   # vote number - 2  ->  the writer's piece number
        self.vote_of = {p: k + self.FIRST for k, p in enumerate(self.piece_ids)}

    @property
    def size(self):
        return len(self.piece_ids) + self.FIRST

    def encode(self, text):
        """clean text -> list of vote numbers"""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return [self.vote_of.get(i, self.UNKNOWN) for i in ids]

    def encode_both(self, text):
        """clean text -> (the writer's own piece numbers, the same pieces as vote numbers). Both lists have the same length."""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return ids, [self.vote_of.get(i, self.UNKNOWN) for i in ids]

    def decode_ids(self, ids):
        """the writer's own piece numbers -> text"""
        return clean_text(self.tokenizer.decode(ids))

    def decode(self, votes):
        """list of vote numbers -> text"""
        ids = [self.piece_ids[v - self.FIRST] for v in votes if v >= self.FIRST]
        return clean_text(self.tokenizer.decode(ids))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tokenizer": self.tokenizer_name, "piece_ids": self.piece_ids}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["tokenizer"], d["piece_ids"])


def edit_distance(a, b):
    """The smallest number of letters to add, remove or change to turn a into b."""
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:
        pass
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def letters_wrong(reference, written):
    """(wrong letters, letters in the reference). Spaces are not counted."""
    r, w = reference.replace(" ", ""), written.replace(" ", "")
    return edit_distance(r, w), len(r)


def texts_differ(a, b):
    """Share of letters that differ between two texts, 0.0 = same, 1.0 or more = unlike."""
    a, b = clean_text(a).replace(" ", ""), clean_text(b).replace(" ", "")
    if not a or not b:
        return 1.0
    return edit_distance(a, b) / max(len(a), len(b))
