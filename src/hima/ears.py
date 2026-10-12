"""The model of stage 1: the ears, and one small vote layer on top of them.

Sound goes through four steps:
  1. sound          16,000 numbers for each second
  2. sound picture  100 columns for each second, 128 numbers in a column   (made on the CPU, see data.py)
  3. frames         about 13 frames for each second, 896 numbers each     (the ears make these)
  4. votes          for each frame, one score for every piece in our table (the vote layer makes these)

The ears come from Qwen3-ASR-0.6B and already hear. Training teaches them our languages.
The vote layer is new and small. It must read the text from the frames alone, with no knowledge of language
to lean on. So it shows, frame by frame, what the ears really carry, and it keeps the writer from guessing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers.models.qwen3_asr.modeling_qwen3_asr as qwen_code
from transformers import AutoProcessor

WINDOW = 50          # the ears read the sound picture in pieces of 100 columns (1 second); this is the model's own setting


def load_feature_maker(model_dir):
    """The part that turns sound into the sound picture. It has no weights to train."""
    return AutoProcessor.from_pretrained(model_dir).feature_extractor


def make_sound_picture(feature_maker, sounds):
    """list of 1-D sounds (16 kHz) -> picture [clips, 128, columns] and a mask [clips, columns] (1 = real, 0 = padding)"""
    out = feature_maker(sounds, sampling_rate=16000, padding=True, truncation=False,
                        return_attention_mask=True, n_window=WINDOW, return_tensors="pt")
    picture = torch.as_tensor(out["input_features"]).float()
    mask = torch.as_tensor(out["attention_mask"]).long()
    assert picture.shape[-1] == mask.shape[-1] and picture.shape[-1] % (2 * WINDOW) == 0, (picture.shape, mask.shape)
    return picture, mask


def frames_for(columns):
    """How many frames the ears give for a clip with this many real columns (a tensor of counts)."""
    return qwen_code._get_feat_extract_output_lengths(columns, WINDOW)


def hide_parts(picture, mask, time_spans=2, time_width=40, freq_spans=2, freq_width=14):
    """Training only: blank out a few short stretches of time and a few bands of pitch.

    The model cannot lean on one moment or one band of the picture, so it learns sturdier frames.
    (This is "SpecAugment". Width 40 columns = 0.4 seconds.)
    """
    picture = picture.clone()
    clips, bands, _ = picture.shape
    real = mask.sum(-1)
    for c in range(clips):
        for _ in range(time_spans):
            width = int(torch.randint(0, time_width + 1, (1,)))
            start = int(torch.randint(0, max(1, int(real[c]) - width), (1,)))
            picture[c, :, start:start + width] = 0.0
        for _ in range(freq_spans):
            width = int(torch.randint(0, freq_width + 1, (1,)))
            start = int(torch.randint(0, bands - width, (1,)))
            picture[c, start:start + width, :] = 0.0
    return picture


class VoteLayer(nn.Module):
    """frames -> for each frame, a score for every piece in our table"""

    def __init__(self, width, number_of_votes, hidden=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, number_of_votes))

    def forward(self, frames, counts):
        """frames: [all real frames of the batch, 896]  ->  scores [longest clip's frames, clips, votes] (log of probability)"""
        scores = self.layers(frames).float().log_softmax(-1)
        return nn.utils.rnn.pad_sequence(torch.split(scores, counts.tolist()))


def vote_loss(scores, counts, targets, target_lengths):
    """The vote loss (CTC), one number for each clip.

    We know the pieces of a clip and their order. We do not know at which frame each piece is spoken.
    The vote loss adds up the probability of EVERY way to place the pieces on the frames, in order,
    with "no piece" allowed between them. Training makes that sum larger.
    Divided by the number of pieces, so that a long clip does not count more than a short one.
    """
    per_clip = F.ctc_loss(scores, targets, counts, target_lengths, blank=0, reduction="none", zero_infinity=True)
    return per_clip / target_lengths.clamp(min=1)


@torch.no_grad()
def read_votes(scores, counts):
    """The simple way to read text from votes: take the best vote of each frame,
    merge frames that repeat the same vote, and drop "no piece". -> one list of votes for each clip"""
    best = scores.argmax(-1).transpose(0, 1)                          # [clips, frames]
    out = []
    for c in range(best.shape[0]):
        row = best[c, :int(counts[c])].tolist()
        kept, last = [], None
        for v in row:
            if v != last and v != 0:
                kept.append(v)
            last = v
        out.append(kept)
    return out


def load_ears(model_dir):
    """Load the whole speech model and keep only its ears."""
    full = qwen_code.Qwen3ASRForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32)
    ears = full.model.audio_tower
    del full
    return ears
