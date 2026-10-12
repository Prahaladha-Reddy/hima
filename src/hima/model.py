"""The whole model in one place: ears, then two readers of the frames.

                         +--> vote layer --> a piece for each frame        (vote loss)
    sound picture --> ears
                         +--> bridge --> writer --> the text, piece by piece (writer loss)

The two losses are added with weights from the config:
    writer_weight > 0, vote_weight = 0    the writer alone
    writer_weight = 0, vote_weight > 0    the ears and the vote layer alone (no writer is loaded)
    both > 0                              the writer, with the vote layer as a helper
A part with weight 0 is not built at all, so every weight that exists gets a gradient in every step.
(With many GPUs, a weight that gets no gradient stops the run.)
"""
import torch
import torch.nn as nn

from .ears import VoteLayer, frames_for, load_ears, vote_loss
from .writer import WriterHead


class Hima(nn.Module):
    def __init__(self, ears, number_of_votes=None, writer=None, vote_hidden=1024, dropout=0.1, bridge_hidden=1024):
        super().__init__()
        self.ears = ears
        width = ears.config.d_model                                   # 896 numbers in a frame
        self.vote = VoteLayer(width, number_of_votes, vote_hidden, dropout) if number_of_votes else None
        self.head = WriterHead(writer, width, bridge_hidden) if writer is not None else None

    def hear(self, picture, mask):
        """-> frames [all real frames of the batch, 896], clip after clip, and the number of frames of each clip"""
        frames = self.ears(picture, mask).last_hidden_state
        counts = frames_for(mask.sum(-1))
        assert int(counts.sum()) == frames.shape[0], (int(counts.sum()), frames.shape)
        return frames, counts

    def forward(self, picture, mask, votes, text_ids, lengths):
        """-> the vote loss and the writer loss, one number each for every clip (zeros for a part that is not built)"""
        frames, counts = self.hear(picture, mask)
        nothing = torch.zeros(len(counts), device=frames.device)
        vote_part = vote_loss(self.vote(frames, counts), counts, votes, lengths) if self.vote is not None else nothing
        writer_part = self.head.loss(frames, counts, text_ids, lengths).float() if self.head is not None else nothing
        return vote_part, writer_part


def build_model(cfg, number_of_votes):
    """Builds the parts that the config asks for."""
    tr = cfg["train"]
    writer = None
    if tr["writer_weight"] > 0:
        from transformers import AutoModelForCausalLM
        writer = AutoModelForCausalLM.from_pretrained(cfg["writer"], dtype=torch.float32)
    return Hima(load_ears(cfg["model_dir"]), number_of_votes if tr["vote_weight"] > 0 else None, writer,
                tr["vote_hidden"], tr["dropout"], tr["bridge_hidden"])
