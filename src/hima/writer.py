"""The writer side of the model: a bridge, and a small language model that writes the text.

The writer reads one long row of vectors and predicts the next piece of text, again and again:

    [start]  frame frame frame ... frame  [now write]  piece piece piece ...  [end]
             \\________ the sound ________/              \\____ the text ____/

  - a frame comes from the ears (896 numbers). The BRIDGE turns it into a vector of the writer's size (640).
  - a piece is a small part of a word. The writer has its own vector for each piece.
  - [now write] is one learned vector that says: the sound is over, the text begins.

Training: the true text is given, and the loss asks at each place "how likely was the true next piece?".
Use:      nothing is given after [now write]; the writer chooses a piece, reads it back in, and goes on.

This is the same family of objective as the first stage of the ears' makers:
a writer that looks at all the frames and writes the text piece by piece.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bridge(nn.Module):
    """frame of the ears -> a vector that the writer can read"""

    def __init__(self, ears_width, writer_width, hidden=1024):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(ears_width, hidden), nn.GELU(), nn.Linear(hidden, writer_width), nn.LayerNorm(writer_width))

    def forward(self, frames):
        return self.layers(frames)


class WriterHead(nn.Module):
    def __init__(self, writer, ears_width, bridge_hidden=1024):
        super().__init__()
        self.writer = writer                                          # the small language model
        width = writer.config.hidden_size
        self.bridge = Bridge(ears_width, width, bridge_hidden)
        self.now_write = nn.Parameter(torch.randn(width) * 0.02)     # the learned [now write] vector
        self.start_id = writer.config.bos_token_id
        self.end_id = writer.config.eos_token_id

    def piece_vectors(self, ids):
        return self.writer.get_input_embeddings()(ids)

    def rows(self, frames, counts):
        """For each clip: the vectors [start] + its frames + [now write].   frames: [all frames of the batch, 896]"""
        sound = torch.split(self.bridge(frames), counts.tolist())
        start = self.piece_vectors(torch.tensor([self.start_id], device=frames.device))
        return [torch.cat([start, s.to(start.dtype), self.now_write[None].to(start.dtype)]) for s in sound]

    def loss(self, frames, counts, text_ids, text_lengths):
        """One number for each clip: the mean over its pieces of "how unlikely was the true next piece?".

        text_ids: [clips, longest text] piece numbers of the writer, padded.   text_lengths: [clips]
        """
        clips, device = len(counts), frames.device
        prefixes = self.rows(frames, counts)
        width = prefixes[0].shape[-1]
        longest = max(p.shape[0] + int(n) for p, n in zip(prefixes, text_lengths))
        inputs = torch.zeros(clips, longest, width, device=device, dtype=prefixes[0].dtype)
        seen = torch.zeros(clips, longest, device=device, dtype=torch.long)          # 1 = a real place, 0 = padding
        ask_at, answer = [], []                                                    # where we ask for the next piece, and the true answer
        for i, prefix in enumerate(prefixes):
            n, p = int(text_lengths[i]), prefix.shape[0]
            ids = text_ids[i, :n]
            inputs[i, :p] = prefix
            inputs[i, p:p + n] = self.piece_vectors(ids)
            seen[i, :p + n] = 1
            # After [now write] the answer is piece 0. After piece k the answer is piece k+1. After the last piece: [end].
            places = torch.arange(p - 1, p + n, device=device)
            ask_at.append(i * longest + places)
            answer.append(torch.cat([ids, torch.tensor([self.end_id], device=device)]))
        hidden = self.writer.model(inputs_embeds=inputs, attention_mask=seen).last_hidden_state      # [clips, longest, 640]
        asked = hidden.reshape(clips * longest, width)[torch.cat(ask_at)]                             # only the places we ask about
        logits = self.writer.lm_head(asked).float()                                                   # [asked places, all pieces of the writer]
        per_place = F.cross_entropy(logits, torch.cat(answer), reduction="none")
        per_clip = torch.stack([part.mean() for part in torch.split(per_place, [int(n) + 1 for n in text_lengths])])
        return per_clip

    @torch.no_grad()
    def write(self, frames, counts, most_pieces=200, pieces_per_frame=0.75, stop_loops=True):
        """The writer writes by itself, always taking its most likely next piece. -> one list of piece numbers for each clip

        A writer can get stuck and repeat one piece for ever. Two guards stop that:
          - a clip may not get more pieces than its sound can hold: pieces_per_frame x its frames, plus 8.
            (People speak 2 to 8 pieces a second, and there are about 13 frames a second, so 0.75 is far above real speech.)
          - if the end of the text is a loop (see is_stuck), the clip is ended and the loop is cut off.
        """
        prefixes = self.rows(frames, counts)
        clips, device, width = len(prefixes), frames.device, prefixes[0].shape[-1]
        longest = max(p.shape[0] for p in prefixes)
        limits = [min(most_pieces, int(pieces_per_frame * int(n)) + 8) for n in counts]
        # Clips have different lengths. We pad on the LEFT, so that every clip ends at the same place and all can write together.
        inputs = torch.zeros(clips, longest, width, device=device, dtype=prefixes[0].dtype)
        seen = torch.zeros(clips, longest, device=device, dtype=torch.long)
        for i, prefix in enumerate(prefixes):
            inputs[i, longest - prefix.shape[0]:] = prefix
            seen[i, longest - prefix.shape[0]:] = 1
        out = self.writer.model(inputs_embeds=inputs, attention_mask=seen, use_cache=True)
        memory, last = out.past_key_values, out.last_hidden_state[:, -1]
        written = [[] for _ in range(clips)]
        done = torch.zeros(clips, dtype=torch.bool, device=device)
        for _ in range(most_pieces):
            choice = self.writer.lm_head(last).argmax(-1)                          # the most likely next piece of each clip
            done |= choice == self.end_id
            for i in range(clips):
                if done[i]:
                    continue
                written[i].append(int(choice[i]))
                if len(written[i]) >= limits[i]:
                    done[i] = True
                elif stop_loops and is_stuck(written[i]):
                    written[i] = cut_loop(written[i])
                    done[i] = True
            if bool(done.all()):
                break
            seen = torch.cat([seen, torch.ones(clips, 1, device=device, dtype=torch.long)], dim=1)
            out = self.writer.model(inputs_embeds=self.piece_vectors(choice)[:, None], attention_mask=seen, past_key_values=memory, use_cache=True)
            memory, last = out.past_key_values, out.last_hidden_state[:, -1]
        return written


def is_stuck(pieces):
    """True if the text ends in a loop: the same piece 6 times, or the same pair of pieces 4 times."""
    if len(pieces) >= 6 and len(set(pieces[-6:])) == 1:
        return True
    return len(pieces) >= 8 and pieces[-8:] == pieces[-2:] * 4


def cut_loop(pieces):
    """Removes the loop at the end, and keeps one copy of what was repeated."""
    unit = pieces[-1:] if len(set(pieces[-6:])) == 1 else pieces[-2:]
    while len(pieces) >= 2 * len(unit) and pieces[-2 * len(unit):] == unit * 2:
        pieces = pieces[:-len(unit)]
    return pieces
