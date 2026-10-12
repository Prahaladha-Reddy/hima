"""Makes the table of pieces for the vote layer.

Reads the text of some shards, cuts it with the writer's tokenizer, and keeps every piece that
appears at least `min_count` times (at most `max_pieces`, the most frequent first).

usage: python scripts/build_pieces.py <shard list> <tables.yaml> <out pieces.json> [tokenizer] [shards per language] [min_count] [max_pieces]
"""
import collections
import random
import sys

import yaml

sys.path.insert(0, "src")
from hima.data import load_shard_list, read_table            # noqa: E402
from hima.text import PieceTable                              # noqa: E402

shard_list, tables_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
tokenizer_name = sys.argv[4] if len(sys.argv) > 4 else "unsloth/gemma-3-270m"
per_lang = int(sys.argv[5]) if len(sys.argv) > 5 else 3
min_count = int(sys.argv[6]) if len(sys.argv) > 6 else 5
max_pieces = int(sys.argv[7]) if len(sys.argv) > 7 else 24000

specs = yaml.safe_load(open(tables_path))
shards = load_shard_list(shard_list)
random.Random(0).shuffle(shards)
table = PieceTable(tokenizer_name, [])
counts, taken, clips = collections.Counter(), collections.Counter(), 0
for shard in shards:
    if taken[(shard["lang"], shard["kind"])] >= per_lang:
        continue
    taken[(shard["lang"], shard["kind"])] += 1
    rows, _ = read_table(shard["table"], specs[shard["kind"]])
    for text, _, _ in rows.values():
        counts.update(table.tokenizer(text, add_special_tokens=False)["input_ids"])
    clips += len(rows)
    print("read", shard["lang"], shard["kind"], len(rows), "clips | pieces so far", len(counts), flush=True)
kept = [p for p, n in counts.most_common(max_pieces) if n >= min_count]
covered = sum(counts[p] for p in kept) / max(1, sum(counts.values()))
PieceTable(tokenizer_name, kept).save(out)
print("PIECES_DONE clips %d | different pieces seen %d | kept %d | they cover %.2f%% of all pieces in the text" % (clips, len(counts), len(kept), 100 * covered))
