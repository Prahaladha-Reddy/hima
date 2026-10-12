"""Makes the test pack: clips whose key is held out, a fixed number for each language, saved as plain files.

It reads shards with the same reader as training, but keeps ONLY the held-out clips (training keeps only the others).

usage: python scripts/build_test_pack.py <config.yaml> <shard list> <clips per language> [most shards per language] [kinds, e.g. b] [one language]

Give the kinds of shards whose text you trust enough to score against. A shard may hold no held-out clip at all
(a shard with a few long talks, for example), so the script moves on after it has read `most shards`.
"""
import collections
import json
import os
import sys

import soundfile as sf
import yaml

sys.path.insert(0, "src")
from hima.data import ClipStream, load_shard_list            # noqa: E402
from hima.text import PieceTable                              # noqa: E402

cfg = yaml.safe_load(open(sys.argv[1]))
shards = load_shard_list(sys.argv[2])
per_lang = int(sys.argv[3])
most_shards = int(sys.argv[4]) if len(sys.argv) > 4 else 8
kinds = sys.argv[5].split(",") if len(sys.argv) > 5 else None
only_lang = sys.argv[6] if len(sys.argv) > 6 else None       # to build the languages side by side: one process for each, then join the lists
specs = yaml.safe_load(open(cfg["data"]["tables"]))
out = cfg["test_pack"]
os.makedirs(out + "/audio", exist_ok=True)
pieces = PieceTable.load(cfg["pieces"])
reader = ClipStream(shards, specs, cfg["pieces"], cfg["model_dir"], cfg["data"], train=False)
have, used = collections.Counter(), collections.Counter()
counts = collections.Counter()                          # why clips were dropped; a Counter accepts any reason
with open(out + ("/clips_%s.jsonl" % only_lang if only_lang else "/clips.jsonl"), "w", encoding="utf-8") as listing:
    for shard in sorted(shards, key=lambda s: (s["lang"], s["kind"], s["audio"].split("?")[0])):
        lang = shard["lang"]
        if (kinds and shard["kind"] not in kinds) or (only_lang and lang != only_lang):
            continue
        if have[lang] >= per_lang or used[lang] >= most_shards:
            continue
        used[lang] += 1
        for clip in reader.clips_of_shard(shard, pieces, counts):
            name = "%s_%05d.flac" % (lang, have[lang])
            sf.write("%s/audio/%s" % (out, name), clip["sound"], 16000)
            listing.write(json.dumps({"file": name, "lang": lang, "kind": shard["kind"], "tag": shard.get("tag", ""), "seconds": round(clip["seconds"], 2), "text": clip["text"]}, ensure_ascii=False) + "\n")
            have[lang] += 1
            if have[lang] >= per_lang:
                break
        print(lang, shard["kind"], "clips so far", have[lang], flush=True)
print("TEST_PACK_DONE", dict(have))
