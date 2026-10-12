# Decisions

| Date | Decision | Why |
|---|---|---|
| 12 Oct 2026 | The writer is Gemma 3 270M | the aim is a small model |
| 13 Oct 2026 | The ears start from the audio part of Qwen3-ASR-1.7B | the whole model still stays near 600 million weights |
| 13 Oct 2026 | Ears, bridge and writer are trained together; the main loss is the writer loss | the same family of objective as the first stage of the ears' makers |
| 13 Oct 2026 | A vote loss is added as a helper, with weight 0.3 | in a small test the writer was 9 points better with it (61% against 69% letters wrong, very early in training) |
| 12 Oct 2026 | The model is named Hima | short and easy to say |

## Open

- The form of the text: numbers as words or digits, English words in which script, marks or no marks.
- The decision head: after the model writes text well.
