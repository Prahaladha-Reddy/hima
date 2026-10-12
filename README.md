# Hima

A small open model that listens. Speech in, text out, for 11 Indian languages and English.
Later it will also give decisions from the same sound: is the speaker done, should the agent stop.

**Status: the design is set. No model is trained yet, and there is no checkpoint.**

## The model

| Part | What it does | Where it comes from |
|---|---|---|
| Ears | turn sound into frames | the audio part of Qwen3-ASR-0.6B |
| Bridge | fit the frames to the writer | new, small |
| Writer | write the text | Gemma 3 270M |
| Decision head | answer fixed questions from the frames | later |

Languages: Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, English.

## Where things live

- **This repo:** code, configs, notes.
- **Hugging Face ([Prahaladha/hima](https://huggingface.co/Prahaladha/hima)):** checkpoints, pushed as the model trains.

## Layout

```
src/        model and training code
configs/    one file per run
scripts/    launch scripts
docs/       decisions and how-to notes
notes/      run logs, one file per run
```

## Rules for a run

1. A small test first. Then the full run.
2. Every run writes its settings and its code version next to its checkpoints.
3. A checkpoint on Hugging Face names the commit of this repo that made it.

## Licence

The code is under the Apache 2.0 licence. Each checkpoint states its own licence, because the parts it starts from have their own terms.
