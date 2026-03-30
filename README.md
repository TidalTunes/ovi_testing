# Ovi Mini Demo

This repo contains a very small, dependency-free demo inspired by one core mechanism from the Ovi
paper: symmetric audio/video branches connected by bidirectional cross-attention, with scaled RoPE
used to reconcile different temporal resolutions.

It does **not** attempt to reproduce the full model. Ovi is an 11B joint audio-video diffusion system
built on paired DiT backbones, latent audio/video autoencoders, a shared T5 text encoder, and a
large-scale two-stage training recipe. This demo isolates the part that is easiest to test in a tiny
workspace: temporal alignment inside cross-modal attention.

## Research Notes

Reading Ovi and the adjacent papers suggests a useful split in the open literature:

- Ovi pushes toward true one-pass audio-video generation with symmetric twin backbones and
  blockwise bidirectional fusion. Its most demo-friendly idea is scaling audio RoPE so audio and
  video positions live on a compatible temporal grid.
- MMAudio is a strong public baseline for **video-to-audio** rather than full joint generation. It uses
  multimodal joint training plus a dedicated synchronization module.
- UniVerse-1 reaches joint generation by stitching together pretrained video and music experts with
  lightweight cross-modal connectors instead of training a symmetric audio tower from scratch.
- JavisDiT also targets joint generation, but emphasizes explicit hierarchical spatio-temporal priors
  for synchronization rather than Ovi's simpler symmetric-fusion recipe.

That makes scaled positional alignment a good target for a tiny reproduction: it is central to Ovi,
easy to isolate, and different from the more auxiliary-sync designs in neighboring work.

## What The Demo Does

The demo builds two token timelines:

- video: 16 tokens
- audio: 64 tokens

Each timeline carries the same synthetic sequence of events (`speech`, `drum`, `bird`, `engine`, ...)
at different temporal resolutions. Cross-attention uses only rotary position encoding and token values:

- queries/keys are the same base vector rotated to their modality-specific positions
- values contain the event labels that should be recovered from the opposite modality

This means the experiment directly tests whether positional geometry alone is sufficient to line up the
two streams.

Two variants are compared:

1. `unscaled`: audio positions use their native indices
2. `scaled`: audio positions are multiplied by `video_tokens / audio_tokens`

If Ovi's claim is correct, the scaled version should produce a much cleaner diagonal in the
cross-attention map and much lower retrieval error.

## Files

- `/Users/kashi/Desktop/ml_workspace/ovi_testing/run_demo.py`: CLI entry point
- `/Users/kashi/Desktop/ml_workspace/ovi_testing/src/ovi_demo/demo.py`: RoPE, cross-attention, metrics, and SVG generation
- `/Users/kashi/Desktop/ml_workspace/ovi_testing/artifacts/`: generated outputs after running the demo

## Run

```bash
python3 run_demo.py
```

Optional flags:

```bash
python3 run_demo.py --video-tokens 20 --audio-tokens 80 --seed 3 --output-dir artifacts_alt
```

## Expected Outputs

Running the script writes:

- `artifacts/summary.json`
- `artifacts/report.md`
- `artifacts/unscaled_video_to_audio.svg`
- `artifacts/unscaled_audio_to_video.svg`
- `artifacts/scaled_video_to_audio.svg`
- `artifacts/scaled_audio_to_video.svg`

## Sources

- Ovi paper: [arXiv PDF](https://arxiv.org/pdf/2510.01284)
- Ovi project page: [aaxwaz.github.io/Ovi](https://aaxwaz.github.io/Ovi/)
- MMAudio paper: [arXiv PDF](https://arxiv.org/pdf/2412.15322)
- UniVerse-1 paper: [arXiv PDF](https://arxiv.org/pdf/2509.06155)
- JavisDiT paper: [arXiv PDF](https://arxiv.org/pdf/2503.23377)
