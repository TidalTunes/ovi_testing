# Ovi Mini Demo [FOR CLASS]

This is a tiny, runnable demo of one idea from the Ovi paper: if audio tokens and video tokens live on
different time grids, cross-modal attention gets cleaner when you scale the audio RoPE positions to match
the video timeline.

The real Ovi system is a large joint audio-video diffusion model. This repo does not try to reproduce that.
It just isolates one small mechanism that is easy to understand and easy to run locally.

## What it does

The script builds:

- a short video timeline
- a denser audio timeline
- bidirectional cross-attention between them
- two variants: `unscaled` and `scaled`

The only difference between the two variants is the audio RoPE scale. In practice, the scaled version
produces much better alignment, which is the point of the demo.

## Run it

```bash
python3 run_demo.py
```

You can also change the toy setup:

```bash
python3 run_demo.py --video-tokens 20 --audio-tokens 80 --seed 3 --output-dir artifacts_alt
```

## What you get

After running, look in `/Users/kashi/Desktop/ml_workspace/ovi_testing/artifacts/` for:

- `summary.json` with the metrics
- `report.md` with a short explanation
- SVG attention maps for the scaled and unscaled cases

## Main files

- [run_demo.py](/Users/kashi/Desktop/ml_workspace/ovi_testing/run_demo.py)
- [demo.py](/Users/kashi/Desktop/ml_workspace/ovi_testing/src/ovi_demo/demo.py)

## References

- [Ovi paper](https://arxiv.org/pdf/2510.01284)
- [MMAudio](https://arxiv.org/pdf/2412.15322)
- [UniVerse-1](https://arxiv.org/pdf/2509.06155)
- [JavisDiT](https://arxiv.org/pdf/2503.23377)
