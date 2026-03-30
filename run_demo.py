from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ovi_demo.demo import DemoConfig, run_demo


def parse_args() -> DemoConfig:
    parser = argparse.ArgumentParser(
        description="Run a tiny dependency-free demo of Ovi-style audio/video temporal alignment."
    )
    parser.add_argument("--video-tokens", type=int, default=16)
    parser.add_argument("--audio-tokens", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--theta", type=float, default=10_000.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    return DemoConfig(
        video_tokens=args.video_tokens,
        audio_tokens=args.audio_tokens,
        model_dim=args.model_dim,
        temperature=args.temperature,
        theta=args.theta,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def main() -> None:
    config = parse_args()
    summary = run_demo(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

