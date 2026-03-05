#!/usr/bin/env python3
"""Run telephony transform demo on one wav file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import soundfile as sf
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transforms.telephony import TelephonyTransform


def pick_default_input(repo_root: Path) -> Path:
    candidates = [
        repo_root / "data" / "processed" / "vctk" / "wav16k",
        repo_root / "data" / "processed" / "librispeech" / "wav16k",
        repo_root / "data" / "processed" / "wav16k",
    ]
    for base in candidates:
        if base.exists():
            wavs = sorted(base.rglob("*.wav"))
            if wavs:
                return wavs[0]
    raise FileNotFoundError("No input wav found in processed directories.")


def main() -> int:
    repo = REPO_ROOT
    parser = argparse.ArgumentParser(description="Telephony transform demo")
    parser.add_argument("--input_wav", type=str, default=None, help="Input wav path")
    parser.add_argument("--config", type=str, default=str(repo / "configs" / "telephony.yaml"))
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(repo / "artifacts" / "figures" / "demo_audio"),
        help="Output directory for demo audio files",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    telephony_cfg = cfg.get("telephony", cfg)
    seed = args.seed if args.seed is not None else int(telephony_cfg.get("seed", 42))

    inp = Path(args.input_wav) if args.input_wav else pick_default_input(repo)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = sf.read(inp, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    transform = TelephonyTransform(telephony_cfg, seed=seed)
    telephony_audio = transform(audio, sample_rate=sr)

    clean_out = out_dir / f"{inp.stem}_clean.wav"
    tele_out = out_dir / f"{inp.stem}_telephony.wav"
    sf.write(clean_out, audio, sr, subtype="PCM_16")
    sf.write(tele_out, telephony_audio, sr, subtype="PCM_16")

    print(f"[telephony_demo] input: {inp}")
    print(f"[telephony_demo] clean: {clean_out}")
    print(f"[telephony_demo] telephony: {tele_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
