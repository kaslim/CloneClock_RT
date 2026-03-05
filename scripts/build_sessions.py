#!/usr/bin/env python3
"""Build call-session wavs from short test clips."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf


def get_data_paths(repo_root: Path, dataset: str) -> Dict[str, Path]:
    return {
        "processed_dir": repo_root / "data" / "processed" / dataset,
        "splits_dir": repo_root / "data" / "splits" / dataset,
    }


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32), int(sr)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Build 30s call sessions from test split")
    p.add_argument("--dataset", type=str, choices=["librispeech", "vctk"], default="vctk")
    p.add_argument("--session_len_sec", type=float, default=30.0)
    p.add_argument("--sessions_per_speaker", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_speakers", type=int, default=0, help="0 means all speakers")
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(repo / "data" / "processed" / "sessions"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    repo = Path(__file__).resolve().parents[1]
    paths = get_data_paths(repo, args.dataset)
    split_csv = paths["splits_dir"] / "test.csv"
    wav_root = paths["processed_dir"] / "wav16k"
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing split file: {split_csv}")

    rows: List[Dict[str, str]] = []
    with open(split_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("No rows in test split.")

    by_spk: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_spk[r["speaker_id"]].append(r)

    speakers = sorted(by_spk.keys())
    if args.max_speakers > 0:
        speakers = speakers[: args.max_speakers]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "sessions.csv"

    target_len = float(args.session_len_sec)
    session_rows: List[Dict[str, str]] = []

    for spk in speakers:
        clips = by_spk[spk]
        if not clips:
            continue
        for idx in range(args.sessions_per_speaker):
            chosen = []
            total_dur = 0.0
            wav_parts: List[np.ndarray] = []
            sr_ref = None

            guard = 0
            while total_dur < target_len and guard < 2000:
                guard += 1
                c = rng.choice(clips)
                wav_path = wav_root / c["path"]
                if not wav_path.exists():
                    continue
                wav, sr = read_wav_mono(wav_path)
                dur = float(len(wav) / sr)
                if sr_ref is None:
                    sr_ref = sr
                if sr != sr_ref:
                    continue
                wav_parts.append(wav)
                chosen.append(c["path"])
                total_dur += dur

            if not wav_parts:
                continue
            full = np.concatenate(wav_parts, axis=0).astype(np.float32)
            max_len = int(target_len * sr_ref)
            if len(full) > max_len:
                full = full[:max_len]
                total_dur = target_len

            session_id = f"session_{spk}_{idx:03d}"
            out_wav = out_dir / f"{session_id}.wav"
            sf.write(out_wav, full, sr_ref)
            session_rows.append(
                {
                    "session_id": session_id,
                    "speaker_id": spk,
                    "path": str(out_wav),
                    "n_segments": str(len(chosen)),
                    "total_duration": f"{total_dur:.3f}",
                    "segment_list": json.dumps(chosen, ensure_ascii=False),
                }
            )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["session_id", "speaker_id", "path", "n_segments", "total_duration", "segment_list"],
        )
        w.writeheader()
        w.writerows(session_rows)

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "sessions": len(session_rows),
                "session_len_sec": args.session_len_sec,
                "output_csv": str(out_csv),
                "output_dir": str(out_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
