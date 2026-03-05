#!/usr/bin/env python3
"""Build session-consistent window pool from split CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_wav(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"Expected 16k wav, got {sr}: {path}")
    return x


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-10))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build session pool index from split CSV.")
    p.add_argument("--dataset", type=str, default="vctk", choices=["vctk", "librispeech"])
    p.add_argument("--win_sec", type=float, default=1.5)
    p.add_argument("--hop_sec", type=float, default=0.75)
    p.add_argument("--rms_thr", type=float, default=0.008)
    p.add_argument("--min_windows_per_utter", type=int, default=6)
    p.add_argument("--output_prefix", type=str, default="session_pool")
    p.add_argument("--tile_to_min_windows", action="store_true")
    return p.parse_args()


def build_for_split(
    split_name: str,
    rows: List[Dict[str, str]],
    wav_root: Path,
    win_sec: float,
    hop_sec: float,
    rms_thr: float,
    min_windows: int,
    tile_to_min_windows: bool,
) -> List[Dict[str, str]]:
    out_rows: List[Dict[str, str]] = []
    win = int(round(win_sec * 16000))
    hop = int(round(hop_sec * 16000))
    for row in rows:
        wav_path = wav_root / row["path"]
        x = read_wav(wav_path)
        if len(x) < win:
            continue
        if tile_to_min_windows:
            min_len_needed = win + max(0, min_windows - 1) * hop
            if len(x) < min_len_needed:
                rep = int(np.ceil(float(min_len_needed) / max(1, len(x))))
                x = np.tile(x, rep)
        candidates = []
        chunk_idx = 0
        for s in range(0, len(x) - win + 1, hop):
            seg = x[s : s + win]
            if rms(seg) < rms_thr:
                continue
            candidates.append(
                {
                    "split": split_name,
                    "speaker_id": row["speaker_id"],
                    "utter_id": row["utter_id"],
                    "chunk_id": f"{row['utter_id']}_{chunk_idx:03d}",
                    "path": row["path"],
                    "start_sec": f"{s / 16000.0:.3f}",
                    "dur_sec": f"{win / 16000.0:.3f}",
                }
            )
            chunk_idx += 1
        if len(candidates) >= min_windows:
            out_rows.extend(candidates)
    return out_rows


def split_stats(rows: List[Dict[str, str]]) -> Dict[str, float]:
    speakers = {r["speaker_id"] for r in rows}
    utter_to_n: Dict[str, int] = {}
    for r in rows:
        k = f"{r['speaker_id']}::{r['utter_id']}"
        utter_to_n[k] = utter_to_n.get(k, 0) + 1
    vals = list(utter_to_n.values())
    avg_n = float(np.mean(vals)) if vals else 0.0
    min_n = float(np.min(vals)) if vals else 0.0
    max_n = float(np.max(vals)) if vals else 0.0
    return {
        "speakers": float(len(speakers)),
        "utters": float(len(utter_to_n)),
        "windows": float(len(rows)),
        "avg_windows_per_utter": avg_n,
        "min_windows_per_utter": min_n,
        "max_windows_per_utter": max_n,
    }


def main() -> int:
    args = parse_args()
    split_dir = REPO_ROOT / "data" / "splits" / args.dataset
    wav_root = REPO_ROOT / "data" / "processed" / args.dataset / "wav16k"
    out_split_dir = REPO_ROOT / "data" / "splits"
    out_split_dir.mkdir(parents=True, exist_ok=True)
    out_art_dir = REPO_ROOT / "artifacts" / "tables"
    out_art_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    by_split: Dict[str, List[Dict[str, str]]] = {}
    for split in ["train", "val", "test"]:
        with open(split_dir / f"{split}.csv", "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        built = build_for_split(
            split_name=split,
            rows=rows,
            wav_root=wav_root,
            win_sec=args.win_sec,
            hop_sec=args.hop_sec,
            rms_thr=args.rms_thr,
            min_windows=args.min_windows_per_utter,
            tile_to_min_windows=bool(args.tile_to_min_windows),
        )
        by_split[split] = built
        all_rows.extend(built)

    fieldnames = ["split", "speaker_id", "utter_id", "chunk_id", "path", "start_sec", "dur_sec"]
    prefix = str(args.output_prefix).strip()
    if not prefix:
        prefix = "session_pool"
    with open(out_art_dir / f"{prefix}_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    for split in ["train", "val", "test"]:
        with open(out_split_dir / f"{prefix}_{split}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(by_split[split])

    stats_lines = []
    for split in ["train", "val", "test"]:
        st = split_stats(by_split[split])
        line = (
            f"[{split}] speakers={int(st['speakers'])} utters={int(st['utters'])} "
            f"windows={int(st['windows'])} avg_windows/utter={st['avg_windows_per_utter']:.2f} "
            f"min_windows/utter={int(st['min_windows_per_utter'])} "
            f"max_windows/utter={int(st['max_windows_per_utter'])}"
        )
        stats_lines.append(line)
        print(line)
    with open(out_art_dir / f"{prefix}_stats.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
