#!/usr/bin/env python3
"""Build session-concatenation based session_pool_v3 (no tiling)."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

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
    p = argparse.ArgumentParser(description="Build session_pool_v3 by session concatenation.")
    p.add_argument("--dataset", type=str, default="vctk", choices=["vctk", "librispeech"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sessions_per_speaker", type=int, default=3)
    p.add_argument("--window_sec", type=float, default=1.0)
    p.add_argument("--hop_sec", type=float, default=0.25)
    p.add_argument("--rms_thr", type=float, default=0.008)
    p.add_argument("--min_windows_per_session", type=int, default=64)
    p.add_argument("--target_session_sec", type=float, default=24.0)
    p.add_argument("--min_gap_sec", type=float, default=0.1)
    p.add_argument("--max_gap_sec", type=float, default=0.3)
    p.add_argument("--max_build_tries", type=int, default=8)
    return p.parse_args()


def build_one_session(
    rows: List[Dict[str, str]],
    wav_root: Path,
    rng: random.Random,
    target_session_sec: float,
    min_gap_sec: float,
    max_gap_sec: float,
) -> Tuple[np.ndarray, List[str]]:
    unique_paths = sorted({r["path"] for r in rows})
    if len(unique_paths) == 0:
        return np.zeros(1, dtype=np.float32), []
    segs: List[np.ndarray] = []
    used_paths: List[str] = []
    cur_sec = 0.0
    while cur_sec < target_session_sec:
        n_pick = min(len(unique_paths), max(3, int(target_session_sec // 3)))
        picks = rng.sample(unique_paths, k=n_pick) if len(unique_paths) >= n_pick else [rng.choice(unique_paths) for _ in range(n_pick)]
        for p in picks:
            x = read_wav(wav_root / p)
            segs.append(x)
            used_paths.append(p)
            cur_sec += len(x) / 16000.0
            gap = rng.uniform(min_gap_sec, max_gap_sec)
            segs.append(np.zeros(int(round(gap * 16000)), dtype=np.float32))
            cur_sec += gap
            if cur_sec >= target_session_sec:
                break
    if not segs:
        return np.zeros(1, dtype=np.float32), []
    y = np.concatenate(segs, axis=0).astype(np.float32)
    return y, used_paths


def extract_windows(
    wav: np.ndarray,
    window_sec: float,
    hop_sec: float,
    rms_thr: float,
) -> List[Tuple[int, int]]:
    win = int(round(window_sec * 16000))
    hop = int(round(hop_sec * 16000))
    if len(wav) < win:
        return []
    out: List[Tuple[int, int]] = []
    for s in range(0, len(wav) - win + 1, hop):
        seg = wav[s : s + win]
        if rms(seg) < rms_thr:
            continue
        out.append((s, win))
    return out


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))

    split_dir = REPO_ROOT / "data" / "splits" / args.dataset
    wav_root = REPO_ROOT / "data" / "processed" / args.dataset / "wav16k"
    out_split_dir = REPO_ROOT / "data" / "splits"
    out_split_dir.mkdir(parents=True, exist_ok=True)
    out_tables = REPO_ROOT / "artifacts" / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_sess_root = REPO_ROOT / "data" / "processed" / "sessions_v3"
    out_sess_root.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "split",
        "speaker_id",
        "session_id",
        "chunk_id",
        "path",
        "start_sec",
        "dur_sec",
        "source_utter_ids",
    ]

    stats_lines: List[str] = []
    for split in ["train", "val", "test"]:
        rows = list(csv.DictReader((split_dir / f"{split}.csv").open("r", encoding="utf-8")))
        by_spk: Dict[str, List[Dict[str, str]]] = {}
        for r in rows:
            by_spk.setdefault(r["speaker_id"], []).append(r)
        split_rows: List[Dict[str, str]] = []
        session_window_counts: List[int] = []
        split_sess_dir = out_sess_root / split
        split_sess_dir.mkdir(parents=True, exist_ok=True)

        for spk in sorted(by_spk.keys()):
            spk_rows = by_spk[spk]
            for sid in range(int(args.sessions_per_speaker)):
                accepted = False
                wav = np.zeros(1, dtype=np.float32)
                used_paths: List[str] = []
                windows: List[Tuple[int, int]] = []
                for _ in range(int(args.max_build_tries)):
                    wav, used_paths = build_one_session(
                        rows=spk_rows,
                        wav_root=wav_root,
                        rng=rng,
                        target_session_sec=float(args.target_session_sec),
                        min_gap_sec=float(args.min_gap_sec),
                        max_gap_sec=float(args.max_gap_sec),
                    )
                    windows = extract_windows(
                        wav=wav,
                        window_sec=float(args.window_sec),
                        hop_sec=float(args.hop_sec),
                        rms_thr=float(args.rms_thr),
                    )
                    if len(windows) >= int(args.min_windows_per_session):
                        accepted = True
                        break
                if not accepted:
                    continue

                session_id = f"{spk}_s{sid:03d}"
                rel_path = Path("data") / "processed" / "sessions_v3" / split / f"session_{session_id}.wav"
                abs_path = REPO_ROOT / rel_path
                sf.write(str(abs_path), wav, 16000)

                for i, (s, d) in enumerate(windows):
                    split_rows.append(
                        {
                            "split": split,
                            "speaker_id": spk,
                            "session_id": session_id,
                            "chunk_id": f"{session_id}_{i:04d}",
                            "path": str(rel_path),
                            "start_sec": f"{s / 16000.0:.3f}",
                            "dur_sec": f"{d / 16000.0:.3f}",
                            "source_utter_ids": json.dumps(used_paths, ensure_ascii=False),
                        }
                    )
                session_window_counts.append(len(windows))

        out_csv = out_split_dir / f"session_pool_v3_{split}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(split_rows)

        n_sessions = len({r["session_id"] for r in split_rows})
        if session_window_counts:
            smin = int(np.min(session_window_counts))
            smax = int(np.max(session_window_counts))
            smean = float(np.mean(session_window_counts))
        else:
            smin, smax, smean = 0, 0, 0.0
        line = (
            f"[{split}] sessions={n_sessions} windows_total={len(split_rows)} "
            f"windows/session(min,mean,max)=({smin},{smean:.2f},{smax})"
        )
        print(line)
        stats_lines.append(line)

    (out_tables / "session_pool_v3_stats.txt").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
