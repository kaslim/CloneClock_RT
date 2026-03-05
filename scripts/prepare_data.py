#!/usr/bin/env python3
"""
prepare_data.py — 生成「电话诈骗采样风格」短片段（2–6s），按 speaker 划分 train/val/test。
支持 VCTK / LibriSpeech（优先 VCTK；下载困难时用 LibriSpeech dev-clean）。
可复现：固定随机种子。
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

# 固定种子以保证可复现
DEFAULT_SEED = 42
TARGET_SR = 16000
MIN_DUR = 2.0
MAX_DUR = 6.0
TARGET_DURATION_PER_SPEAKER = 300.0  # ~5 min per speaker


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def discover_librispeech(raw_dir: Path) -> list[tuple[str, Path]]:
    """(speaker_id, wav_path) 列表。LibriSpeech 为 flac，结构: dev-clean/<speaker>/<chapter>/*.flac"""
    raw_dir = Path(raw_dir)
    dev_clean = raw_dir / "LibriSpeech" / "dev-clean"
    if not dev_clean.exists():
        dev_clean = raw_dir / "dev-clean"
    if not dev_clean.exists():
        raise FileNotFoundError(f"未找到 LibriSpeech dev-clean 目录: {dev_clean}")
    out = []
    for speaker_dir in sorted(dev_clean.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name
        for chapter_dir in speaker_dir.iterdir():
            if not chapter_dir.is_dir():
                continue
            for ext in ("*.flac", "*.wav"):
                for f in chapter_dir.glob(ext):
                    out.append((speaker_id, f))
    return out


def create_demo_librispeech(raw_dir: Path, num_speakers: int, sec_per_speaker: float, sr: int, seed: int) -> None:
    """生成与 LibriSpeech dev-clean 同结构的 demo 数据（wav），用于无真实数据时跑通流程。"""
    _set_seed(seed)
    rng = random.Random(seed)
    raw_dir = Path(raw_dir)
    base = raw_dir / "LibriSpeech" / "dev-clean"
    base.mkdir(parents=True, exist_ok=True)
    for i in range(num_speakers):
        sp_id = str(3000 + i)
        ch_id = str(100000 + i)
        sp_dir = base / sp_id / ch_id
        sp_dir.mkdir(parents=True, exist_ok=True)
        remaining = sec_per_speaker
        fi = 0
        while remaining > 1.0:
            dur = min(rng.uniform(3.0, 10.0), remaining)
            n = int(dur * sr)
            y = np.sin(2 * np.pi * 440 * np.arange(n) / sr).astype(np.float32) * 0.3
            path = sp_dir / f"{sp_id}-{ch_id}-{fi:04d}.wav"
            sf.write(path, y, sr, subtype="PCM_16")
            remaining -= dur
            fi += 1
    print(f"[prepare_data] 已生成 demo 数据: {num_speakers} speakers, ~{num_speakers * sec_per_speaker:.0f}s, 位于 {base}")

def discover_vctk(raw_dir: Path) -> list[tuple[str, Path]]:
    """(speaker_id, wav_path) 列表。支持: VCTK-Corpus/wav48, VCTK-Corpus-0.92/wav48_silence_trimmed 或 wav48（含 .wav 或 .flac）"""
    raw_dir = Path(raw_dir)
    candidates = [
        raw_dir / "VCTK-Corpus" / "wav48",
        raw_dir / "VCTK-Corpus-0.92" / "wav48_silence_trimmed",
        raw_dir / "VCTK-Corpus-0.92" / "wav48",
        raw_dir / "wav48",
    ]
    wav48 = None
    for c in candidates:
        if c.exists():
            wav48 = c
            break
    if wav48 is None:
        raise FileNotFoundError(f"未找到 VCTK 音频目录，已尝试: {candidates}")
    out = []
    for speaker_dir in sorted(wav48.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name
        for ext in ("*.wav", "*.flac"):
            for f in sorted(speaker_dir.glob(ext)):
                out.append((speaker_id, f))
    return out


def load_audio(path: Path, sr: int) -> tuple[Any, int]:
    """返回 (y, sr)。若原始 sr 与目标不同则重采样。"""
    y, orig_sr = librosa.load(path, sr=None, mono=True)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    return y, sr


def write_wav(path: Path, y, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, sr, subtype="PCM_16")


def slice_speaker_segments(
    speaker_id: str,
    file_list: list[Path],
    out_root: Path,
    target_duration: float,
    min_dur: float,
    max_dur: float,
    sr: int,
    rng: random.Random,
) -> list[tuple[str, str, float]]:
    """
    对该 speaker 的所有文件随机裁 2–6s 片段，直到总时长约 target_duration。
    返回 [(utter_id, relative_path, duration), ...]
    """
    # 先收集所有文件的时长
    file_durations: list[tuple[Path, float]] = []
    for f in file_list:
        try:
            y, _ = load_audio(f, sr)
            dur = len(y) / sr
            if dur >= min_dur:
                file_durations.append((f, dur))
        except Exception as e:
            print(f"  [skip] {f}: {e}")
            continue

    if not file_durations:
        return []

    segments: list[tuple[str, str, float]] = []
    total_dur = 0.0
    seg_idx = 0

    while total_dur < target_duration:
        # 随机选一个文件
        f, fdur = rng.choice(file_durations)
        seg_len = rng.uniform(min_dur, max_dur)
        if fdur < seg_len:
            continue
        start = rng.uniform(0.0, fdur - seg_len)
        try:
            y, _ = load_audio(f, sr)
            st_sample = int(start * sr)
            end_sample = int((start + seg_len) * sr)
            clip = y[st_sample:end_sample]
            utter_id = f"{speaker_id}_{seg_idx:04d}"
            rel_path = f"{speaker_id}/{utter_id}.wav"
            out_path = out_root / rel_path
            write_wav(out_path, clip, sr)
            actual_dur = len(clip) / sr
            segments.append((utter_id, rel_path, actual_dur))
            total_dur += actual_dur
            seg_idx += 1
        except Exception as e:
            print(f"  [slice skip] {f}: {e}")
            continue

    return segments


def run_prepare(
    dataset: str,
    raw_dir: Path,
    out_wav_dir: Path,
    splits_dir: Path,
    target_duration_per_speaker: float,
    seed: int,
) -> None:
    _set_seed(seed)
    rng = random.Random(seed)
    raw_dir = Path(raw_dir)
    out_wav_dir = Path(out_wav_dir)
    splits_dir = Path(splits_dir)
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "librispeech":
        pairs = discover_librispeech(raw_dir)
    elif dataset == "vctk":
        pairs = discover_vctk(raw_dir)
    else:
        raise ValueError(f"未知 dataset: {dataset}，支持 librispeech | vctk")

    # 按 speaker 聚合文件
    from collections import defaultdict
    speaker_files: dict[str, list[Path]] = defaultdict(list)
    for sid, path in pairs:
        speaker_files[sid].append(path)

    print(f"[prepare_data] 共 {len(speaker_files)} 个 speaker，原始文件数 {len(pairs)}")

    all_rows: list[dict] = []
    for speaker_id in sorted(speaker_files.keys()):
        segs = slice_speaker_segments(
            speaker_id,
            speaker_files[speaker_id],
            out_wav_dir,
            target_duration_per_speaker,
            MIN_DUR,
            MAX_DUR,
            TARGET_SR,
            rng,
        )
        for utter_id, rel_path, dur in segs:
            all_rows.append({
                "utter_id": utter_id,
                "speaker_id": speaker_id,
                "path": rel_path,
                "duration": f"{dur:.3f}",
            })

    # metadata.csv（与 wav16k 同级的 processed/<dataset>/ 下）
    meta_path = out_wav_dir.parent / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utter_id", "speaker_id", "path", "duration"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"[prepare_data] 已写 {meta_path}，片段数 {len(all_rows)}")

    # Speaker-disjoint 80/10/10
    speakers = sorted(speaker_files.keys())
    rng.shuffle(speakers)
    n = len(speakers)
    n_train = int(round(n * 0.8))
    n_val = int(round(n * 0.1))
    n_test = n - n_train - n_val
    train_sp = set(speakers[:n_train])
    val_sp = set(speakers[n_train : n_train + n_val])
    test_sp = set(speakers[n_train + n_val :])

    def split_rows(sp_set):
        return [r for r in all_rows if r["speaker_id"] in sp_set]

    train_rows = split_rows(train_sp)
    val_rows = split_rows(val_sp)
    test_rows = split_rows(test_sp)

    for name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        path = splits_dir / f"{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["utter_id", "speaker_id", "path", "duration"])
            w.writeheader()
            w.writerows(rows)
        print(f"[prepare_data] 已写 {path} ({len(rows)} 条)")

    # 统计表
    def sum_dur(rows):
        return sum(float(r["duration"]) for r in rows)

    stats = [
        {"split": "train", "speakers": len(train_sp), "utterances": len(train_rows), "duration_sec": f"{sum_dur(train_rows):.2f}"},
        {"split": "val",   "speakers": len(val_sp),   "utterances": len(val_rows),   "duration_sec": f"{sum_dur(val_rows):.2f}"},
        {"split": "test",  "speakers": len(test_sp),  "utterances": len(test_rows),  "duration_sec": f"{sum_dur(test_rows):.2f}"},
    ]
    stats_path = splits_dir / "split_stats.csv"
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "speakers", "utterances", "duration_sec"])
        w.writeheader()
        w.writerows(stats)
    print(f"[prepare_data] 已写 {stats_path}")

    # 控制台汇总（用于回传）
    total_dur = sum_dur(all_rows)
    print("\n========== 数据规模 ==========")
    print(f"  speaker 数:     {len(speakers)}")
    print(f"  片段数:        {len(all_rows)}")
    print(f"  总时长 (s):    {total_dur:.2f}")
    print("\n========== split 统计表 ==========")
    for s in stats:
        print(f"  {s['split']:5s}  speakers={s['speakers']:4d}  utterances={s['utterances']:5d}  duration_sec={s['duration_sec']}")


def main():
    p = argparse.ArgumentParser(description="准备电话风格短片段并做 speaker-disjoint 划分")
    p.add_argument("--dataset", choices=["vctk", "librispeech"], default="librispeech",
                   help="数据源，优先 vctk；下载困难时用 librispeech")
    p.add_argument("--raw_dir", type=str, default=None,
                   help="原始数据根目录，默认 <repo>/data/raw")
    p.add_argument("--out_dir", type=str, default=None,
                   help="处理后根目录，默认 <repo>/data/processed")
    p.add_argument("--splits_dir", type=str, default=None,
                   help="划分 csv 目录，默认 <repo>/data/splits")
    p.add_argument("--target_duration_per_speaker", type=float, default=TARGET_DURATION_PER_SPEAKER,
                   help=f"每 speaker 目标总时长（秒），默认 {TARGET_DURATION_PER_SPEAKER}")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    p.add_argument("--demo", action="store_true", help="无真实数据时生成 demo 数据并跑通流程")
    p.add_argument("--demo_speakers", type=int, default=30, help="demo 时生成的 speaker 数")
    p.add_argument("--demo_sec_per_speaker", type=float, default=60.0, help="demo 时每 speaker 约多少秒原始音频")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    raw_dir = Path(args.raw_dir) if args.raw_dir else repo / "data" / "raw"
    if args.demo:
        create_demo_librispeech(raw_dir, args.demo_speakers, args.demo_sec_per_speaker, TARGET_SR, args.seed)
        args.dataset = "librispeech"
        args.target_duration_per_speaker = min(args.target_duration_per_speaker, 60.0)
    # 按 dataset 隔离：不覆盖，实验时用 --dataset 选择
    out_base = Path(args.out_dir) if args.out_dir else repo / "data" / "processed"
    out_wav_dir = out_base / args.dataset / "wav16k"
    splits_root = Path(args.splits_dir) if args.splits_dir else repo / "data" / "splits"
    splits_dir = splits_root / args.dataset

    run_prepare(
        dataset=args.dataset,
        raw_dir=raw_dir,
        out_wav_dir=out_wav_dir,
        splits_dir=splits_dir,
        target_duration_per_speaker=args.target_duration_per_speaker,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
