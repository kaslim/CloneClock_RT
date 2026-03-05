#!/usr/bin/env python3
"""Streaming-like chunk latency/RTF benchmark for baseline and v0p1_B."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.defense_stftmask import STFTMaskDefense


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run streaming chunk benchmark for baseline and v0p1_B.")
    p.add_argument("--dataset", type=str, default="vctk", choices=["vctk", "librispeech"])
    p.add_argument(
        "--session_pool_csv",
        type=str,
        default=str(REPO_ROOT / "data" / "splits" / "session_pool_v3_test.csv"),
    )
    p.add_argument("--input_wav", type=str, default="")
    p.add_argument("--target_sec", type=float, default=60.0)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--chunk_ms_list", type=str, default="20,40")
    p.add_argument(
        "--defense_checkpoint",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "defense" / "v0p1_B_best.pt"),
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_wav(path: Path) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    return x.astype(np.float32), int(sr)


def resolve_long_wav(args: argparse.Namespace) -> Path:
    if args.input_wav:
        return Path(args.input_wav)

    src_csv = Path(args.session_pool_csv)
    if not src_csv.exists():
        raise FileNotFoundError(f"session_pool_csv not found: {src_csv}")
    rows: List[Dict[str, str]] = []
    with open(src_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    uniq_paths: List[str] = []
    seen = set()
    for r in rows:
        p = str(r.get("path", "")).strip()
        if p and p not in seen:
            seen.add(p)
            uniq_paths.append(p)
    if not uniq_paths:
        raise RuntimeError("No audio paths found in session pool CSV.")

    target_samples = int(round(float(args.target_sec) * int(args.sample_rate)))
    chunks = []
    total = 0
    for rel in uniq_paths:
        p = REPO_ROOT / rel
        if not p.exists():
            continue
        x, sr = read_wav(p)
        if sr != args.sample_rate:
            idx = np.linspace(0, max(0, len(x) - 1), int(len(x) * args.sample_rate / max(1, sr)))
            x = x[np.clip(np.round(idx).astype(np.int64), 0, max(0, len(x) - 1))]
        chunks.append(x.astype(np.float32))
        total += len(x)
        if total >= target_samples:
            break
    if total == 0:
        raise RuntimeError("Failed to collect any waveform from session pool paths.")
    y = np.concatenate(chunks, axis=0)
    y = y[:target_samples] if len(y) > target_samples else np.pad(y, (0, target_samples - len(y)))

    out_dir = REPO_ROOT / "artifacts" / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / f"benchmark_input_{args.dataset}_{int(args.target_sec)}s.wav"
    sf.write(out_wav, y, args.sample_rate)
    return out_wav


def load_defense_model(ckpt_path: Path, device: torch.device) -> STFTMaskDefense:
    payload = torch.load(ckpt_path, map_location=device)
    cfg = payload.get("model_config", {}) if isinstance(payload, dict) else {}
    model = STFTMaskDefense(cfg).to(device).eval()
    if isinstance(payload, dict) and "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def chunk_indices(n: int, chunk: int) -> List[Tuple[int, int]]:
    out = []
    s = 0
    while s < n:
        e = min(n, s + chunk)
        out.append((s, e))
        s = e
    return out


def bench_one(
    wav: np.ndarray,
    sr: int,
    chunk_ms: int,
    mode: str,
    device: torch.device,
    model: STFTMaskDefense,
) -> Dict[str, float]:
    chunk_n = max(1, int(round(sr * chunk_ms / 1000.0)))
    segs = chunk_indices(len(wav), chunk_n)
    times_ms: List[float] = []
    warmup = 50

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for i, (s, e) in enumerate(segs):
        x = wav[s:e]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        if mode == "baseline":
            _ = x
        else:
            xt = torch.from_numpy(x).to(device)
            with torch.inference_mode():
                _ = model(xt).detach()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        if i >= warmup:
            times_ms.append((t1 - t0) * 1000.0)
    total_proc = float(np.sum(np.asarray(times_ms, dtype=np.float64)) / 1000.0)
    n_eval = max(1, len(segs) - warmup)
    audio_dur = (n_eval * chunk_n) / float(sr)
    avg_ms = float(np.mean(times_ms)) if times_ms else float("nan")
    p95_ms = float(np.percentile(np.asarray(times_ms), 95)) if times_ms else float("nan")
    rtf = float(total_proc / max(1e-6, audio_dur))

    if device.type == "cuda":
        peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    else:
        peak_mem_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)

    return {
        "chunk_ms": float(chunk_ms),
        "avg_ms": avg_ms,
        "p95_ms": p95_ms,
        "rtf": rtf,
        "peak_mem_mb": peak_mem_mb,
        "n_chunks_total": float(len(segs)),
        "n_chunks_eval": float(n_eval),
        "warmup_chunks": float(min(warmup, len(segs))),
    }


def main() -> int:
    args = parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    input_wav = resolve_long_wav(args)
    wav, sr = read_wav(input_wav)
    if sr != int(args.sample_rate):
        raise ValueError(f"Expected sample_rate={args.sample_rate}, but got sr={sr} from {input_wav}")

    chunk_ms_list = [int(x.strip()) for x in str(args.chunk_ms_list).split(",") if x.strip()]
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    rows_out: List[Dict[str, str]] = []
    logs: List[Dict[str, object]] = []

    for dev_name in devices:
        dev = torch.device(dev_name)
        model = load_defense_model(Path(args.defense_checkpoint), dev)
        for chunk_ms in chunk_ms_list:
            for mode in ["baseline", "v0p1_B"]:
                res = bench_one(wav=wav, sr=sr, chunk_ms=chunk_ms, mode=mode, device=dev, model=model)
                rows_out.append(
                    {
                        "method": mode,
                        "device": dev_name,
                        "chunk_ms": str(int(res["chunk_ms"])),
                        "avg_ms": f"{res['avg_ms']:.6f}",
                        "p95_ms": f"{res['p95_ms']:.6f}",
                        "rtf": f"{res['rtf']:.6f}",
                        "peak_mem_mb": f"{res['peak_mem_mb']:.3f}",
                    }
                )
                logs.append(
                    {
                        "method": mode,
                        "device": dev_name,
                        "chunk_ms": int(chunk_ms),
                        "metrics": res,
                    }
                )

    out_table = REPO_ROOT / "artifacts" / "tables" / "latency_v0p1_B_v2.csv"
    out_log = REPO_ROOT / "artifacts" / "logs" / "latency_v0p1_B_v2.log"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_log.parent.mkdir(parents=True, exist_ok=True)

    with open(out_table, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["method", "device", "chunk_ms", "avg_ms", "p95_ms", "rtf", "peak_mem_mb"],
        )
        w.writeheader()
        w.writerows(rows_out)

    with open(out_log, "w", encoding="utf-8") as f:
        payload = {
            "input_wav": str(input_wav),
            "sample_rate": sr,
            "duration_sec": len(wav) / float(sr),
            "chunk_ms_list": chunk_ms_list,
            "results": logs,
        }
        f.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    print(json.dumps({"latency_csv": str(out_table), "latency_log": str(out_log)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

