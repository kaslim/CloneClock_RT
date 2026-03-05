#!/usr/bin/env python3
"""run_eval.py — clean/telephony/defended 评测."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import yaml
from jiwer import wer as jiwer_wer
try:
    from pystoi import stoi as pystoi_stoi  # type: ignore
except Exception:  # pragma: no cover
    pystoi_stoi = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics.asr import ASRMetric
from src.metrics.speaker import SpeakerMetric
from src.models.defense_stftmask import STFTMaskDefense
from src.transforms.telephony import TelephonyTransform


def get_data_paths(repo_root: Path, dataset: str) -> Dict[str, Path]:
    return {
        "processed_dir": repo_root / "data" / "processed" / dataset,
        "splits_dir": repo_root / "data" / "splits" / dataset,
    }


def simple_defense(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """A lightweight defense baseline: soft clipping + tiny moving average."""
    x = np.tanh(audio * 1.2).astype(np.float32)
    k = max(3, int(0.0015 * sample_rate) | 1)
    kernel = np.ones(k, dtype=np.float32) / k
    y = np.convolve(x, kernel, mode="same")
    return y.astype(np.float32)


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


def align_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    if n <= 0:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)
    return a[:n].astype(np.float32), b[:n].astype(np.float32)


def safe_stoi(ref: np.ndarray, hyp: np.ndarray, sample_rate: int) -> float:
    if pystoi_stoi is None:
        return float("nan")
    ref_a, hyp_a = align_pair(ref, hyp)
    try:
        return float(pystoi_stoi(ref_a, hyp_a, sample_rate, extended=False))
    except Exception:
        return float("nan")


def _window_stats(seg: np.ndarray, energy_thr: float, voiced_amp_thr: float) -> Tuple[float, float]:
    rms = float(np.sqrt(np.mean(np.square(seg)) + 1e-10))
    voiced_ratio = float(np.mean(np.abs(seg) >= voiced_amp_thr))
    return rms, voiced_ratio


def temporal_embedding_metrics(
    speaker_metric: SpeakerMetric,
    wav: np.ndarray,
    sample_rate: int,
    win_sec: float = 1.0,
    hop_sec: float = 0.5,
    energy_thr: float = 0.012,
    voiced_ratio_thr: float = 0.35,
    voiced_amp_thr: float = 0.015,
) -> Tuple[float, float, float, int]:
    win = max(1, int(win_sec * sample_rate))
    hop = max(1, int(hop_sec * sample_rate))
    if len(wav) < win:
        return float("nan"), float("nan"), float("nan"), 0
    embs: List[torch.Tensor] = []
    for s in range(0, len(wav) - win + 1, hop):
        seg_np = wav[s : s + win].astype(np.float32)
        rms, voiced_ratio = _window_stats(seg_np, energy_thr=energy_thr, voiced_amp_thr=voiced_amp_thr)
        if rms < energy_thr or voiced_ratio < voiced_ratio_thr:
            continue
        seg = torch.from_numpy(seg_np)
        emb = speaker_metric.embed(seg, sample_rate)
        emb = torch.nn.functional.normalize(emb, dim=0)
        embs.append(emb)
    if len(embs) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    E = torch.stack(embs, dim=0)  # [T, D]
    T = int(E.size(0))

    if T >= 2:
        adj_cos = torch.nn.functional.cosine_similarity(E[:-1], E[1:], dim=-1).mean().item()
    else:
        adj_cos = float("nan")

    mean_vec = E.mean(dim=0)
    mean_vec = torch.nn.functional.normalize(mean_vec, dim=0)
    drift_to_mean = torch.nn.functional.cosine_similarity(E, mean_vec.unsqueeze(0), dim=-1).mean().item()

    if T >= 2:
        cos_mat = E @ E.t()
        idx = torch.triu_indices(T, T, offset=1)
        pair_vals = cos_mat[idx[0], idx[1]]
        pairwise_cos_std = float(pair_vals.std(unbiased=False).item())
    else:
        pairwise_cos_std = float("nan")
    return float(adj_cos), float(drift_to_mean), pairwise_cos_std, T


def is_full_utterance_with_gt(row: Dict[str, str]) -> bool:
    text = str(row.get("text", "")).strip()
    if not text:
        return False
    crop_flags = ["is_cropped", "cropped", "chunked", "is_segment"]
    for k in crop_flags:
        if k in row and str(row.get(k, "")).strip().lower() in {"1", "true", "yes", "y"}:
            return False
    # If explicit segment boundaries exist, treat as cropped.
    segment_keys = ["start_sec", "end_sec", "segment_start", "segment_end"]
    if any(str(row.get(k, "")).strip() != "" for k in segment_keys):
        return False
    return True


def load_clean_transcript_cache(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    cache: Dict[Tuple[str, str], Dict[str, str]] = {}
    if not path.exists():
        return cache
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = str(row.get("dataset", "")).strip()
            rel_path = str(row.get("path", "")).strip()
            if dataset and rel_path:
                cache[(dataset, rel_path)] = {
                    "clean_text": str(row.get("clean_text", "")).strip(),
                    "telephony_clean_text": str(row.get("telephony_clean_text", "")).strip(),
                }
    return cache


def save_clean_transcript_cache(path: Path, cache: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["dataset", "path", "clean_text", "telephony_clean_text"],
        )
        w.writeheader()
        for (dataset, rel_path), payload in sorted(cache.items()):
            w.writerow(
                {
                    "dataset": dataset,
                    "path": rel_path,
                    "clean_text": payload.get("clean_text", ""),
                    "telephony_clean_text": payload.get("telephony_clean_text", ""),
                }
            )


def finite_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def finite_std(values: List[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.std())


def extract_window_embeddings(
    speaker_metric: SpeakerMetric,
    wav: np.ndarray,
    sample_rate: int,
    win_sec: float = 1.0,
    hop_sec: float = 0.5,
    energy_thr: float = 0.012,
    voiced_ratio_thr: float = 0.35,
    voiced_amp_thr: float = 0.015,
) -> List[torch.Tensor]:
    win = max(1, int(win_sec * sample_rate))
    hop = max(1, int(hop_sec * sample_rate))
    if len(wav) < win:
        return []
    embs: List[torch.Tensor] = []
    for s in range(0, len(wav) - win + 1, hop):
        seg = wav[s : s + win].astype(np.float32)
        rms, voiced_ratio = _window_stats(seg, energy_thr=energy_thr, voiced_amp_thr=voiced_amp_thr)
        if rms < energy_thr or voiced_ratio < voiced_ratio_thr:
            continue
        emb = speaker_metric.embed(torch.from_numpy(seg), sample_rate)
        emb = torch.nn.functional.normalize(emb, dim=0)
        embs.append(emb)
    return embs


def temporal_metrics_from_embeddings(embs: List[torch.Tensor]) -> Tuple[float, float, float, int]:
    T = len(embs)
    if T == 0:
        return float("nan"), float("nan"), float("nan"), 0
    E = torch.stack(embs, dim=0)
    if T >= 2:
        adj_cos = float(torch.nn.functional.cosine_similarity(E[:-1], E[1:], dim=-1).mean().item())
    else:
        adj_cos = float("nan")
    m = torch.nn.functional.normalize(E.mean(dim=0), dim=0)
    drift = float(torch.nn.functional.cosine_similarity(E, m.unsqueeze(0), dim=-1).mean().item())
    if T >= 2:
        cmat = E @ E.t()
        idx = torch.triu_indices(T, T, offset=1)
        pair_std = float(cmat[idx[0], idx[1]].std(unbiased=False).item())
    else:
        pair_std = float("nan")
    return adj_cos, drift, pair_std, T


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="评估 clean / telephony / defended")
    p.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "eval.yaml"))
    p.add_argument("--telephony_config", type=str, default=str(REPO_ROOT / "configs" / "telephony.yaml"))
    p.add_argument("--dataset", type=str, choices=["librispeech", "vctk"], default=None)
    p.add_argument(
        "--mode",
        type=str,
        choices=["eval_clean", "eval_telephony", "eval_defended", "eval_session"],
        default="eval_clean",
    )
    p.add_argument("--max_samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--defense_with_telephony", action="store_true", help="eval_defended 模式下追加 telephony")
    p.add_argument("--defense_checkpoint", type=str, default="", help="训练得到的防御模型 checkpoint（eval_defended）")
    p.add_argument("--device", type=str, default=None, help="cpu/cuda")
    p.add_argument(
        "--pseudo_ref_source",
        type=str,
        choices=["clean", "telephony_clean"],
        default="telephony_clean",
        help="wer_pseudo 参考文本来源：ASR(clean) 或 ASR(telephony(clean))",
    )
    p.add_argument("--sanity_samples", type=int, default=10, help="打印 sanity 样本数量")
    p.add_argument("--with_codec", action="store_true", help="评测时强制打开 telephony codec")
    p.add_argument("--codec_name", type=str, default="opus", choices=["opus", "g711", "pcm_mulaw", "mulaw"])
    p.add_argument(
        "--session_csv",
        type=str,
        default=str(REPO_ROOT / "data" / "processed" / "sessions" / "sessions.csv"),
        help="eval_session 模式的 session 清单",
    )
    p.add_argument("--max_sessions", type=int, default=0, help="eval_session 最多评测 session 数，0 为全部")
    p.add_argument("--session_min_windows", type=int, default=10, help="有效 session 的最小窗口数")
    p.add_argument("--session_k_list", type=str, default="1,2,4,8,16", help="聚合攻击 K 列表")
    p.add_argument("--session_with_defense", action="store_true", help="eval_session 模式启用 defense")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = {}
    if Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    dataset = args.dataset or (cfg.get("data") or {}).get("dataset", "librispeech")
    paths = get_data_paths(REPO_ROOT, dataset)
    split_csv = paths["splits_dir"] / "test.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing split file: {split_csv}")

    with open(args.telephony_config, "r", encoding="utf-8") as f:
        tcfg = yaml.safe_load(f) or {}
    telephony_cfg = tcfg.get("telephony", tcfg)
    if args.with_codec:
        telephony_cfg = dict(telephony_cfg)
        codec_cfg = dict(telephony_cfg.get("codec", {}))
        codec_cfg["enabled"] = True
        codec_cfg["name"] = args.codec_name
        telephony_cfg["codec"] = codec_cfg

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    telephony = TelephonyTransform(telephony_cfg, seed=args.seed)
    sp_cfg = cfg.get("speaker_metric", {}) if isinstance(cfg, dict) else {}
    asr_cfg = cfg.get("asr_metric", {}) if isinstance(cfg, dict) else {}
    speaker_metric = SpeakerMetric(
        {
            "encoder_name": sp_cfg.get("encoder_name", "WAV2VEC2_BASE"),
            "device": device,
            "checkpoint_dir": sp_cfg.get(
                "checkpoint_dir",
                str(REPO_ROOT / "checkpoints" / "speaker_encoders" / "speechbrain_ecapa"),
            ),
        }
    )
    asr_metric = ASRMetric(
        {
            "bundle_name": asr_cfg.get("bundle_name", "WAV2VEC2_ASR_BASE_960H"),
            "device": device,
        }
    )
    defense_model = None
    if args.mode == "eval_defended" and args.defense_checkpoint:
        ckpt_path = Path(args.defense_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"defense checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device)
        model_cfg = payload.get("model_config", {}) if isinstance(payload, dict) else {}
        defense_model = STFTMaskDefense(model_cfg).to(device).eval()
        if isinstance(payload, dict) and "model_state_dict" in payload:
            defense_model.load_state_dict(payload["model_state_dict"], strict=True)

    if args.mode == "eval_session":
        session_csv = Path(args.session_csv)
        if not session_csv.exists():
            raise FileNotFoundError(f"Missing session csv: {session_csv}")
        with open(session_csv, "r", encoding="utf-8") as f:
            srows = list(csv.DictReader(f))
        if args.max_sessions > 0:
            srows = srows[: args.max_sessions]

        out_dir = REPO_ROOT / "artifacts" / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"results_{args.mode}_{dataset}.csv"
        out_json = out_dir / f"results_{args.mode}_{dataset}.json"
        out_curve = out_dir / "session_attack_curve.csv"
        out_curve_abs = out_dir / "session_attack_curve_abs.csv"
        out_summary = out_dir / "session_eval_summary.csv"
        out_summary_abs = out_dir / "session_eval_summary_abs.csv"
        out_sanity = out_dir / "time_metrics_sanity.txt"

        ks = [int(x) for x in args.session_k_list.split(",") if x.strip()]
        agg_by_k: Dict[int, List[float]] = {k: [] for k in ks}
        session_results: List[Dict[str, object]] = []

        t0 = time.time()
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        for r in srows:
            wav_path = Path(r["path"])
            clean_audio, sr = load_wav(wav_path)

            if args.session_with_defense:
                if args.defense_checkpoint and defense_model is None:
                    ckpt_path = Path(args.defense_checkpoint)
                    payload = torch.load(ckpt_path, map_location=device)
                    model_cfg = payload.get("model_config", {}) if isinstance(payload, dict) else {}
                    defense_model = STFTMaskDefense(model_cfg).to(device).eval()
                    if isinstance(payload, dict) and "model_state_dict" in payload:
                        defense_model.load_state_dict(payload["model_state_dict"], strict=True)
                if defense_model is not None:
                    with torch.inference_mode():
                        proc_audio = defense_model(torch.from_numpy(clean_audio).to(device)).detach().cpu().numpy().astype(np.float32)
                else:
                    proc_audio = simple_defense(clean_audio, sr)
            else:
                proc_audio = clean_audio

            # Shared telephony view for speaker/session aggregation metrics.
            tp = telephony.sample_params()
            tel_clean = telephony.apply_with_params(clean_audio, sample_rate=sr, params=tp)
            tel_proc = telephony.apply_with_params(proc_audio, sample_rate=sr, params=tp)

            # Independent telephony view for absolute quality metrics (aligned with eval_telephony).
            tel_clean_abs = telephony(clean_audio, sample_rate=sr)
            tel_proc_abs = telephony(proc_audio, sample_rate=sr)

            # Session-level speaker metrics.
            clean_emb = speaker_metric.embed(torch.from_numpy(tel_clean), sr)
            proc_emb = speaker_metric.embed(torch.from_numpy(tel_proc), sr)
            speaker_sim_tel = speaker_metric.cosine(clean_emb, proc_emb)

            # Relative quality metrics under shared channel (optional diagnostics).
            clean_text_rel = asr_metric.transcribe(torch.from_numpy(tel_clean), sr)
            proc_text_rel = asr_metric.transcribe(torch.from_numpy(tel_proc), sr)
            wer_pseudo_rel = float(jiwer_wer([clean_text_rel], [proc_text_rel]))
            stoi_rel = safe_stoi(tel_clean, tel_proc, sr)

            # Absolute quality metrics (primary reporting, aligned to eval_telephony).
            clean_text_abs = asr_metric.transcribe(torch.from_numpy(clean_audio), sr)
            tel_clean_text_abs = asr_metric.transcribe(torch.from_numpy(tel_clean_abs), sr)
            proc_text_abs = asr_metric.transcribe(torch.from_numpy(tel_proc_abs), sr)
            pseudo_ref_abs = clean_text_abs if args.pseudo_ref_source == "clean" else tel_clean_text_abs
            wer_pseudo_abs = float(jiwer_wer([pseudo_ref_abs], [proc_text_abs]))
            stoi_abs = safe_stoi(tel_clean_abs, tel_proc_abs, sr)

            clean_ws = extract_window_embeddings(speaker_metric, tel_clean, sr)
            proc_ws = extract_window_embeddings(speaker_metric, tel_proc, sr)
            adj_cos, drift, pair_std, n_used = temporal_metrics_from_embeddings(proc_ws)
            valid = int(n_used >= args.session_min_windows)

            if valid and len(clean_ws) > 0:
                ref_clean = torch.nn.functional.normalize(torch.stack(clean_ws, dim=0).mean(dim=0), dim=0)
                for k in ks:
                    if len(proc_ws) >= k:
                        e_k = torch.nn.functional.normalize(torch.stack(proc_ws[:k], dim=0).mean(dim=0), dim=0)
                        agg = float(torch.dot(e_k, ref_clean).item())
                        agg_by_k[k].append(agg)

            session_results.append(
                {
                    "session_id": r["session_id"],
                    "speaker_id": r["speaker_id"],
                    "path": r["path"],
                    "n_segments": int(r.get("n_segments", "0") or 0),
                    "total_duration": float(r.get("total_duration", "0") or 0.0),
                    "speaker_sim_tel": speaker_sim_tel,
                    "wer_pseudo": wer_pseudo_abs,
                    "stoi": stoi_abs,
                    "wer_pseudo_abs": wer_pseudo_abs,
                    "stoi_abs": stoi_abs,
                    "wer_pseudo_rel": wer_pseudo_rel,
                    "stoi_rel": stoi_rel,
                    "adj_cos_mean": adj_cos,
                    "drift_to_mean": drift,
                    "pairwise_cos_std": pair_std,
                    "n_windows_used": n_used,
                    "valid_session": valid,
                }
            )

        elapsed = time.time() - t0
        peak_mem_mb = 0.0
        if torch.cuda.is_available() and device.startswith("cuda"):
            peak_mem_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))

        # Save curve table
        curve_fields = ["K", "agg_cos_to_clean_mean", "agg_cos_to_clean_std", "count"]
        curve_rows = []
        for k in ks:
            vals = agg_by_k[k]
            curve_rows.append(
                {
                    "K": k,
                    "agg_cos_to_clean_mean": finite_mean(vals),
                    "agg_cos_to_clean_std": finite_std(vals),
                    "count": len(vals),
                }
            )
        for curve_path in [out_curve, out_curve_abs]:
            with open(curve_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=curve_fields)
                w.writeheader()
                w.writerows(curve_rows)

        # Save per-session results
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "session_id",
                    "speaker_id",
                    "path",
                    "n_segments",
                    "total_duration",
                    "speaker_sim_tel",
                    "wer_pseudo",
                    "stoi",
                    "wer_pseudo_abs",
                    "stoi_abs",
                    "wer_pseudo_rel",
                    "stoi_rel",
                    "adj_cos_mean",
                    "drift_to_mean",
                    "pairwise_cos_std",
                    "n_windows_used",
                    "valid_session",
                ],
            )
            w.writeheader()
            w.writerows(session_results)

        valid_rows = [x for x in session_results if int(x["valid_session"]) == 1]
        summary = {
            "mode": args.mode,
            "dataset": dataset,
            "sessions_total": len(session_results),
            "sessions_valid": len(valid_rows),
            "session_min_windows": args.session_min_windows,
            "speaker_sim_tel_mean": finite_mean([float(x["speaker_sim_tel"]) for x in valid_rows]),
            "wer_pseudo_mean": finite_mean([float(x["wer_pseudo"]) for x in valid_rows]),
            "stoi_mean": finite_mean([float(x["stoi"]) for x in valid_rows]),
            "wer_pseudo_abs_mean": finite_mean([float(x["wer_pseudo_abs"]) for x in valid_rows]),
            "stoi_abs_mean": finite_mean([float(x["stoi_abs"]) for x in valid_rows]),
            "wer_pseudo_rel_mean": finite_mean([float(x["wer_pseudo_rel"]) for x in valid_rows]),
            "stoi_rel_mean": finite_mean([float(x["stoi_rel"]) for x in valid_rows]),
            "adj_cos_mean": finite_mean([float(x["adj_cos_mean"]) for x in valid_rows]),
            "drift_to_mean": finite_mean([float(x["drift_to_mean"]) for x in valid_rows]),
            "pairwise_cos_std_mean": finite_mean([float(x["pairwise_cos_std"]) for x in valid_rows]),
            "n_windows_used_mean": finite_mean([float(x["n_windows_used"]) for x in valid_rows]),
            "agg_curve_csv": str(out_curve_abs),
            "elapsed_sec": float(elapsed),
            "peak_gpu_mem_mb": peak_mem_mb,
            "with_codec": bool(args.with_codec),
            "session_with_defense": bool(args.session_with_defense),
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "sessions": session_results}, f, ensure_ascii=False, indent=2)

        # one-row summary for quick comparison
        fields = [
            "mode",
            "dataset",
            "sessions_total",
            "sessions_valid",
            "speaker_sim_tel_mean",
            "wer_pseudo_mean",
            "stoi_mean",
            "wer_pseudo_abs_mean",
            "stoi_abs_mean",
            "wer_pseudo_rel_mean",
            "stoi_rel_mean",
            "adj_cos_mean",
            "drift_to_mean",
            "pairwise_cos_std_mean",
            "n_windows_used_mean",
            "session_min_windows",
            "with_codec",
            "session_with_defense",
            "agg_curve_csv",
            "elapsed_sec",
            "peak_gpu_mem_mb",
        ]
        for summary_path in [out_summary, out_summary_abs]:
            with open(summary_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerow({k: summary.get(k, "") for k in fields})

        # Sanity file: at least up to 10 session records
        sanity_rows = random.sample(valid_rows, k=min(10, len(valid_rows))) if valid_rows else []
        with open(out_sanity, "w", encoding="utf-8") as f:
            f.write(f"mode={args.mode}, dataset={dataset}, valid={len(valid_rows)}/{len(session_results)}\n")
            f.write("session_id,speaker_id,n_windows_used,adj_cos_mean,drift_to_mean,pairwise_cos_std\n")
            for x in sanity_rows:
                f.write(
                    f"{x['session_id']},{x['speaker_id']},{x['n_windows_used']},"
                    f"{x['adj_cos_mean']},{x['drift_to_mean']},{x['pairwise_cos_std']}\n"
                )

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    with open(split_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    random.shuffle(rows)
    rows = rows[: args.max_samples]

    out_dir = REPO_ROOT / "artifacts" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"results_{args.mode}_{dataset}.csv"
    out_json = out_dir / f"results_{args.mode}_{dataset}.json"
    out_trans = out_dir / f"transcripts_{args.mode}_{dataset}.csv"
    out_time_sanity = out_dir / "time_metrics_sanity.txt"
    clean_cache_path = out_dir / "transcripts_clean.csv"
    clean_cache = load_clean_transcript_cache(clean_cache_path)

    t0 = time.time()
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    sample_results: List[Dict[str, object]] = []
    clean_embs = []
    eval_embs = []
    pos_scores = []
    speaker_sim_clean_proc_scores = []
    speaker_sim_tel_scores = []
    wers = []
    wers_gt = []
    wers_pseudo = []
    stois = []
    adj_cos_means = []
    drift_to_means = []
    pairwise_cos_stds = []
    n_windows_list = []

    for row in rows:
        rel_path = row["path"]
        wav_path = paths["processed_dir"] / "wav16k" / rel_path
        clean_audio, sr = load_wav(wav_path)

        if args.mode == "eval_clean":
            proc_audio = clean_audio
            eval_audio = clean_audio
        elif args.mode == "eval_telephony":
            proc_audio = clean_audio
            eval_audio = telephony(clean_audio, sample_rate=sr)
        elif args.mode == "eval_defended":
            if defense_model is not None:
                with torch.inference_mode():
                    defended_t = defense_model(torch.from_numpy(clean_audio).to(device))
                proc_audio = defended_t.detach().cpu().numpy().astype(np.float32)
            else:
                proc_audio = simple_defense(clean_audio, sr)
            eval_audio = telephony(proc_audio, sample_rate=sr) if args.defense_with_telephony else proc_audio
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        tel_clean_audio = telephony(clean_audio, sample_rate=sr)
        tel_proc_audio = telephony(proc_audio, sample_rate=sr)

        clean_tensor = torch.from_numpy(clean_audio)
        proc_tensor = torch.from_numpy(proc_audio)
        eval_tensor = torch.from_numpy(eval_audio)
        tel_clean_tensor = torch.from_numpy(tel_clean_audio)
        tel_proc_tensor = torch.from_numpy(tel_proc_audio)

        clean_emb = speaker_metric.embed(clean_tensor, sr)
        proc_emb = speaker_metric.embed(proc_tensor, sr)
        eval_emb = speaker_metric.embed(eval_tensor, sr)
        tel_clean_emb = speaker_metric.embed(tel_clean_tensor, sr)
        tel_proc_emb = speaker_metric.embed(tel_proc_tensor, sr)
        clean_embs.append(clean_emb)
        eval_embs.append(eval_emb)
        cos = float(torch.nn.functional.cosine_similarity(clean_emb, eval_emb, dim=0).item())
        pos_scores.append(cos)
        speaker_sim_clean_proc = speaker_metric.cosine(clean_emb, proc_emb)
        speaker_sim_tel = speaker_metric.cosine(tel_clean_emb, tel_proc_emb)
        speaker_sim_clean_proc_scores.append(speaker_sim_clean_proc)
        speaker_sim_tel_scores.append(speaker_sim_tel)

        cache_key = (dataset, rel_path)
        cache_entry = clean_cache.get(cache_key, {})
        clean_text = cache_entry.get("clean_text", "")
        tel_clean_text = cache_entry.get("telephony_clean_text", "")
        if not clean_text:
            clean_text = asr_metric.transcribe(clean_tensor, sr)
        if not tel_clean_text:
            tel_clean_text = asr_metric.transcribe(torch.from_numpy(tel_clean_audio), sr)
        clean_cache[cache_key] = {
            "clean_text": clean_text,
            "telephony_clean_text": tel_clean_text,
        }

        eval_text = asr_metric.transcribe(eval_tensor, sr)
        tel_proc_text = asr_metric.transcribe(torch.from_numpy(tel_proc_audio), sr)

        gt_text = row.get("text", "").strip()
        if is_full_utterance_with_gt(row):
            wer_gt = float(jiwer_wer([gt_text], [eval_text]))
        else:
            wer_gt = float("nan")
        wers_gt.append(wer_gt)

        pseudo_ref = clean_text if args.pseudo_ref_source == "clean" else tel_clean_text
        wer_pseudo = float(jiwer_wer([pseudo_ref], [tel_proc_text]))
        wers_pseudo.append(wer_pseudo)
        wers.append(wer_pseudo)

        stoi = safe_stoi(tel_clean_audio, tel_proc_audio, sr)
        stois.append(stoi)
        adj_cos_mean, drift_to_mean, pairwise_cos_std, n_windows_used = temporal_embedding_metrics(
            speaker_metric=speaker_metric,
            wav=tel_proc_audio,
            sample_rate=sr,
            win_sec=1.0,
            hop_sec=0.5,
        )
        adj_cos_means.append(adj_cos_mean)
        drift_to_means.append(drift_to_mean)
        pairwise_cos_stds.append(pairwise_cos_std)
        n_windows_list.append(float(n_windows_used))

        sample_results.append(
            {
                "utter_id": row["utter_id"],
                "speaker_id": row["speaker_id"],
                "path": rel_path,
                "mode": args.mode,
                "cosine_sim": cos,
                "speaker_sim_clean_vs_proc": speaker_sim_clean_proc,
                "speaker_sim_tel_clean_vs_tel_proc": speaker_sim_tel,
                "reference_text": gt_text,
                "clean_text": clean_text,
                "telephony_clean_text": tel_clean_text,
                "eval_text": eval_text,
                "telephony_proc_text": tel_proc_text,
                "wer": wer_pseudo,
                "wer_gt": wer_gt,
                "wer_pseudo": wer_pseudo,
                "stoi": stoi,
                "adj_cos_mean": adj_cos_mean,
                "drift_to_mean": drift_to_mean,
                "pairwise_cos_std": pairwise_cos_std,
                "n_windows_used": n_windows_used,
            }
        )

    neg_scores = []
    if len(clean_embs) > 1:
        for i in range(len(clean_embs)):
            j = (i + 1) % len(clean_embs)
            neg = float(torch.nn.functional.cosine_similarity(clean_embs[i], eval_embs[j], dim=0).item())
            neg_scores.append(neg)
        eer = speaker_metric.batch_eer(pos_scores, neg_scores)
    else:
        eer = float("nan")

    elapsed = time.time() - t0
    peak_mem_mb = 0.0
    if torch.cuda.is_available() and device.startswith("cuda"):
        peak_mem_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))

    summary = {
        "mode": args.mode,
        "dataset": dataset,
        "samples": len(sample_results),
        "speaker_cosine_mean": float(np.mean(pos_scores)) if pos_scores else float("nan"),
        "speaker_cosine_std": float(np.std(pos_scores)) if pos_scores else float("nan"),
        "speaker_eer_optional": float(eer),
        "speaker_sim_clean_vs_proc_mean": finite_mean(speaker_sim_clean_proc_scores),
        "speaker_sim_clean_vs_proc_std": finite_std(speaker_sim_clean_proc_scores),
        "speaker_sim_tel_clean_vs_tel_proc_mean": finite_mean(speaker_sim_tel_scores),
        "speaker_sim_tel_clean_vs_tel_proc_std": finite_std(speaker_sim_tel_scores),
        "wer_mean": finite_mean(wers),
        "wer_std": finite_std(wers),
        "wer_gt_mean": finite_mean(wers_gt),
        "wer_gt_std": finite_std(wers_gt),
        "wer_pseudo_mean": finite_mean(wers_pseudo),
        "wer_pseudo_std": finite_std(wers_pseudo),
        "stoi_mean": finite_mean(stois),
        "stoi_std": finite_std(stois),
        "adj_cos_mean": finite_mean(adj_cos_means),
        "adj_cos_std": finite_std(adj_cos_means),
        "drift_to_mean": finite_mean(drift_to_means),
        "drift_to_mean_std": finite_std(drift_to_means),
        "pairwise_cos_std_mean": finite_mean(pairwise_cos_stds),
        "pairwise_cos_std_std": finite_std(pairwise_cos_stds),
        "n_windows_used_mean": finite_mean(n_windows_list),
        "pseudo_ref_source": args.pseudo_ref_source,
        "stoi_available": bool(pystoi_stoi is not None),
        "with_codec": bool(args.with_codec),
        "codec_name": args.codec_name if args.with_codec else "",
        "elapsed_sec": float(elapsed),
        "device": device,
        "peak_gpu_mem_mb": peak_mem_mb,
        "outputs": {
            "results_csv": str(out_csv),
            "results_json": str(out_json),
            "transcripts_csv": str(out_trans),
        },
    }

    fieldnames = [
        "utter_id",
        "speaker_id",
        "path",
        "mode",
        "cosine_sim",
        "speaker_sim_clean_vs_proc",
        "speaker_sim_tel_clean_vs_tel_proc",
        "reference_text",
        "clean_text",
        "telephony_clean_text",
        "eval_text",
        "telephony_proc_text",
        "wer",
        "wer_gt",
        "wer_pseudo",
        "stoi",
        "adj_cos_mean",
        "drift_to_mean",
        "pairwise_cos_std",
        "n_windows_used",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sample_results)
    with open(out_trans, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "utter_id",
                "reference_text",
                "clean_text",
                "telephony_clean_text",
                "eval_text",
                "telephony_proc_text",
            ],
        )
        w.writeheader()
        for r in sample_results:
            w.writerow(
                {
                    "utter_id": r["utter_id"],
                    "reference_text": r["reference_text"],
                    "clean_text": r["clean_text"],
                    "telephony_clean_text": r["telephony_clean_text"],
                    "eval_text": r["eval_text"],
                    "telephony_proc_text": r["telephony_proc_text"],
                }
            )
    save_clean_transcript_cache(clean_cache_path, clean_cache)

    sanity_pool = [r for r in sample_results if int(r.get("n_windows_used", 0)) > 0]
    with open(out_time_sanity, "w", encoding="utf-8") as f:
        f.write(f"mode={args.mode}, dataset={dataset}, with_codec={args.with_codec}, codec={args.codec_name}\n")
        f.write("utter_id,speaker_id,n_windows_used,adj_cos_mean,drift_to_mean,pairwise_cos_std\n")
        for r in random.sample(sanity_pool, k=min(5, len(sanity_pool))):
            f.write(
                f"{r['utter_id']},{r['speaker_id']},{r['n_windows_used']},"
                f"{r['adj_cos_mean']},{r['drift_to_mean']},{r['pairwise_cos_std']}\n"
            )

    sanity_n = max(0, min(args.sanity_samples, len(sample_results)))
    if sanity_n > 0:
        sanity_rows = random.sample(sample_results, k=sanity_n)
        print("\n[SanityCheck] utter_id | GT | ASR(clean) | ASR(proc_tel)")
        for r in sanity_rows:
            gt = str(r["reference_text"]).strip() if str(r["reference_text"]).strip() else "<NA>"
            print(
                f"- {r['utter_id']} | GT: {gt} | "
                f"ASR(clean): {r['clean_text']} | ASR(proc_tel): {r['telephony_proc_text']}"
            )
        for r in random.sample(sanity_pool, k=min(5, len(sanity_pool))):
            print(
                f"[TimeSanity] {r['utter_id']} windows={r['n_windows_used']} "
                f"adj={r['adj_cos_mean']:.4f} drift={r['drift_to_mean']:.4f}"
            )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": sample_results}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
