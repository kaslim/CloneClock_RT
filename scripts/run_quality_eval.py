#!/usr/bin/env python3
"""
P0.2 Quality evaluation: WER + STOI (optional PESQ/DNSMOS) under telephony/codec.
Uses VCTK test utterances with official transcript (or ASR(clean) as pseudo ref).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics.asr import ASRMetric
from src.models.defense_stftmask import STFTMaskDefense
from src.transforms.telephony import TelephonyConfig, TelephonyTransform

try:
    from jiwer import wer as jiwer_wer
except Exception:
    jiwer_wer = None

try:
    from pystoi import stoi as pystoi_stoi  # type: ignore
except Exception:
    pystoi_stoi = None

try:
    import pesq  # type: ignore
except Exception:
    pesq = None

# DNSMOS / other optional
DNSMOS_AVAILABLE = False


def load_wav(path: Path) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32), int(sr)


def align_len(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    if n <= 0:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)
    return a[:n].astype(np.float32), b[:n].astype(np.float32)


def compute_stoi(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    if pystoi_stoi is None:
        return float("nan")
    r, d = align_len(ref, deg)
    try:
        return float(pystoi_stoi(r, d, sr, extended=False))
    except Exception:
        return float("nan")


def load_vctk_transcript(utter_id: str, speaker_id: str, dataset: str) -> Optional[str]:
    """Load VCTK official transcript from raw corpus txt folder."""
    raw_base = REPO_ROOT / "data" / "raw"
    for name in ("VCTK-Corpus", "VCTK-Corpus-0.92", "vctk"):
        txt_path = raw_base / name / "txt" / speaker_id / f"{utter_id}.txt"
        if txt_path.exists():
            try:
                return txt_path.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                pass
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quality eval: WER + STOI under telephony (utter-level, VCTK test).")
    p.add_argument("--dataset", type=str, default="vctk", choices=["vctk", "librispeech"])
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--telephony_codec", type=str, default="none", choices=["none", "opus", "g711"])
    p.add_argument("--telephony_codec_bitrate", type=str, default="16k")
    p.add_argument("--max_utts", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--telephony_config", type=str, default=str(REPO_ROOT / "configs" / "telephony.yaml"))
    p.add_argument("--eval_config", type=str, default=str(REPO_ROOT / "configs" / "eval.yaml"))
    p.add_argument("--defense_checkpoint", type=str, default="", help="v0p1_B checkpoint; empty = baseline only in one run")
    return p.parse_args()


def run_one_condition(
    rows: List[Dict],
    defense_model: Optional[torch.nn.Module],
    defense_name: str,
    telephony: TelephonyTransform,
    asr_metric: ASRMetric,
    device: torch.device,
    sr: int,
    out_rows: List[Dict],
    gt_source: str,
) -> None:
    for r in rows:
        utt_id = r.get("utter_id", "")
        spk_id = r.get("speaker_id", "")
        rel_path = r.get("path", "")
        wav_path = REPO_ROOT / "data" / "processed" / "vctk" / "wav16k" / rel_path
        if not wav_path.exists():
            wav_path = REPO_ROOT / rel_path
        if not wav_path.exists():
            continue
        clean, file_sr = load_wav(wav_path)
        if file_sr != sr:
            import librosa
            clean = librosa.resample(clean.astype(np.float64), orig_sr=file_sr, target_sr=sr).astype(np.float32)

        # Reference text: official transcript or ASR(clean) pseudo
        gt_text = load_vctk_transcript(utt_id, spk_id, "vctk") if utt_id and spk_id else None
        if not gt_text or not gt_text.strip():
            gt_tensor = torch.from_numpy(clean).unsqueeze(0)
            gt_text = asr_metric.transcribe(gt_tensor, sr)
            ref_source = "pseudo_asr_clean"
        else:
            ref_source = gt_source

        # Shared telephony params for this utterance
        params = telephony.sample_params()
        tele_clean = telephony.apply_with_params(clean, sample_rate=sr, params=params)

        if defense_model is not None:
            with torch.inference_mode():
                def_audio = defense_model(torch.from_numpy(clean).to(device)).detach().cpu().numpy().astype(np.float32)
        else:
            def_audio = clean.copy()
        tele_def = telephony.apply_with_params(def_audio, sample_rate=sr, params=params)

        # WER: ASR(tele_clean) vs gt, ASR(tele_def) vs gt
        hyp_clean = asr_metric.transcribe(torch.from_numpy(tele_clean).unsqueeze(0), sr)
        hyp_def = asr_metric.transcribe(torch.from_numpy(tele_def).unsqueeze(0), sr)
        wer_clean = float(jiwer_wer([gt_text], [hyp_clean])) if jiwer_wer and gt_text else float("nan")
        wer_def = float(jiwer_wer([gt_text], [hyp_def])) if jiwer_wer and gt_text else float("nan")

        # STOI: tele_clean vs tele_def (more relevant to telephony)
        stoi_val = compute_stoi(tele_clean, tele_def, sr)

        out_rows.append({
            "utt_id": utt_id,
            "spk_id": spk_id,
            "telephony": args_holder.telephony_tag,
            "defense": defense_name,
            "wer": wer_def,
            "wer_clean": wer_clean,
            "stoi": stoi_val,
            "gt_source": ref_source,
        })


def main() -> int:
    global args_holder
    args = parse_args()
    args_holder = args
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Telephony tag for output
    if args.telephony_codec == "none":
        args_holder.telephony_tag = "none"
    elif args.telephony_codec == "opus":
        args_holder.telephony_tag = "opus16k"
    else:
        args_holder.telephony_tag = "g711"

    with open(args.telephony_config, "r", encoding="utf-8") as f:
        import yaml
        full_cfg = yaml.safe_load(f) or {}
    tp_cfg = full_cfg.get("telephony", full_cfg)
    if args.telephony_codec != "none":
        tp_cfg.setdefault("codec", {})
        tp_cfg["codec"]["enabled"] = True
        tp_cfg["codec"]["name"] = str(args.telephony_codec)
        if str(args.telephony_codec).lower() == "opus":
            tp_cfg["codec"]["bitrate"] = str(args.telephony_codec_bitrate)
    else:
        tp_cfg.setdefault("codec", {})
        tp_cfg["codec"]["enabled"] = False

    telephony = TelephonyTransform(tp_cfg, seed=args.seed)

    with open(args.eval_config, "r", encoding="utf-8") as f:
        import yaml
        eval_cfg = yaml.safe_load(f) or {}
    device_name = eval_cfg.get("eval", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    asr_cfg = eval_cfg.get("asr_metric", eval_cfg.get("asr", {})) or {}
    asr_cfg.setdefault("device", str(device))
    asr_metric = ASRMetric(asr_cfg)
    sr = 16000

    split_csv = REPO_ROOT / "data" / "splits" / args.dataset / f"{args.split}.csv"
    if not split_csv.exists():
        print(f"Split not found: {split_csv}", file=sys.stderr)
        return 1
    with open(split_csv, "r", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    if args.max_utts > 0:
        rng = random.Random(args.seed)
        if len(all_rows) > args.max_utts:
            all_rows = rng.sample(all_rows, args.max_utts)
        else:
            all_rows = rng.sample(all_rows, len(all_rows))

    defense_model = None
    defense_ckpt = getattr(args, "defense_checkpoint", "") or ""
    if defense_ckpt and Path(defense_ckpt).exists():
        payload = torch.load(defense_ckpt, map_location=device)
        cfg = payload.get("model_config", {}) if isinstance(payload, dict) else {}
        defense_model = STFTMaskDefense(cfg).to(device).eval()
        if isinstance(payload, dict) and "model_state_dict" in payload:
            defense_model.load_state_dict(payload["model_state_dict"], strict=True)

    out_dir = REPO_ROOT / "artifacts" / "tables" / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rows: List[Dict] = []

    # Baseline: no defense
    run_one_condition(
        all_rows,
        None,
        "baseline",
        telephony,
        asr_metric,
        device,
        sr,
        out_rows,
        "vctk_txt",
    )
    # v0p1_B if checkpoint provided
    if defense_model is not None:
        run_one_condition(
            all_rows,
            defense_model,
            "v0p1_B",
            telephony,
            asr_metric,
            device,
            sr,
            out_rows,
            "vctk_txt",
        )

    # Per-telephony CSV
    out_csv = out_dir / f"quality_eval_{args_holder.telephony_tag}.csv"
    fields = ["utt_id", "spk_id", "telephony", "defense", "wer", "wer_clean", "stoi", "gt_source"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {out_csv}")

    # JSON mirror (optional)
    out_json = out_dir / f"quality_eval_{args_holder.telephony_tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_json}")

    # Summary: aggregate by telephony, defense (this run only)
    from collections import defaultdict
    agg: Dict[Tuple[str, str], List[Tuple[str, float]]] = defaultdict(list)
    for row in out_rows:
        key = (row["telephony"], row["defense"])
        w = row.get("wer")
        s = row.get("stoi")
        if isinstance(w, (int, float)) and (w == w):
            agg[key].append(("wer", float(w)))
        if isinstance(s, (int, float)) and (s == s):
            agg[key].append(("stoi", float(s)))
    summary_rows: List[Dict] = []
    for (tel, defense), pairs in sorted(agg.items()):
        wers = [v for k, v in pairs if k == "wer"]
        stois = [v for k, v in pairs if k == "stoi"]
        n_utts = len(wers) or len(stois) or 0
        summary_rows.append({
            "telephony": tel,
            "defense": defense,
            "wer_mean": f"{np.mean(wers):.4f}" if wers else "nan",
            "wer_std": f"{np.std(wers):.4f}" if len(wers) > 1 else "0",
            "stoi_mean": f"{np.mean(stois):.4f}" if stois else "nan",
            "stoi_std": f"{np.std(stois):.4f}" if len(stois) > 1 else "0",
            "n_utts": str(n_utts),
        })
    paper_dir = REPO_ROOT / "artifacts" / "tables" / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = paper_dir / "quality_summary.csv"
    # Merge with existing summary (other telephony conditions): read existing, merge by (telephony, defense), write
    summary_fields = ["telephony", "defense", "wer_mean", "wer_std", "stoi_mean", "stoi_std", "n_utts"]
    existing: List[Dict] = []
    if summary_csv.exists():
        with open(summary_csv, "r", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    key_to_row: Dict[Tuple[str, str], Dict] = {}
    for row in existing:
        key_to_row[(row["telephony"], row["defense"])] = row
    for row in summary_rows:
        key_to_row[(row["telephony"], row["defense"])] = row
    merged = sorted(key_to_row.values(), key=lambda x: (x["telephony"], x["defense"]))
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(merged)
    print(f"Wrote {summary_csv}")

    return 0


# Hold args for run_one_condition (telephony_tag)
args_holder: Optional[argparse.Namespace] = None

if __name__ == "__main__":
    sys.exit(main())
