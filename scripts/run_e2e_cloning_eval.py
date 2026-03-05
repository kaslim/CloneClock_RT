#!/usr/bin/env python3
"""
P0.1 End-to-end zero-shot cloning evaluation: baseline_ref vs defended_ref.
Uses session pool v3, targeted bestK refs, XTTS v2 or YourTTS; outputs cos(synth, target_ref).
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics.speaker import SpeakerMetric
from src.metrics.session_attack import pick_indices
from src.models.defense_stftmask import STFTMaskDefense
from src.transforms.telephony import TelephonyTransform

# Fixed synthesis text (short, reproducible)
SYNTH_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
]

SILENCE_NSAMPLES = int(0.1 * 16000)  # 0.1 s at 16 kHz


def load_wav(path: Path) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32), int(sr)


def concat_segments(segments: List[np.ndarray], sr: int, silence_samples: int = SILENCE_NSAMPLES) -> np.ndarray:
    out: List[np.ndarray] = []
    sil = np.zeros(silence_samples, dtype=np.float32)
    for i, seg in enumerate(segments):
        if i > 0:
            out.append(sil)
        out.append(seg)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def get_cloning_model(model_name: str, device: str):
    """Load XTTS v2 or YourTTS; return (model, name) or (None, reason)."""
    if model_name.lower() == "xtts":
        try:
            from TTS.api import TTS
            m = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            return m, "xtts"
        except Exception as e:
            return None, f"xtts_fail:{e}"
    if model_name.lower() in ("yourtts", "your_tts"):
        try:
            from TTS.api import TTS
            m = TTS("tts_models/multilingual/multi-dataset/your_tts").to(device)
            return m, "yourtts"
        except Exception as e:
            return None, f"yourtts_fail:{e}"
    return None, "unknown_model"


def synthesize(cloning_model, ref_wav_path: Path, text: str, out_path: Path, language: str = "en", model_name: str = "xtts") -> bool:
    try:
        if model_name == "xtts":
            cloning_model.tts_to_file(
                text=text,
                file_path=str(out_path),
                speaker_wav=str(ref_wav_path),
                language=language,
            )
        else:
            cloning_model.tts_to_file(
                text=text,
                file_path=str(out_path),
                speaker_wav=str(ref_wav_path),
                language=language,
            )
        return out_path.exists()
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2E cloning eval: baseline_ref vs defended_ref, K=1,16,64.")
    p.add_argument("--dataset", type=str, default="vctk")
    p.add_argument("--session_pool_csv", type=str, default="")
    p.add_argument("--model", type=str, default="xtts", choices=["xtts", "yourtts"])
    p.add_argument("--max_sessions", type=int, default=20)
    p.add_argument("--Ks", type=str, default="1,16,64")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--defense_checkpoint", type=str, default="")
    p.add_argument("--telephony_config", type=str, default=str(REPO_ROOT / "configs" / "telephony.yaml"))
    p.add_argument("--eval_config", type=str, default=str(REPO_ROOT / "configs" / "eval.yaml"))
    p.add_argument("--speaker_encoder_name", type=str, default="speechbrain_ecapa")
    p.add_argument("--speaker_checkpoint_dir", type=str, default="")
    p.add_argument("--speaker_checkpoint_dir_xvector", type=str, default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pool_csv = args.session_pool_csv or str(REPO_ROOT / "data" / "splits" / "session_pool_v3_test.csv")
    if not Path(pool_csv).exists():
        print(f"Session pool not found: {pool_csv}", file=sys.stderr)
        return 1

    with open(args.telephony_config, "r", encoding="utf-8") as f:
        import yaml
        full_tp = yaml.safe_load(f) or {}
    tp_cfg = full_tp.get("telephony", full_tp)
    telephony = TelephonyTransform(tp_cfg, seed=args.seed)

    with open(args.eval_config, "r", encoding="utf-8") as f:
        import yaml
        eval_cfg = yaml.safe_load(f) or {}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spk_cfg = eval_cfg.get("speaker_metric", eval_cfg.get("speaker", {})) or {}
    spk_cfg.setdefault("device", str(device))
    spk_cfg["encoder_name"] = args.speaker_encoder_name
    if args.speaker_checkpoint_dir:
        spk_cfg["checkpoint_dir"] = args.speaker_checkpoint_dir
    if args.speaker_checkpoint_dir_xvector:
        spk_cfg["checkpoint_dir_xvector"] = args.speaker_checkpoint_dir_xvector
    speaker_metric = SpeakerMetric(spk_cfg)
    sr = 16000

    defense_model = None
    if args.defense_checkpoint and Path(args.defense_checkpoint).exists():
        payload = torch.load(args.defense_checkpoint, map_location=device)
        cfg = payload.get("model_config", {}) if isinstance(payload, dict) else {}
        defense_model = STFTMaskDefense(cfg).to(device).eval()
        if isinstance(payload, dict) and "model_state_dict" in payload:
            defense_model.load_state_dict(payload["model_state_dict"], strict=True)

    cloning_model, model_tag = get_cloning_model(args.model, str(device))
    if cloning_model is None:
        print(f"[run_e2e_cloning_eval] Cloning model unavailable: {model_tag}", file=sys.stderr)

    Ks = [int(x.strip()) for x in args.Ks.split(",") if x.strip()]
    if not Ks:
        Ks = [1, 16, 64]

    with open(pool_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_session: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_session[r["session_id"]].append(r)
    session_ids = sorted(by_session.keys())
    if args.max_sessions > 0:
        rng = random.Random(args.seed)
        session_ids = rng.sample(session_ids, min(args.max_sessions, len(session_ids)))

    out_rows: List[Dict] = []
    rng = random.Random(args.seed)

    for sid in session_ids:
        win_rows = sorted(by_session[sid], key=lambda x: int(x.get("chunk_id", "0").split("_")[-1]) if x.get("chunk_id") else 0)
        if not win_rows:
            continue
        spk_id = win_rows[0]["speaker_id"]
        rel_path = win_rows[0]["path"]
        wav_path = REPO_ROOT / rel_path
        if not wav_path.exists():
            wav_path = REPO_ROOT / "data" / "processed" / args.dataset / rel_path
        if not wav_path.exists():
            continue
        full_wav, file_sr = load_wav(wav_path)
        if file_sr != sr:
            import librosa
            full_wav = librosa.resample(full_wav.astype(np.float64), orig_sr=file_sr, target_sr=sr).astype(np.float32)

        clean_segs: List[np.ndarray] = []
        def_segs: List[np.ndarray] = []
        tele_clean_segs: List[np.ndarray] = []
        tele_def_segs: List[np.ndarray] = []
        params = telephony.sample_params()

        for wr in win_rows:
            st = int(round(float(wr["start_sec"]) * sr))
            du = int(round(float(wr["dur_sec"]) * sr))
            if st + du > len(full_wav):
                continue
            cseg = full_wav[st : st + du]
            if defense_model is not None:
                with torch.inference_mode():
                    dseg = defense_model(torch.from_numpy(cseg).to(device)).detach().cpu().numpy().astype(np.float32)
            else:
                dseg = cseg.copy()
            tclean = telephony.apply_with_params(cseg, sample_rate=sr, params=params)
            tdef = telephony.apply_with_params(dseg, sample_rate=sr, params=params)
            clean_segs.append(cseg)
            def_segs.append(dseg)
            tele_clean_segs.append(tclean)
            tele_def_segs.append(tdef)

        n_win = len(clean_segs)
        if n_win < 1:
            continue

        clean_embs = []
        tele_clean_embs = []
        tele_def_embs = []
        for i in range(n_win):
            ce = speaker_metric.embed(torch.from_numpy(clean_segs[i]), sr)
            tce = speaker_metric.embed(torch.from_numpy(tele_clean_segs[i]), sr)
            tde = speaker_metric.embed(torch.from_numpy(tele_def_segs[i]), sr)
            clean_embs.append(F.normalize(ce, dim=0))
            tele_clean_embs.append(F.normalize(tce, dim=0))
            tele_def_embs.append(F.normalize(tde, dim=0))
        clean_z = torch.stack(clean_embs, dim=0)
        tele_clean_z = torch.stack(tele_clean_embs, dim=0)
        tele_def_z = torch.stack(tele_def_embs, dim=0)
        e_ref = F.normalize(clean_z.mean(dim=0), dim=0)
        target_ref_wav = concat_segments(clean_segs, sr)

        synth_text = SYNTH_TEXTS[0]

        for K in Ks:
            if K > n_win:
                continue
            idx_baseline = pick_indices("bestK_by_ref_similarity", clean_z, K, rng, selection_z=tele_clean_z)
            idx_defended = pick_indices("bestK_by_ref_similarity", clean_z, K, rng, selection_z=tele_def_z)
            baseline_ref_wav = concat_segments([tele_clean_segs[i] for i in idx_baseline], sr)
            defended_ref_wav = concat_segments([tele_def_segs[i] for i in idx_defended], sr)

            cos_baseline_ecapa = float("nan")
            cos_defended_ecapa = float("nan")
            cos_baseline_xv = float("nan")
            cos_defended_xv = float("nan")
            wer_synth_baseline = ""
            wer_synth_defended = ""

            if cloning_model is not None and len(baseline_ref_wav) >= sr * 2 and len(defended_ref_wav) >= sr * 2:
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    ref_b = tmp / "ref_b.wav"
                    ref_d = tmp / "ref_d.wav"
                    out_b = tmp / "synth_b.wav"
                    out_d = tmp / "synth_d.wav"
                    sf.write(ref_b, baseline_ref_wav, sr)
                    sf.write(ref_d, defended_ref_wav, sr)
                    ok_b = synthesize(cloning_model, ref_b, synth_text, out_b, "en", model_tag)
                    ok_d = synthesize(cloning_model, ref_d, synth_text, out_d, "en", model_tag)
                    if ok_b and out_b.exists():
                        syn_b, _ = load_wav(out_b)
                        emb_syn_b = speaker_metric.embed(torch.from_numpy(syn_b), sr)
                        cos_baseline_ecapa = float(F.cosine_similarity(emb_syn_b.unsqueeze(0), e_ref.unsqueeze(0)).item())
                    if ok_d and out_d.exists():
                        syn_d, _ = load_wav(out_d)
                        emb_syn_d = speaker_metric.embed(torch.from_numpy(syn_d), sr)
                        cos_defended_ecapa = float(F.cosine_similarity(emb_syn_d.unsqueeze(0), e_ref.unsqueeze(0)).item())

            out_rows.append({
                "session_id": sid,
                "spk_id": spk_id,
                "K": str(K),
                "model": model_tag,
                "condition": "baseline_ref",
                "cos_ecapa": f"{cos_baseline_ecapa:.6f}" if cos_baseline_ecapa == cos_baseline_ecapa else "nan",
                "cos_xvector": "nan",
                "wer_synth": wer_synth_baseline,
            })
            out_rows.append({
                "session_id": sid,
                "spk_id": spk_id,
                "K": str(K),
                "model": model_tag,
                "condition": "defended_ref",
                "cos_ecapa": f"{cos_defended_ecapa:.6f}" if cos_defended_ecapa == cos_defended_ecapa else "nan",
                "cos_xvector": "nan",
                "wer_synth": wer_synth_defended,
            })

    paper_dir = REPO_ROOT / "artifacts" / "tables" / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    eval_csv = paper_dir / "e2e_cloning_eval.csv"
    fields = ["session_id", "spk_id", "K", "model", "condition", "cos_ecapa", "cos_xvector", "wer_synth"]
    with open(eval_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {eval_csv}")

    # Summary: model, K, metric, baseline_mean, defended_mean, delta
    def f(x):
        try:
            return float(x)
        except Exception:
            return float("nan")
    agg: Dict[Tuple[str, int, str], List[float]] = defaultdict(list)
    for r in out_rows:
        k = (r["model"], int(r["K"]), r["condition"])
        v = r.get("cos_ecapa")
        if v and str(v).lower() != "nan":
            agg[k].append(f(v))
    summary_rows = []
    for (model, K) in sorted(set((r["model"], int(r["K"])) for r in out_rows)):
        bl_vals = agg.get((model, K, "baseline_ref"), [])
        def_vals = agg.get((model, K, "defended_ref"), [])
        bl_mean = np.nanmean(bl_vals) if bl_vals else float("nan")
        def_mean = np.nanmean(def_vals) if def_vals else float("nan")
        delta = (def_mean - bl_mean) if (bl_mean == bl_mean and def_mean == def_mean) else float("nan")
        summary_rows.append({
            "model": model,
            "K": str(K),
            "metric": "cos_ecapa",
            "baseline_mean": f"{bl_mean:.4f}" if bl_mean == bl_mean else "nan",
            "defended_mean": f"{def_mean:.4f}" if def_mean == def_mean else "nan",
            "delta": f"{delta:.4f}" if delta == delta else "nan",
        })
    summary_csv = paper_dir / "e2e_cloning_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "K", "metric", "baseline_mean", "defended_mean", "delta"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote {summary_csv}")
    print(f"Effective sessions: {len(session_ids)}; cloning model: {model_tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
