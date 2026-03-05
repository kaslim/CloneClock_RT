#!/usr/bin/env python3
"""ASV risk evaluation (EER) on v3 session pool, targeted tele-defended setup."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics.session_attack import pick_indices
from src.metrics.speaker import SpeakerMetric, compute_eer
from src.models.defense_stftmask import STFTMaskDefense
from src.transforms.telephony import TelephonyTransform


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute ASV EER for baseline/v0p1_B on v3 targeted tele-defended setting.")
    p.add_argument("--dataset", type=str, default="vctk", choices=["vctk", "librispeech"])
    p.add_argument("--session_pool_csv", type=str, required=True)
    p.add_argument(
        "--method_specs",
        type=str,
        required=True,
        help="Comma-separated name:checkpoint, use empty checkpoint for baseline, e.g. baseline:,v0p1_B:/abs/path.pt",
    )
    p.add_argument("--speaker_encoder_name", type=str, default="speechbrain_ecapa")
    p.add_argument("--speaker_checkpoint_dir", type=str, default="")
    p.add_argument("--speaker_checkpoint_dir_xvector", type=str, default="")
    p.add_argument("--telephony_config", type=str, default=str(REPO_ROOT / "configs" / "telephony.yaml"))
    p.add_argument("--group_key", type=str, default="session_id", choices=["session_id", "utter_id"])
    p.add_argument("--selection_source", type=str, default="tele_defended", choices=["clean", "defended", "tele_defended"])
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--impostors_per_target", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument("--output_margin_csv", type=str, default="")
    return p.parse_args()


def load_wav(path: Path) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32), int(sr)


def parse_method_specs(spec: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            out.append((tok, ""))
        else:
            name, ckpt = tok.split(":", 1)
            out.append((name.strip(), ckpt.strip()))
    return out


def load_defense(ckpt: str, device: str) -> STFTMaskDefense | None:
    if not ckpt:
        return None
    payload = torch.load(ckpt, map_location=device)
    cfg = payload.get("model_config", {}) if isinstance(payload, dict) else {}
    model = STFTMaskDefense(cfg).to(device).eval()
    if isinstance(payload, dict) and "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))

    with open(args.telephony_config, "r", encoding="utf-8") as f:
        import yaml

        tp_cfg = yaml.safe_load(f) or {}
    telephony = TelephonyTransform(tp_cfg)

    spk_cfg: Dict[str, str] = {"encoder_name": str(args.speaker_encoder_name), "device": "cuda" if torch.cuda.is_available() else "cpu"}
    if args.speaker_checkpoint_dir:
        spk_cfg["checkpoint_dir"] = str(args.speaker_checkpoint_dir)
    if args.speaker_checkpoint_dir_xvector:
        spk_cfg["checkpoint_dir_xvector"] = str(args.speaker_checkpoint_dir_xvector)
    speaker_metric = SpeakerMetric(spk_cfg)

    with open(args.session_pool_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_group: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        gid = str(r.get(args.group_key, r.get("session_id", r.get("utter_id", "unknown"))))
        by_group[(r["speaker_id"], gid, r["path"])].append(r)
    keys = sorted(by_group.keys())

    method_specs = parse_method_specs(args.method_specs)
    out_rows: List[Dict[str, str]] = []
    margin_rows: List[Dict[str, str]] = []

    for method_name, ckpt in method_specs:
        defense_model = load_defense(ckpt, str(speaker_metric.device))
        reps: List[Dict[str, object]] = []
        for spk, gid, rel_path in keys:
            wav_path = REPO_ROOT / rel_path
            if not wav_path.exists():
                wav_path = REPO_ROOT / "data" / "processed" / args.dataset / "wav16k" / rel_path
            if not wav_path.exists():
                continue
            clean, sr = load_wav(wav_path)
            if defense_model is not None:
                with torch.inference_mode():
                    proc = defense_model(torch.from_numpy(clean).to(speaker_metric.device)).detach().cpu().numpy().astype(np.float32)
            else:
                proc = clean

            tp = telephony.sample_params()
            tel_clean = telephony.apply_with_params(clean, sample_rate=sr, params=tp)
            tel_proc = telephony.apply_with_params(proc, sample_rate=sr, params=tp)

            win_rows = sorted(by_group[(spk, gid, rel_path)], key=lambda x: x["chunk_id"])
            clean_ws: List[torch.Tensor] = []
            def_ws: List[torch.Tensor] = []
            raw_def_ws: List[torch.Tensor] = []
            for wr in win_rows:
                s = int(round(float(wr["start_sec"]) * sr))
                d = int(round(float(wr["dur_sec"]) * sr))
                cseg = tel_clean[s : s + d]
                dseg = tel_proc[s : s + d]
                if len(cseg) < d or len(dseg) < d:
                    continue
                clean_ws.append(speaker_metric.embed(torch.from_numpy(cseg), sr))
                def_ws.append(speaker_metric.embed(torch.from_numpy(dseg), sr))
                raw = proc[s : s + d]
                if len(raw) >= d:
                    raw_def_ws.append(speaker_metric.embed(torch.from_numpy(raw), sr))

            if len(clean_ws) < 2 or len(def_ws) < 2:
                continue
            clean_z = F.normalize(torch.stack(clean_ws, dim=0), dim=-1)
            def_z = F.normalize(torch.stack(def_ws, dim=0), dim=-1)
            raw_def_z = F.normalize(torch.stack(raw_def_ws, dim=0), dim=-1) if len(raw_def_ws) == len(def_ws) and len(raw_def_ws) > 0 else None

            if args.selection_source == "clean":
                sel_z = clean_z
            elif args.selection_source == "defended" and raw_def_z is not None:
                sel_z = raw_def_z
            else:
                sel_z = def_z

            idx = pick_indices("bestK_by_ref_similarity", clean_z, int(args.k), rng, selection_z=sel_z)
            e_def = F.normalize(def_z[idx].mean(dim=0), dim=0)
            e_ref = F.normalize(clean_z.mean(dim=0), dim=0)
            reps.append({"speaker_id": spk, "e_ref": e_ref, "e_def": e_def})

        if not reps:
            out_rows.append(
                {
                    "encoder": args.speaker_encoder_name,
                    "method": method_name,
                    "eer": "nan",
                    "n_target": "0",
                    "n_impostor": "0",
                }
            )
            continue

        target_scores: List[float] = []
        impostor_scores: List[float] = []
        margins: List[float] = []
        by_spk_idx: Dict[str, List[int]] = defaultdict(list)
        for i, rr in enumerate(reps):
            by_spk_idx[str(rr["speaker_id"])].append(i)

        all_idx = list(range(len(reps)))
        for i, rr in enumerate(reps):
            e_ref = rr["e_ref"]
            e_def = rr["e_def"]
            target_scores.append(float(torch.dot(e_ref, e_def).item()))

            candidates = [j for j in all_idx if str(reps[j]["speaker_id"]) != str(rr["speaker_id"])]
            if not candidates:
                continue
            k_imp = min(len(candidates), int(args.impostors_per_target))
            picked = rng.sample(candidates, k=k_imp) if len(candidates) >= k_imp else candidates
            local_imp: List[float] = []
            for j in picked:
                s_imp = float(torch.dot(e_ref, reps[j]["e_def"]).item())
                impostor_scores.append(s_imp)
                local_imp.append(s_imp)
            if local_imp:
                margins.append(float(target_scores[-1] - max(local_imp)))

        scores = np.asarray(target_scores + impostor_scores, dtype=np.float32)
        labels = np.asarray([1] * len(target_scores) + [0] * len(impostor_scores), dtype=np.int32)
        eer = compute_eer(scores=scores, labels=labels) if len(target_scores) > 0 and len(impostor_scores) > 0 else float("nan")
        out_rows.append(
            {
                "encoder": args.speaker_encoder_name,
                "method": method_name,
                "eer": f"{eer:.6f}" if np.isfinite(eer) else "nan",
                "n_target": str(len(target_scores)),
                "n_impostor": str(len(impostor_scores)),
            }
        )
        margin_rows.append(
            {
                "encoder": args.speaker_encoder_name,
                "method": method_name,
                "margin_mean": f"{float(np.mean(np.asarray(margins, dtype=np.float32))):.6f}" if margins else "nan",
                "n_target": str(len(target_scores)),
                "n_impostor": str(len(impostor_scores)),
            }
        )

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["encoder", "method", "eer", "n_target", "n_impostor"])
        w.writeheader()
        w.writerows(out_rows)

    margin_csv = None
    if str(args.output_margin_csv).strip():
        margin_csv = Path(args.output_margin_csv)
        margin_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(margin_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["encoder", "method", "margin_mean", "n_target", "n_impostor"])
            w.writeheader()
            w.writerows(margin_rows)

    print(json.dumps({"output_csv": str(out_csv), "rows": out_rows, "output_margin_csv": str(margin_csv) if margin_csv else ""}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

