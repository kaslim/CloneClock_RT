#!/usr/bin/env python3
"""Evaluate session-pool aggregation attack with unified definitions."""

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
from jiwer import wer as jiwer_wer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics.speaker import SpeakerMetric
from src.metrics.session_attack import (
    aggregate_cos_to_ref,
    pick_indices,
    sanity_compare_k,
    sanity_debug_topk,
)
from src.models.defense_stftmask import STFTMaskDefense
from src.transforms.telephony import TelephonyTransform

try:
    from pystoi import stoi as pystoi_stoi  # type: ignore
except Exception:
    pystoi_stoi = None


def load_wav(path: Path) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype='float32')
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x, sr


def align_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    if n <= 0:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)
    return a[:n].astype(np.float32), b[:n].astype(np.float32)


def safe_stoi(ref: np.ndarray, hyp: np.ndarray, sample_rate: int) -> float:
    if pystoi_stoi is None:
        return float('nan')
    ra, ha = align_pair(ref, hyp)
    try:
        return float(pystoi_stoi(ra, ha, sample_rate, extended=False))
    except Exception:
        return float('nan')


def finite_mean(vals: List[float]) -> float:
    a = np.asarray(vals, dtype=np.float32)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float('nan')
    return float(a.mean())


def finite_std(vals: List[float]) -> float:
    a = np.asarray(vals, dtype=np.float32)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float('nan')
    return float(a.std())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Session-pool eval (random_K + bestK).')
    p.add_argument('--dataset', type=str, default='vctk', choices=['vctk', 'librispeech'])
    p.add_argument('--session_pool_csv', type=str, required=True)
    p.add_argument('--method_name', type=str, required=True)
    p.add_argument('--defense_checkpoint', type=str, default='')
    p.add_argument('--max_utters', type=int, default=0)
    p.add_argument('--quality_max_utters', type=int, default=50)
    p.add_argument('--group_key', type=str, default='auto', choices=['auto', 'utter_id', 'session_id'])
    p.add_argument('--compute_quality', action='store_true')
    p.add_argument('--sanity_output', type=str, default='')
    p.add_argument('--sanity_stats_output', type=str, default='')
    p.add_argument('--sanity_n_sessions', type=int, default=20)
    p.add_argument('--sanity_random_trials', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--k_list', type=str, default='1,2,4,8,16')
    p.add_argument(
        '--strategies',
        type=str,
        default='random_K,bestK_by_clean_consistency',
        help='Comma-separated strategies, e.g. random_K,bestK_by_clean_consistency,bestK_by_ref_similarity',
    )
    p.add_argument(
        '--targeted_selection_source',
        type=str,
        default='clean',
        choices=['clean', 'defended', 'tele_defended'],
        help='Window source for targeted bestK selection.',
    )
    p.add_argument('--telephony_config', type=str, default=str(REPO_ROOT / 'configs' / 'telephony.yaml'))
    p.add_argument('--eval_config', type=str, default=str(REPO_ROOT / 'configs' / 'eval.yaml'))
    p.add_argument('--speaker_encoder_name', type=str, default='')
    p.add_argument('--speaker_checkpoint_dir', type=str, default='')
    p.add_argument('--speaker_checkpoint_dir_xvector', type=str, default='')
    p.add_argument('--telephony_codec', type=str, default='none', choices=['none', 'opus', 'g711'])
    p.add_argument('--telephony_codec_bitrate', type=str, default='16k')
    p.add_argument('--output_tag', type=str, default='')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    ks = [int(x) for x in args.k_list.split(',') if x.strip()]
    strategies = [x.strip() for x in str(args.strategies).split(",") if x.strip()]
    if not strategies:
        strategies = ["random_K", "bestK_by_clean_consistency"]

    with open(args.eval_config, 'r', encoding='utf-8') as f:
        import yaml
        eval_cfg = yaml.safe_load(f) or {}
    with open(args.telephony_config, 'r', encoding='utf-8') as f:
        import yaml
        tp_cfg = yaml.safe_load(f) or {}

    if args.telephony_codec != 'none':
        tp_cfg.setdefault('codec', {})
        tp_cfg['codec']['enabled'] = True
        tp_cfg['codec']['name'] = str(args.telephony_codec)
        if str(args.telephony_codec).lower() == 'opus':
            tp_cfg['codec']['bitrate'] = str(args.telephony_codec_bitrate)
    elif 'codec' in tp_cfg:
        tp_cfg.setdefault('codec', {})
        tp_cfg['codec']['enabled'] = False

    device = eval_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    spk_cfg = dict(eval_cfg.get('speaker', eval_cfg.get('speaker_metric', {})) or {})
    spk_cfg.setdefault("device", str(device))
    if args.speaker_encoder_name:
        spk_cfg["encoder_name"] = str(args.speaker_encoder_name)
    if args.speaker_checkpoint_dir:
        spk_cfg["checkpoint_dir"] = str(args.speaker_checkpoint_dir)
    if args.speaker_checkpoint_dir_xvector:
        spk_cfg["checkpoint_dir_xvector"] = str(args.speaker_checkpoint_dir_xvector)
    speaker_metric = SpeakerMetric(spk_cfg)
    asr_metric = None
    if bool(args.compute_quality):
        from src.metrics.asr import ASRMetric
        asr_metric = ASRMetric(eval_cfg.get('asr', {}))
    telephony = TelephonyTransform(tp_cfg)

    defense_model = None
    if args.defense_checkpoint:
        payload = torch.load(args.defense_checkpoint, map_location=device)
        cfg = payload.get('model_config', {}) if isinstance(payload, dict) else {}
        defense_model = STFTMaskDefense(cfg).to(device).eval()
        if isinstance(payload, dict) and 'model_state_dict' in payload:
            defense_model.load_state_dict(payload['model_state_dict'], strict=True)

    rows = []
    with open(args.session_pool_csv, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    key_field = str(args.group_key)
    if key_field == "auto":
        if rows and "session_id" in rows[0]:
            key_field = "session_id"
        elif rows and "utter_id" in rows[0]:
            key_field = "utter_id"
        else:
            key_field = "utter_id"

    by_utter: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        uid = r.get(key_field, r.get('utter_id', 'unknown'))
        by_utter[(r['speaker_id'], uid, r['path'])].append(r)

    utter_keys = sorted(by_utter.keys())
    if args.max_utters > 0:
        utter_keys = utter_keys[: args.max_utters]
    sanity_n = max(0, int(args.sanity_n_sessions))
    sanity_pick = set(rng.sample(range(len(utter_keys)), k=min(len(utter_keys), sanity_n))) if utter_keys else set()

    curve = {(st, k): [] for st in strategies for k in ks}
    stoi_abs = []
    wer_abs = []

    sanity_lines: List[str] = []
    sanity_vals = {"rand": [], "cons": [], "tgt": []}
    sanity_vals_ext = {"tgt_clean": [], "tgt_defended": [], "tgt_tele_defended": []}
    sanity_fail_debug: List[str] = []
    for uidx, (spk, utt, rel_path) in enumerate(utter_keys):
        wav_path = REPO_ROOT / rel_path
        if not wav_path.exists():
            wav_path = REPO_ROOT / 'data' / 'processed' / args.dataset / 'wav16k' / rel_path
        clean, sr = load_wav(wav_path)
        if defense_model is not None:
            with torch.inference_mode():
                proc = defense_model(torch.from_numpy(clean).to(device)).detach().cpu().numpy().astype(np.float32)
        else:
            proc = clean

        tp = telephony.sample_params()
        tel_clean_shared = telephony.apply_with_params(clean, sample_rate=sr, params=tp)
        tel_proc_shared = telephony.apply_with_params(proc, sample_rate=sr, params=tp)

        win_rows = sorted(by_utter[(spk, utt, rel_path)], key=lambda x: x['chunk_id'])
        clean_ws = []
        proc_ws = []
        proc_ws_raw = []
        need_raw_def = bool(args.targeted_selection_source == "defended" or sanity_n > 0)
        for wr in win_rows:
            s = int(round(float(wr['start_sec']) * sr))
            d = int(round(float(wr['dur_sec']) * sr))
            cseg = tel_clean_shared[s : s + d]
            pseg = tel_proc_shared[s : s + d]
            if len(cseg) < d or len(pseg) < d:
                continue
            ce = speaker_metric.embed(torch.from_numpy(cseg), sr)
            pe = speaker_metric.embed(torch.from_numpy(pseg), sr)
            clean_ws.append(F.normalize(ce, dim=0))
            proc_ws.append(F.normalize(pe, dim=0))
            if need_raw_def:
                pseg_raw = proc[s : s + d]
                if len(pseg_raw) >= d:
                    pre = speaker_metric.embed(torch.from_numpy(pseg_raw), sr)
                    proc_ws_raw.append(F.normalize(pre, dim=0))

        if len(clean_ws) < 2 or len(proc_ws) < 2:
            continue

        clean_z = torch.stack(clean_ws, dim=0)
        def_z = torch.stack(proc_ws, dim=0)
        def_raw_z = torch.stack(proc_ws_raw, dim=0) if len(proc_ws_raw) == len(proc_ws) and len(proc_ws_raw) > 0 else None

        for st in strategies:
            for k in ks:
                idx = pick_indices(st, clean_z, k, rng)
                if st == "bestK_by_ref_similarity":
                    if args.targeted_selection_source == "defended" and def_raw_z is not None:
                        idx = pick_indices(st, clean_z, k, rng, selection_z=def_raw_z)
                    elif args.targeted_selection_source == "tele_defended":
                        idx = pick_indices(st, clean_z, k, rng, selection_z=def_z)
                    else:
                        idx = pick_indices(st, clean_z, k, rng, selection_z=clean_z)
                cos = aggregate_cos_to_ref(def_z, clean_z, idx)
                curve[(st, k)].append(cos)

        if uidx in sanity_pick:
            vv = sanity_compare_k(
                def_z=def_z,
                clean_z=clean_z,
                rng=rng,
                n_random_trials=max(1, int(args.sanity_random_trials)),
            )
            idx_cons = pick_indices("bestK_by_clean_consistency", clean_z, 16, rng)
            idx_t_clean = pick_indices("bestK_by_ref_similarity", clean_z, 16, rng, selection_z=clean_z)
            idx_t_def = pick_indices(
                "bestK_by_ref_similarity",
                clean_z,
                16,
                rng,
                selection_z=def_raw_z if def_raw_z is not None else def_z,
            )
            idx_t_tel = pick_indices("bestK_by_ref_similarity", clean_z, 16, rng, selection_z=def_z)
            cos_cons = aggregate_cos_to_ref(def_z, clean_z, idx_cons)
            cos_t_clean = aggregate_cos_to_ref(def_z, clean_z, idx_t_clean)
            cos_t_def = aggregate_cos_to_ref(def_z, clean_z, idx_t_def)
            cos_t_tel = aggregate_cos_to_ref(def_z, clean_z, idx_t_tel)
            line = (
                f"speaker={spk} key={utt} "
                f"K1_rand={vv['cos_K1_random']:.4f} "
                f"K16_rand={vv['cos_K16_random']:.4f} "
                f"K16_cons={cos_cons:.4f} "
                f"K16_tgt_clean={cos_t_clean:.4f} "
                f"K16_tgt_def={cos_t_def:.4f} "
                f"K16_tgt_teldef={cos_t_tel:.4f}"
            )
            sanity_lines.append(line)
            sanity_vals["rand"].append(vv["cos_K16_random"])
            sanity_vals["cons"].append(cos_cons)
            sanity_vals["tgt"].append(cos_t_tel)
            sanity_vals_ext["tgt_clean"].append(cos_t_clean)
            sanity_vals_ext["tgt_defended"].append(cos_t_def)
            sanity_vals_ext["tgt_tele_defended"].append(cos_t_tel)
            if not (
                cos_t_tel >= cos_cons >= vv["cos_K16_random"]
            ):
                dbg = sanity_debug_topk(def_z=def_z, clean_z=clean_z, k=16, debug_topn=5)
                sanity_fail_debug.append(
                    json.dumps(
                        {
                            "speaker": spk,
                            "key": utt,
                            "vals": {
                                "K16_rand": vv["cos_K16_random"],
                                "K16_cons": cos_cons,
                                "K16_tgt_clean": cos_t_clean,
                                "K16_tgt_def": cos_t_def,
                                "K16_tgt_teldef": cos_t_tel,
                            },
                            "debug": dbg,
                        },
                        ensure_ascii=False,
                    )
                )

        if asr_metric is not None and uidx < int(args.quality_max_utters):
            tel_clean_abs = telephony(clean, sample_rate=sr)
            tel_proc_abs = telephony(proc, sample_rate=sr)
            clean_text = asr_metric.transcribe(torch.from_numpy(clean), sr)
            proc_text = asr_metric.transcribe(torch.from_numpy(tel_proc_abs), sr)
            wer_abs.append(float(jiwer_wer([clean_text], [proc_text])))
            stoi_abs.append(safe_stoi(tel_clean_abs, tel_proc_abs, sr))

    out_dir = REPO_ROOT / 'artifacts' / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_tag}" if str(args.output_tag).strip() else ""
    curve_out = out_dir / f'session_attack_curve_abs_{args.method_name}{suffix}.csv'
    with open(curve_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'method', 'strategy', 'K', 'cos_mean', 'cos_std', 'count'])
        w.writeheader()
        for st in strategies:
            for k in ks:
                vals = curve[(st, k)]
                w.writerow({
                    'model': args.method_name,
                    'method': args.method_name,
                    'strategy': st,
                    'K': k,
                    'cos_mean': finite_mean(vals),
                    'cos_std': finite_std(vals),
                    'count': len(vals),
                })

    summary = {
        'method': args.method_name,
        'n_utters': len(utter_keys),
        'quality_utters': min(len(utter_keys), int(args.quality_max_utters)),
        'stoi_abs': finite_mean(stoi_abs),
        'wer_abs': finite_mean(wer_abs),
        'curve_csv': str(curve_out),
    }
    strat_rows: List[dict] = []
    for st in strategies:
        by_k = {int(k): finite_mean(curve[(st, k)]) for k in ks}
        valid_auc = [by_k[k] for k in ks if np.isfinite(by_k[k])]
        auc = float(np.mean(valid_auc)) if valid_auc else float('nan')
        slope = float(by_k.get(16, float('nan')) - by_k.get(1, float('nan')))
        strat_rows.append(
            {
                "method": args.method_name,
                "strategy": st,
                "K1_mean": by_k.get(1, float('nan')),
                "K16_mean": by_k.get(16, float('nan')),
                "AUC": auc,
                "slope_16_1": slope,
                "count": len(curve[(st, ks[0])]) if ks else 0,
            }
        )
    summary_csv = out_dir / f"session_attack_summary_abs_{args.method_name}{suffix}.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["method", "strategy", "K1_mean", "K16_mean", "AUC", "slope_16_1", "count"],
        )
        w.writeheader()
        w.writerows(strat_rows)
    summary["summary_csv"] = str(summary_csv)
    summary["metadata"] = {
        "speaker_encoder_name": spk_cfg.get("encoder_name", ""),
        "targeted_selection_source": str(args.targeted_selection_source),
        "telephony_codec": str(args.telephony_codec),
        "telephony_codec_bitrate": str(args.telephony_codec_bitrate),
        "strategies": strategies,
    }

    if sanity_vals["rand"]:
        rand_m, rand_s = finite_mean(sanity_vals["rand"]), finite_std(sanity_vals["rand"])
        cons_m, cons_s = finite_mean(sanity_vals["cons"]), finite_std(sanity_vals["cons"])
        tgt_m, tgt_s = finite_mean(sanity_vals["tgt"]), finite_std(sanity_vals["tgt"])
        tgtc_m, tgtc_s = finite_mean(sanity_vals_ext["tgt_clean"]), finite_std(sanity_vals_ext["tgt_clean"])
        tgtd_m, tgtd_s = finite_mean(sanity_vals_ext["tgt_defended"]), finite_std(sanity_vals_ext["tgt_defended"])
        tgtt_m, tgtt_s = finite_mean(sanity_vals_ext["tgt_tele_defended"]), finite_std(sanity_vals_ext["tgt_tele_defended"])
        sanity_ok = bool(tgt_m >= cons_m >= rand_m)
        sanity_realistic_ok = bool(tgtd_m >= rand_m)
        sanity_oracle_upper_ok = bool(tgtc_m >= tgtd_m)
        sanity_stats_lines = [
            f"n_sessions={len(sanity_vals['rand'])}",
            f"K16_rand_mean_std={rand_m:.6f},{rand_s:.6f}",
            f"K16_cons_mean_std={cons_m:.6f},{cons_s:.6f}",
            f"K16_tgt_clean_mean_std={tgtc_m:.6f},{tgtc_s:.6f}",
            f"K16_tgt_defended_mean_std={tgtd_m:.6f},{tgtd_s:.6f}",
            f"K16_tgt_tele_defended_mean_std={tgtt_m:.6f},{tgtt_s:.6f}",
            f"order_mean_pass={sanity_ok}",
            f"realistic_ge_rand_pass={sanity_realistic_ok}",
            f"oracle_ge_realistic_pass={sanity_oracle_upper_ok}",
            "note=跨策略攻击强度请优先比较K16_mean/AUC；slope_16_1仅用于同策略下防御比较",
        ]
        if sanity_fail_debug:
            sanity_stats_lines.append("debug_top5_failures_begin")
            sanity_stats_lines.extend(sanity_fail_debug[:5])
            sanity_stats_lines.append("debug_top5_failures_end")
        if args.sanity_stats_output:
            Path(args.sanity_stats_output).write_text("\n".join(sanity_stats_lines) + "\n", encoding="utf-8")
        summary["sanity"] = {
            "n_sessions": len(sanity_vals["rand"]),
            "K16_rand_mean": rand_m,
            "K16_cons_mean": cons_m,
            "K16_tgt_clean_mean": tgtc_m,
            "K16_tgt_defended_mean": tgtd_m,
            "K16_tgt_tele_defended_mean": tgtt_m,
            "order_mean_pass": sanity_ok,
            "realistic_ge_rand_pass": sanity_realistic_ok,
            "oracle_ge_realistic_pass": sanity_oracle_upper_ok,
            "n_fail_cases": len(sanity_fail_debug),
        }
    out_json = out_dir / f'session_attack_eval_{args.method_name}{suffix}.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if args.sanity_output:
        Path(args.sanity_output).write_text("\n".join(sanity_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
