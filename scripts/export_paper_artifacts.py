#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def f(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def main() -> int:
    tables = REPO_ROOT / "artifacts" / "tables"
    out_tables = tables / "paper"
    out_figs = REPO_ROOT / "artifacts" / "figures" / "paper"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    ecapa = read_csv(tables / "ens_ecapa_sweep_v3_bestK_targeted_teledef.csv")
    write_csv(out_tables / "main_ecapa_targeted_teledef.csv", ecapa, ["method", "K16_mean", "AUC", "slope_16_1"])
    m = [r["method"] for r in ecapa]
    k16 = [f(r["K16_mean"]) for r in ecapa]
    auc = [f(r["AUC"]) for r in ecapa]
    x = np.arange(len(m))
    w = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - w / 2, k16, width=w, label="K16_mean")
    plt.bar(x + w / 2, auc, width=w, label="AUC")
    plt.xticks(x, m)
    plt.ylabel("Score")
    plt.title("Main Result ECAPA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_figs / "main_ecapa_k16_auc.png", dpi=200)
    plt.close()

    codec_rows: List[Dict[str, str]] = []
    p_opus = tables / "codec_opus_sweep_v3_bestK_targeted_teledef.csv"
    p_g711 = tables / "codec_g711_sweep_v3_bestK_targeted_teledef.csv"
    if p_opus.exists():
        codec_rows.extend(read_csv(p_opus))
    if p_g711.exists():
        codec_rows.extend(read_csv(p_g711))
    write_csv(
        out_tables / "codec_robustness_targeted_teledef.csv",
        codec_rows,
        ["method", "K16_mean", "AUC", "slope_16_1", "codec", "codec_bitrate", "selection_source", "encoder"],
    )
    if codec_rows:
        codecs = sorted(list({r.get("codec", "none") for r in codec_rows}))
        xpos = np.arange(len(codecs))
        width = 0.35
        b = []
        v = []
        for c in codecs:
            rb = next((r for r in codec_rows if r.get("codec", "") == c and r["method"] == "baseline"), None)
            rv = next((r for r in codec_rows if r.get("codec", "") == c and r["method"] == "v0p1_B"), None)
            b.append(f(rb["K16_mean"]) if rb else float("nan"))
            v.append(f(rv["K16_mean"]) if rv else float("nan"))
        plt.figure(figsize=(6, 4))
        plt.bar(xpos - width / 2, b, width=width, label="baseline")
        plt.bar(xpos + width / 2, v, width=width, label="v0p1_B")
        plt.xticks(xpos, codecs)
        plt.ylabel("K16_mean")
        plt.title("Codec Robustness")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_figs / "codec_robustness_k16.png", dpi=200)
        plt.close()

    asv_rows: List[Dict[str, str]] = []
    p1 = tables / "asv_eer_ecapa_v3_targeted_teledef.csv"
    p2 = tables / "asv_eer_xvector_v3_targeted_teledef.csv"
    m1 = tables / "asv_margin_ecapa_v3_targeted_teledef.csv"
    m2 = tables / "asv_margin_xvector_v3_targeted_teledef.csv"
    if p1.exists():
        asv_rows.extend(read_csv(p1))
    if p2.exists():
        asv_rows.extend(read_csv(p2))
    margin_map: Dict[tuple, str] = {}
    for path in (m1, m2):
        if path.exists():
            for row in read_csv(path):
                key = (row.get("encoder", ""), row.get("method", ""))
                margin_map[key] = row.get("margin_mean", "")
    for row in asv_rows:
        key = (row.get("encoder", ""), row.get("method", ""))
        row["margin_mean"] = margin_map.get(key, "")
    write_csv(out_tables / "cross_encoder_asv_eer.csv", asv_rows, ["encoder", "method", "margin_mean", "eer", "n_target", "n_impostor"])
    if asv_rows:
        encs = ["speechbrain_ecapa", "speechbrain_xvector"]
        labels = ["ecapa", "xvector"]
        b = []
        v = []
        for e in encs:
            rb = next((r for r in asv_rows if r["encoder"] == e and r["method"] == "baseline"), None)
            rv = next((r for r in asv_rows if r["encoder"] == e and r["method"] == "v0p1_B"), None)
            b.append(f(rb["eer"]) if rb else float("nan"))
            v.append(f(rv["eer"]) if rv else float("nan"))
        x2 = np.arange(len(encs))
        plt.figure(figsize=(6, 4))
        plt.bar(x2 - w / 2, b, width=w, label="baseline")
        plt.bar(x2 + w / 2, v, width=w, label="v0p1_B")
        plt.xticks(x2, labels)
        plt.ylabel("EER")
        plt.title("Cross-Encoder ASV Risk")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_figs / "cross_encoder_asv_eer.png", dpi=200)
        plt.close()

    lat = read_csv(tables / "latency_v0p1_B_v2.csv")
    write_csv(out_tables / "latency_v0p1_B_v2.csv", lat, ["method", "device", "chunk_ms", "avg_ms", "p95_ms", "rtf", "peak_mem_mb"])

    # Quality summary: build from quality/ if paper/quality_summary.csv missing
    if not (out_tables / "quality_summary.csv").exists() and (tables / "quality").exists():
        quality_rows = []
        for tel in ["none", "opus16k", "g711"]:
            p = tables / "quality" / f"quality_eval_{tel}.csv"
            if p.exists():
                quality_rows.extend(read_csv(p))
        if quality_rows:
            from collections import defaultdict
            agg = defaultdict(lambda: {"wer": [], "stoi": []})
            for r in quality_rows:
                k = (r.get("telephony", ""), r.get("defense", ""))
                w, s = r.get("wer"), r.get("stoi")
                if w != "" and str(w).lower() != "nan":
                    try:
                        agg[k]["wer"].append(float(w))
                    except Exception:
                        pass
                if s != "" and str(s).lower() != "nan":
                    try:
                        agg[k]["stoi"].append(float(s))
                    except Exception:
                        pass
            summary = []
            for (tel, defense), v in sorted(agg.items()):
                wers, stois = v["wer"], v["stoi"]
                summary.append({
                    "telephony": tel,
                    "defense": defense,
                    "wer_mean": f"{np.mean(wers):.4f}" if wers else "nan",
                    "wer_std": f"{np.std(wers):.4f}" if len(wers) > 1 else "0",
                    "stoi_mean": f"{np.mean(stois):.4f}" if stois else "nan",
                    "stoi_std": f"{np.std(stois):.4f}" if len(stois) > 1 else "0",
                    "n_utts": str(len(wers) or len(stois)),
                })
            write_csv(out_tables / "quality_summary.csv", summary, ["telephony", "defense", "wer_mean", "wer_std", "stoi_mean", "stoi_std", "n_utts"])
    if (out_tables / "quality_summary.csv").exists():
        qrows = read_csv(out_tables / "quality_summary.csv")
        if qrows:
            codecs = sorted(list({r.get("telephony", "none") for r in qrows}))
            xpos = np.arange(len(codecs))
            width = 0.35
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            for metric, ax, ylabel in [
                ("wer_mean", ax1, "WER (mean)"),
                ("stoi_mean", ax2, "STOI (mean)"),
            ]:
                b, v = [], []
                for c in codecs:
                    rb = next((r for r in qrows if r.get("telephony") == c and r.get("defense") == "baseline"), None)
                    rv = next((r for r in qrows if r.get("telephony") == c and r.get("defense") == "v0p1_B"), None)
                    b.append(f(rb[metric]) if rb else float("nan"))
                    v.append(f(rv[metric]) if rv else float("nan"))
                ax.bar(xpos - width / 2, b, width=width, label="baseline")
                ax.bar(xpos + width / 2, v, width=width, label="v0p1_B")
                ax.set_xticks(xpos)
                ax.set_xticklabels(codecs)
                ax.set_ylabel(ylabel)
                ax.legend()
            plt.suptitle("Quality under telephony (WER vs ref, STOI tele_clean vs tele_def)")
            plt.tight_layout()
            plt.savefig(out_figs / "quality_wer_stoi.png", dpi=200)
            plt.close()
    combos = [("cpu", "20"), ("cpu", "40"), ("cuda", "20"), ("cuda", "40")]
    labels = [f"{d}-{c}ms" for d, c in combos]
    br = []
    vr = []
    for d, c in combos:
        rb = next((r for r in lat if r["method"] == "baseline" and r["device"] == d and r["chunk_ms"] == c), None)
        rv = next((r for r in lat if r["method"] == "v0p1_B" and r["device"] == d and r["chunk_ms"] == c), None)
        br.append(f(rb["rtf"]) if rb else float("nan"))
        vr.append(f(rv["rtf"]) if rv else float("nan"))
    x3 = np.arange(len(combos))
    plt.figure(figsize=(8, 4))
    plt.bar(x3 - w / 2, br, width=w, label="baseline")
    plt.bar(x3 + w / 2, vr, width=w, label="v0p1_B")
    plt.xticks(x3, labels)
    plt.ylabel("RTF")
    plt.title("Streaming RTF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_figs / "latency_rtf_cpu_gpu.png", dpi=200)
    plt.close()

    # E2E cloning summary figure (if present)
    if (out_tables / "e2e_cloning_summary.csv").exists():
        e2e_rows = read_csv(out_tables / "e2e_cloning_summary.csv")
        if e2e_rows:
            models = list({r.get("model", "") for r in e2e_rows})
            use_rows = [r for r in e2e_rows if r.get("model") == models[0]] if models else e2e_rows
            ks = sorted(list({int(r["K"]) for r in use_rows}))
            bl = [f(next((r["baseline_mean"] for r in use_rows if int(r.get("K", 0)) == k), "nan")) for k in ks]
            dv = [f(next((r["defended_mean"] for r in use_rows if int(r.get("K", 0)) == k), "nan")) for k in ks]
            bl = [x if x == x else float("nan") for x in bl]
            dv = [x if x == x else float("nan") for x in dv]
            if ks:
                xpos = np.arange(len(ks))
                w = 0.35
                plt.figure(figsize=(6, 4))
                plt.bar(xpos - w / 2, bl, width=w, label="baseline_ref")
                plt.bar(xpos + w / 2, dv, width=w, label="defended_ref")
                plt.xticks(xpos, [str(k) for k in ks])
                plt.ylabel("cos_ecapa (synth vs target_ref)")
                plt.title("E2E Cloning: Identity Similarity vs K")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_figs / "e2e_cloning_cos_vs_K.png", dpi=200)
                plt.close()

    manifest = {"tables": sorted([str(p) for p in out_tables.glob("*")]), "figures": sorted([str(p) for p in out_figs.glob("*")])}
    with open(out_tables / "paper_manifest.json", "w", encoding="utf-8") as fobj:
        fobj.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

