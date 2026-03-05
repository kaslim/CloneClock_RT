#!/usr/bin/env python3
"""Export paper CSVs to LaTeX tables (booktabs, Interspeech-style)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, List

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_TABLES = REPO_ROOT / "artifacts" / "tables" / "paper"


def fmt_num(x: Any, ndec: int = 4) -> str:
    try:
        v = float(x)
        if v != v:  # nan
            return "---"
        return f"{v:.{ndec}f}"
    except (TypeError, ValueError):
        return str(x).replace("_", "\\_")


def _is_numeric(v: str) -> bool:
    if not v:
        return False
    s = v.replace(".", "").replace("-", "").replace("e", "").strip()
    return s.isdigit()


def csv_to_tex(
    csv_path: Path,
    tex_path: Path,
    label: str,
    caption: str,
    ndec: int = 4,
    cols_float: set[str] | None = None,
    rows_override: list[dict] | None = None,
) -> None:
    cols_float = cols_float or set()
    if rows_override is not None:
        rows = rows_override
    else:
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    if not rows:
        tex_path.write_text("% empty table\n", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    header = " & ".join(h.replace("_", "\\_") for h in fieldnames)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label}}}",
        "\\begin{tabular}{" + "l" + "r" * (len(fieldnames) - 1) + "}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        cells = []
        for k in fieldnames:
            v = r.get(k, "")
            if v == "n/a" or (str(v).strip().lower() == "n/a"):
                cells.append("n/a")
            elif k in cols_float or _is_numeric(str(v)):
                cells.append(fmt_num(v, ndec))
            else:
                cells.append(str(v).replace("_", "\\_"))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Convert paper CSVs to LaTeX tables.")
    p.add_argument("--tables_dir", type=str, default=str(PAPER_TABLES))
    p.add_argument("--ndec", type=int, default=4)
    args = p.parse_args()
    root = Path(args.tables_dir)
    ndec = max(1, min(6, args.ndec))

    # main_ecapa_targeted_teledef.csv -> main_ecapa_targeted_teledef.tex
    main_csv = root / "main_ecapa_targeted_teledef.csv"
    if main_csv.exists():
        csv_to_tex(
            main_csv,
            root / "main_ecapa_targeted_teledef.tex",
            "main",
            "Main result (ECAPA, v3 session pool). BestK targeted, selection\\_source=tele\\_defended, K=1,2,4,8,16. Lower is better.",
            ndec=ndec,
            cols_float={"K16_mean", "AUC", "slope_16_1"},
        )
        print(root / "main_ecapa_targeted_teledef.tex")

    # codec_robustness_targeted_teledef.csv -> codec_robustness_targeted_teledef.tex
    codec_csv = root / "codec_robustness_targeted_teledef.csv"
    if codec_csv.exists():
        csv_to_tex(
            codec_csv,
            root / "codec_robustness_targeted_teledef.tex",
            "codec",
            "Codec robustness (Opus / G.711). Targeted tele\\_defended. Lower K16\\_mean and AUC are better.",
            ndec=ndec,
            cols_float={"K16_mean", "AUC", "slope_16_1"},
        )
        print(root / "codec_robustness_targeted_teledef.tex")

    # cross_encoder_asv_eer.csv -> cross_encoder_asv_margin_eer.tex (columns: encoder, method, margin_mean, eer, n_target, n_impostor)
    cross_csv = root / "cross_encoder_asv_eer.csv"
    if cross_csv.exists():
        csv_to_tex(
            cross_csv,
            root / "cross_encoder_asv_margin_eer.tex",
            "crossenc",
            "Cross-encoder ASV risk. EER and margin\\_mean (this test set EER=0, margin is more sensitive). Targeted K16, selection\\_source=tele\\_defended. Lower is better.",
            ndec=ndec,
            cols_float={"margin_mean", "eer"},
        )
        print(root / "cross_encoder_asv_margin_eer.tex")

    # latency_v0p1_B_v2.csv -> latency_v0p1_B_v2.tex (baseline+cuda peak_mem_mb -> n/a)
    lat_csv = root / "latency_v0p1_B_v2.csv"
    if lat_csv.exists():
        with open(lat_csv, "r", encoding="utf-8") as f:
            lat_rows = list(csv.DictReader(f))
        for r in lat_rows:
            if str(r.get("method", "")) == "baseline" and str(r.get("device", "")) == "cuda":
                r["peak_mem_mb"] = "n/a"
        csv_to_tex(
            lat_csv,
            root / "latency_v0p1_B_v2.tex",
            "latency",
            "Streaming latency (v2): warmup 50 chunks, GPU sync. Lower RTF and avg\\_ms are better. Baseline (cuda) does not run GPU kernel, peak\\_mem\\_mb n/a.",
            ndec=ndec,
            cols_float={"avg_ms", "p95_ms", "rtf", "peak_mem_mb"},
            rows_override=lat_rows,
        )
        print(root / "latency_v0p1_B_v2.tex")

    # quality_summary.csv -> quality_summary.tex (tab:quality)
    quality_csv = root / "quality_summary.csv"
    if quality_csv.exists():
        csv_to_tex(
            quality_csv,
            root / "quality_summary.tex",
            "quality",
            "Quality under telephony: WER (vs. reference) and STOI (tele\\_clean vs tele\\_def). Lower WER and higher STOI are better. VCTK test utterances.",
            ndec=ndec,
            cols_float={"wer_mean", "wer_std", "stoi_mean", "stoi_std"},
        )
        print(root / "quality_summary.tex")

    # e2e_cloning_summary.csv -> e2e_cloning_summary.tex (tab:e2e)
    e2e_csv = root / "e2e_cloning_summary.csv"
    if e2e_csv.exists():
        csv_to_tex(
            e2e_csv,
            root / "e2e_cloning_summary.tex",
            "e2e",
            "End-to-end cloning: cos(synth, target\\_ref) by condition (baseline\\_ref vs defended\\_ref). Lower defended\\_mean is better.",
            ndec=ndec,
            cols_float={"baseline_mean", "defended_mean", "delta"},
        )
        print(root / "e2e_cloning_summary.tex")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
