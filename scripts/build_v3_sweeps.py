#!/usr/bin/env python3
"""Merge per-method v3 session summaries into strategy sweep tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build sweep_v3_* csv tables from per-method summaries.")
    p.add_argument("--methods", type=str, required=True, help="Comma-separated method names.")
    p.add_argument("--tables_dir", type=str, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tables_dir = Path(args.tables_dir)
    methods = [x.strip() for x in str(args.methods).split(",") if x.strip()]
    strat_map = {
        "random_K": "sweep_v3_randomK.csv",
        "bestK_by_clean_consistency": "sweep_v3_bestK_consistency.csv",
        "bestK_by_ref_similarity": "sweep_v3_bestK_targeted.csv",
    }
    out_rows: Dict[str, List[dict]] = {k: [] for k in strat_map.keys()}

    for m in methods:
        path = tables_dir / f"session_attack_summary_abs_{m}.csv"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            st = str(r.get("strategy", ""))
            if st not in out_rows:
                continue
            out_rows[st].append(
                {
                    "method": r.get("method", m),
                    "K16_mean": r.get("K16_mean", ""),
                    "AUC": r.get("AUC", ""),
                    "slope_16_1": r.get("slope_16_1", ""),
                    "K1_mean": r.get("K1_mean", ""),
                    "count": r.get("count", ""),
                }
            )

    fields = ["method", "K16_mean", "AUC", "slope_16_1", "K1_mean", "count"]
    for st, name in strat_map.items():
        out_path = tables_dir / name
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for rr in out_rows[st]:
                w.writerow(rr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
