#!/usr/bin/env python3
"""使用 torchaudio 下载 VCTK 到指定 raw 目录（与 download_data.sh 配合，自动下载时调用）。"""
import argparse
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", type=str, default=None, help="data/raw 目录")
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    raw_dir = Path(args.raw_dir) if args.raw_dir else repo / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        from torchaudio.datasets import VCTK_092
        VCTK_092(root=str(raw_dir), download=True)
        print(f"[download_vctk.py] VCTK 已下载到 {raw_dir}/VCTK-Corpus-0.92")
        return 0
    except Exception as e:
        print(f"[download_vctk.py] torchaudio 下载失败: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
