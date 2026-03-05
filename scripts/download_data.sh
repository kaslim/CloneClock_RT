#!/usr/bin/env bash
# download_data.sh — 下载 LibriSpeech dev-clean 和/或 VCTK
# 用法: ./download_data.sh [librispeech|vctk|all]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="${RAW_DIR:-$REPO_ROOT/data/raw}"
mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

# LibriSpeech 使用官方 OpenSLR（更稳定）
OPENSLR_BASE="${OPENSLR_BASE:-https://www.openslr.org/resources/12}"

download_librispeech() {
  echo "[download_data.sh] 下载 LibriSpeech dev-clean (OpenSLR 12) ..."
  if [[ -d "$RAW_DIR/LibriSpeech/dev-clean" ]]; then
    echo "[download_data.sh] 已存在 LibriSpeech/dev-clean，跳过。"
    return 0
  fi
  if [[ ! -f dev-clean.tar.gz ]]; then
    wget -O dev-clean.tar.gz "$OPENSLR_BASE/dev-clean.tar.gz" || {
      echo "主站失败，尝试 EU 镜像 ..."
      wget -O dev-clean.tar.gz "https://openslr.elda.org/resources/12/dev-clean.tar.gz"
    }
  fi
  echo "[download_data.sh] 解压 dev-clean.tar.gz ..."
  tar -xzf dev-clean.tar.gz
  echo "[download_data.sh] LibriSpeech dev-clean 就绪于: $RAW_DIR/LibriSpeech/dev-clean"
}

# VCTK: 尝试直接下载（Edinburgh 可能需浏览器），失败则输出手动说明
VCTK_URL="https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"

download_vctk() {
  echo "[download_data.sh] 尝试下载 VCTK ..."
  if [[ -d "$RAW_DIR/VCTK-Corpus/wav48" ]] || [[ -d "$RAW_DIR/VCTK-Corpus-0.92/wav48_silence_trimmed" ]] || [[ -d "$RAW_DIR/VCTK-Corpus-0.92/wav48" ]]; then
    echo "[download_data.sh] 已存在 VCTK 数据，跳过。"
    return 0
  fi
  local zipname="VCTK-Corpus-0.92.zip"
  if [[ -f "$zipname" ]]; then
    echo "[download_data.sh] 已存在 $zipname，解压中 ..."
  else
    wget -q --no-check-certificate -U "Mozilla/5.0 (compatible; DatasetDownload/1.0)" \
      -O "$zipname" "$VCTK_URL" 2>/dev/null || true
    if [[ ! -f "$zipname" || ! -s "$zipname" ]]; then
      rm -f "$zipname" 2>/dev/null
      echo "[download_data.sh] wget 失败，尝试 torchaudio 下载 ..."
      if python "$REPO_ROOT/scripts/download_vctk.py" --raw_dir "$RAW_DIR"; then
        return 0
      fi
      vctk_manual_instructions
      return 1
    fi
  fi
  if command -v unzip &>/dev/null; then
    unzip -o -q "$zipname"
    if [[ -d "VCTK-Corpus/wav48" ]]; then
      echo "[download_data.sh] VCTK 就绪于: $RAW_DIR/VCTK-Corpus/wav48"
    elif [[ -d "VCTK-Corpus-0.92/wav48_silence_trimmed" || -d "VCTK-Corpus-0.92/wav48" ]]; then
      echo "[download_data.sh] VCTK 就绪于: $RAW_DIR/VCTK-Corpus-0.92"
    else
      echo "[download_data.sh] 解压后请确认存在 wav48 或 VCTK-Corpus-0.92。"
    fi
  else
    echo "[download_data.sh] 需要 unzip，请安装后重试或手动解压 $zipname。"
    vctk_manual_instructions
    return 1
  fi
}

vctk_manual_instructions() {
  echo ""
  echo "========== VCTK 手动下载说明 =========="
  echo "若自动下载失败，请："
  echo "1. 打开: https://datashare.ed.ac.uk/handle/10283/3443"
  echo "2. 同意条款后点击 “Download all” 获取 VCTK-Corpus-0.92.zip"
  echo "3. 将 zip 放到: $RAW_DIR"
  echo "4. 解压并确保存在: $RAW_DIR/VCTK-Corpus/wav48/<speaker_id>/*.wav"
  echo "5. 运行: python $REPO_ROOT/scripts/prepare_data.py --dataset vctk"
  echo "========================================"
}

case "${1:-librispeech}" in
  librispeech)
    download_librispeech
    ;;
  vctk)
    download_vctk || true
    ;;
  all)
    download_librispeech
    download_vctk || vctk_manual_instructions
    ;;
  *)
    echo "用法: $0 [librispeech|vctk|all]"
    echo "  librispeech — 自动下载 LibriSpeech dev-clean"
    echo "  vctk       — 尝试下载 VCTK，失败则显示手动说明"
    echo "  all        — 先 LibriSpeech 再 VCTK"
    exit 1
    ;;
esac
