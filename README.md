# CloneBlockRT

A real-time **voice cloning / blocking** research project.

## Environment

- Conda environment: `CloneClock`
- Environment snapshot: `artifacts/env/`

## Datasets (isolated, no overwrite)

This project supports **LibriSpeech dev-clean** and **VCTK**. All data are stored under a dataset-specific folder (`--dataset`) so the two datasets never overwrite each other.

| Dataset     | Raw data path                              | Processed audio path                          | Split CSVs path                         |
|------------|---------------------------------------------|-----------------------------------------------|-----------------------------------------|
| LibriSpeech | `data/raw/LibriSpeech/dev-clean`            | `data/processed/librispeech/wav16k/`          | `data/splits/librispeech/*.csv`         |
| VCTK        | `data/raw/VCTK-Corpus/` or `VCTK-Corpus-0.92/` | `data/processed/vctk/wav16k/`                 | `data/splits/vctk/*.csv`                |

### Download

```bash
# LibriSpeech only
./scripts/download_data.sh librispeech

# VCTK only (try wget first; fallback to torchaudio downloader if needed)
./scripts/download_data.sh vctk

# Download both
./scripts/download_data.sh all

# Quick Start
conda activate CloneClock
./scripts/download_data.sh all
python scripts/prepare_data.py --dataset librispeech --seed 42
python scripts/prepare_data.py --dataset vctk --seed 42
python scripts/run_train.py --dataset librispeech
