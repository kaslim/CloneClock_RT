#!/usr/bin/env bash
# One-shot reproduction of paper artifacts (no training). Fixed seed=42.
# Run from repo root; conda env CloneClock recommended.
set -e
SEED=42
DATASET=vctk
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
TABLES="$REPO_ROOT/artifacts/tables"
POOL_CSV="$REPO_ROOT/data/splits/session_pool_v3_test.csv"
CKPT="$REPO_ROOT/checkpoints/defense/v0p1_B_best.pt"
SPK_ECAPA="$REPO_ROOT/checkpoints/speaker_encoders/speechbrain_ecapa"
SPK_XV="$REPO_ROOT/checkpoints/speaker_encoders/speechbrain_xvector"

echo "[1/8] Session pool v3 (skip if exists)"
if [ ! -f "$POOL_CSV" ]; then
  python scripts/build_session_pool_v3.py --dataset "$DATASET" --seed "$SEED"
fi
echo "  -> $POOL_CSV"

echo "[2/8] Main ECAPA targeted tele_defended (baseline + v0p1_B)"
python scripts/run_session_pool_eval.py --dataset "$DATASET" --session_pool_csv "$POOL_CSV" --method_name baseline --group_key session_id --strategies bestK_by_ref_similarity --targeted_selection_source tele_defended --speaker_encoder_name speechbrain_ecapa --speaker_checkpoint_dir "$SPK_ECAPA" --output_tag v3_teledef_ecapa --seed "$SEED"
python scripts/run_session_pool_eval.py --dataset "$DATASET" --session_pool_csv "$POOL_CSV" --method_name v0p1_B --defense_checkpoint "$CKPT" --group_key session_id --strategies bestK_by_ref_similarity --targeted_selection_source tele_defended --speaker_encoder_name speechbrain_ecapa --speaker_checkpoint_dir "$SPK_ECAPA" --output_tag v3_teledef_ecapa --seed "$SEED"
python - <<PY
import csv
from pathlib import Path
root = Path("$TABLES")
for m in ["baseline", "v0p1_B"]:
    p = root / f"session_attack_summary_abs_{m}_v3_teledef_ecapa.csv"
    if not p.exists():
        raise SystemExit(1)
rows = []
for m in ["baseline", "v0p1_B"]:
    with open(root / f"session_attack_summary_abs_{m}_v3_teledef_ecapa.csv") as f:
        r = next(x for x in csv.DictReader(f) if x.get("strategy") == "bestK_by_ref_similarity")
    rows.append({"method": r["method"], "K16_mean": r["K16_mean"], "AUC": r["AUC"], "slope_16_1": r["slope_16_1"]})
with open(root / "ens_ecapa_sweep_v3_bestK_targeted_teledef.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "K16_mean", "AUC", "slope_16_1"])
    w.writeheader()
    w.writerows(rows)
PY
echo "  -> $TABLES/ens_ecapa_sweep_v3_bestK_targeted_teledef.csv"

echo "[3/8] Codec Opus + G.711"
python scripts/run_session_pool_eval.py --dataset "$DATASET" --session_pool_csv "$POOL_CSV" --method_name baseline --group_key session_id --strategies bestK_by_ref_similarity --targeted_selection_source tele_defended --speaker_encoder_name speechbrain_ecapa --speaker_checkpoint_dir "$SPK_ECAPA" --telephony_codec opus --telephony_codec_bitrate 16k --output_tag v3_teledef_ecapa_codec_opus16k --seed "$SEED"
python scripts/run_session_pool_eval.py --dataset "$DATASET" --session_pool_csv "$POOL_CSV" --method_name v0p1_B --defense_checkpoint "$CKPT" --group_key session_id --strategies bestK_by_ref_similarity --targeted_selection_source tele_defended --speaker_encoder_name speechbrain_ecapa --speaker_checkpoint_dir "$SPK_ECAPA" --telephony_codec opus --telephony_codec_bitrate 16k --output_tag v3_teledef_ecapa_codec_opus16k --seed "$SEED"
python scripts/run_session_pool_eval.py --dataset "$DATASET" --session_pool_csv "$POOL_CSV" --method_name baseline --group_key session_id --strategies bestK_by_ref_similarity --targeted_selection_source tele_defended --speaker_encoder_name speechbrain_ecapa --speaker_checkpoint_dir "$SPK_ECAPA" --telephony_codec g711 --output_tag v3_teledef_ecapa_codec_g711 --seed "$SEED"
python scripts/run_session_pool_eval.py --dataset "$DATASET" --session_pool_csv "$POOL_CSV" --method_name v0p1_B --defense_checkpoint "$CKPT" --group_key session_id --strategies bestK_by_ref_similarity --targeted_selection_source tele_defended --speaker_encoder_name speechbrain_ecapa --speaker_checkpoint_dir "$SPK_ECAPA" --telephony_codec g711 --output_tag v3_teledef_ecapa_codec_g711 --seed "$SEED"
python - <<PY
import csv
from pathlib import Path
root = Path("$TABLES")
def build_codec(tag, codec, bitrate, out_name):
    rows = []
    for m in ["baseline", "v0p1_B"]:
        p = root / f"session_attack_summary_abs_{m}_v3_teledef_ecapa_codec_{tag}.csv"
        if not p.exists():
            return
        with open(p) as f:
            r = next(csv.DictReader(f))
        rows.append({"method": r["method"], "K16_mean": r["K16_mean"], "AUC": r["AUC"], "slope_16_1": r["slope_16_1"], "codec": codec, "codec_bitrate": bitrate, "selection_source": "tele_defended", "encoder": "speechbrain_ecapa"})
    with open(root / out_name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "K16_mean", "AUC", "slope_16_1", "codec", "codec_bitrate", "selection_source", "encoder"])
        w.writeheader()
        w.writerows(rows)
build_codec("opus16k", "opus", "16k", "codec_opus_sweep_v3_bestK_targeted_teledef.csv")
build_codec("g711", "g711", "n/a", "codec_g711_sweep_v3_bestK_targeted_teledef.csv")
PY
echo "  -> $TABLES/codec_opus_sweep_v3_bestK_targeted_teledef.csv, codec_g711_sweep_v3_bestK_targeted_teledef.csv"

echo "[4/8] Cross-encoder ASV EER (ECAPA + xvector)"
python scripts/run_asv_eer_eval.py --dataset "$DATASET" --session_pool_csv "$POOL_CSV" --method_specs "baseline:,v0p1_B:$CKPT" --speaker_encoder_name speechbrain_ecapa --speaker_checkpoint_dir "$SPK_ECAPA" --selection_source tele_defended --k 16 --impostors_per_target 10 --seed "$SEED" --output_csv "$TABLES/asv_eer_ecapa_v3_targeted_teledef.csv" --output_margin_csv "$TABLES/asv_margin_ecapa_v3_targeted_teledef.csv"
python scripts/run_asv_eer_eval.py --dataset "$DATASET" --session_pool_csv "$POOL_CSV" --method_specs "baseline:,v0p1_B:$CKPT" --speaker_encoder_name speechbrain_xvector --speaker_checkpoint_dir_xvector "$SPK_XV" --selection_source tele_defended --k 16 --impostors_per_target 10 --seed "$SEED" --output_csv "$TABLES/asv_eer_xvector_v3_targeted_teledef.csv" --output_margin_csv "$TABLES/asv_margin_xvector_v3_targeted_teledef.csv"
echo "  -> $TABLES/asv_eer_ecapa_v3_targeted_teledef.csv, asv_eer_xvector_v3_targeted_teledef.csv"

echo "[5/8] Latency v2 benchmark"
python scripts/run_stream_benchmark.py --dataset "$DATASET" --session_pool_csv "$POOL_CSV" --defense_checkpoint "$CKPT" --chunk_ms_list 20,40
echo "  -> $TABLES/latency_v0p1_B_v2.csv"

echo "[6/8] Quality eval (WER + STOI, none / opus16k / g711)"
python scripts/run_quality_eval.py --dataset vctk --split test --telephony_codec none --max_utts 200 --seed "$SEED" --defense_checkpoint "$CKPT"
python scripts/run_quality_eval.py --dataset vctk --split test --telephony_codec opus --telephony_codec_bitrate 16k --max_utts 200 --seed "$SEED" --defense_checkpoint "$CKPT"
python scripts/run_quality_eval.py --dataset vctk --split test --telephony_codec g711 --max_utts 200 --seed "$SEED" --defense_checkpoint "$CKPT"
echo "  -> $TABLES/quality/quality_eval_*.csv, $TABLES/paper/quality_summary.csv"

echo "[7/8] Export paper tables and figures"
python scripts/export_paper_artifacts.py
echo "  -> $TABLES/paper/*.csv, $REPO_ROOT/artifacts/figures/paper/*.png"

echo "[8/8] LaTeX tables"
python scripts/csv_to_latex.py --tables_dir "$TABLES/paper" --ndec 4
echo "  -> $TABLES/paper/*.tex"

echo "Done. Paper artifacts in artifacts/tables/paper/ and artifacts/figures/paper/"
