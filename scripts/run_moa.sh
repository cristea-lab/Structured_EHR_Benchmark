
#!/usr/bin/env bash
set -eo pipefail

TASK=${1:-acute_mi}
ANCHOR=${2:-earliest}
TAG=${3:-seed2}

# A) Summaries
python -m src.pipelines.moa.summarize_qwen   --sqlite_path ./data/omop_db.sqlite3   --assets_root ./data/assets   --task "${TASK}"   --visit_anchor "${ANCHOR}"   --run_tag "${TAG}"

# B) Prepare splits
python -m src.pipelines.moa.prepare_splits   --assets_root ./data/assets   --intermediate_root ./outputs/intermediate   --task "${TASK}"   --run_tag "${TAG}"

# C) Train & evaluate ClinicalBERT
python -m src.models.clinicalbert_predictor   --train_csv outputs/intermediate/${TASK}_${TAG}_train.csv   --val_csv   outputs/intermediate/${TASK}_${TAG}_val.csv   --test_csv  outputs/intermediate/${TASK}_${TAG}_test.csv   --out_dir   outputs/models/moa_clinicalbert_${TASK}_${ANCHOR}
