
#!/usr/bin/env bash
set -eo pipefail

TASK=${1:-acute_mi}          # los|readmission|pancreatic_cancer|acute_mi
ANCHOR=${2:-earliest}        # earliest|latest

python -m src.pipelines.count.run_count   --sqlite_path ./data/omop_db.sqlite3   --assets_root ./data/assets   --task "${TASK}"   --visit_anchor "${ANCHOR}"   --models both   --out_dir ./outputs/count/${TASK}_${ANCHOR}
