
#!/usr/bin/env bash
set -eo pipefail

TASK=${1:-acute_mi}
ANCHOR=${2:-earliest}

python -m src.pipelines.clmbr.run_clmbr   --sqlite_path ./data/omop_db.sqlite3   --assets_root ./data/assets   --task "${TASK}"   --visit_anchor "${ANCHOR}"   --out_dir ./outputs/clmbr/${TASK}_${ANCHOR}
