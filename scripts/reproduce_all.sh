
#!/usr/bin/env bash
set -eo pipefail

TASK=${1:-acute_mi}
ANCHOR=${2:-earliest}
TAG=${3:-seed2}

bash scripts/run_count.sh "${TASK}" "${ANCHOR}"
bash scripts/run_clmbr.sh "${TASK}" "${ANCHOR}"
bash scripts/run_moa.sh   "${TASK}" "${ANCHOR}" "${TAG}"
