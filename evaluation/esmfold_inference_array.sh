#!/bin/bash

#SBATCH --mail-user=ria_vinod@brown.edu
#SBATCH --mail-type=ALL

#SBATCH --output=/users/rvinod/data/rvinod/repos/plm_subnetworks/z_slurm/output-%A_%a.out
#SBATCH --error=/users/rvinod/data/rvinod/repos/plm_subnetworks/z_slurm/output-%A_%a.err

#SBATCH -A cbc-a5000-gcondo
#SBATCH --nodes=1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=50G
#SBATCH -J EVAL-esmfold-array
#SBATCH --array=0-0%12

set -euo pipefail

DEFAULT_CSV="/users/rvinod/data/rvinod/repos/plm_subnetworks/results_figures/esm-all-final.csv"

usage() {
    echo "Usage: $0 [csv_path]" >&2
}

data_error() {
    echo "[WARN] $1" >&2
}

if [[ $# -gt 1 ]]; then
    usage
    exit 1
fi

CSV_PATH="${1:-$DEFAULT_CSV}"
if [[ ! -f "$CSV_PATH" ]]; then
    echo "Error: CSV file not found at $CSV_PATH" >&2
    exit 1
fi

CSV_ABS=$(realpath "$CSV_PATH")

echo "Using run list: $CSV_ABS"

MAX_PARALLEL=${MAX_PARALLEL:-8}
if [[ -z "$MAX_PARALLEL" ]]; then
    MAX_PARALLEL=8
fi

if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]]; then
    echo "MAX_PARALLEL must be an integer; got '$MAX_PARALLEL'" >&2
    exit 1
fi

if (( MAX_PARALLEL < 1 )); then
    MAX_PARALLEL=1
fi

echo "Building task list from CSV"
mapfile -t TASKS < <(
python3 - <<'PY' "$CSV_ABS"
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
with csv_path.open(newline='') as fh:
    reader = csv.DictReader(fh)
    if reader.fieldnames is None:
        sys.exit(0)
    for row in reader:
        lowered = {}
        for key, value in row.items():
            if key is None:
                continue
            lowered[key.strip().lower()] = (value.strip() if isinstance(value, str) else value)

        status = lowered.get('done') or lowered.get('status')
        if status and str(status).strip().lower() not in {'', 'todo', 'pending', '0'}:
            continue

        run_name = lowered.get('run_name') or lowered.get('run dir') or lowered.get('run_dir')
        epoch = lowered.get('epoch')
        category = lowered.get('category')
        target = lowered.get('target')

        if not run_name or not category or not target:
            continue
        if epoch in (None, ''):
            continue

        print(f"{run_name}|{epoch}|{category}|{target}")
PY
)

TASK_COUNT=${#TASKS[@]}
if (( TASK_COUNT == 0 )); then
    echo "No runnable tasks discovered in $CSV_ABS" >&2
    exit 1
fi

echo "Discovered $TASK_COUNT runnable tasks"

if (( TASK_COUNT < MAX_PARALLEL )); then
    EFFECTIVE_PARALLEL=$TASK_COUNT
else
    EFFECTIVE_PARALLEL=$MAX_PARALLEL
fi

if (( EFFECTIVE_PARALLEL < 1 )); then
    EFFECTIVE_PARALLEL=1
fi

ARRAY_SPEC="0-$((TASK_COUNT-1))"
if (( TASK_COUNT > 1 )); then
    ARRAY_SPEC+="%$EFFECTIVE_PARALLEL"
fi

need_resubmit=false
if (( TASK_COUNT > 1 )) && [[ -z "${FORCE_SERIAL:-}" ]]; then
    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        need_resubmit=true
    elif [[ "${SLURM_ARRAY_TASK_COUNT:-1}" -le 1 ]]; then
        need_resubmit=true
    fi
fi

if [[ $need_resubmit == true ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "Current submission only has a single array task; re-submitting with --array=$ARRAY_SPEC" >&2
    else
        echo "Submitting array job with --array=$ARRAY_SPEC (max parallel=$EFFECTIVE_PARALLEL)" >&2
    fi
    sbatch --export=ALL,MAX_PARALLEL=$MAX_PARALLEL --array="$ARRAY_SPEC" "$0" "$CSV_ABS"
    status=$?
    if (( status != 0 )); then
        echo "Failed to submit array job" >&2
    fi
    exit $status
fi

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate esmfold_new

REPO_ROOT="/users/rvinod/data/rvinod/repos/plm_subnetworks"
cd "$REPO_ROOT/evaluation"

# Ensure Python can import the local plm_subnetworks package
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

PYTHON_BIN="$(conda run -n esmfold_new which python)"

echo "Python: $PYTHON_BIN"

run_task() {
    local idx=$1
    IFS='|' read -r RUN_NAME EPOCH CATEGORY TARGET <<< "${TASKS[$idx]}"

    if [[ -z "$RUN_NAME" || -z "$CATEGORY" || -z "$TARGET" ]]; then
        data_error "Task $idx missing required fields; skipping."
        return
    fi

    local epoch_arg="$EPOCH"
    if [[ -z "$epoch_arg" ]]; then
        data_error "Task $idx has empty epoch; skipping."
        return
    fi

    echo "[$(date)] Task $idx: Folding $RUN_NAME (epoch $epoch_arg, category $CATEGORY, target $TARGET)"
    $PYTHON_BIN fold_sequences.py \
        --run_name "$RUN_NAME" \
        --epoch "$epoch_arg" \
        --category "$CATEGORY"

    local epoch_int
    epoch_int=$(echo "$epoch_arg" | sed 's/^0*//')
    if [[ -z "$epoch_int" ]]; then
        epoch_int=0
    fi

    echo "[$(date)] Task $idx: TM-score evaluation for $RUN_NAME"
    $PYTHON_BIN tm_scores.py \
        --run_name "$RUN_NAME" \
        --mode pred \
        --epoch "$epoch_int" \
        --category "$CATEGORY" \
        --target "$TARGET" \
        --subnetwork_eval
}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "No SLURM_ARRAY_TASK_ID detected; running all $TASK_COUNT tasks sequentially."
    for idx in "${!TASKS[@]}"; do
        run_task "$idx"
    done
    echo "Done."
    exit 0
fi

if (( SLURM_ARRAY_TASK_ID >= TASK_COUNT )); then
    echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID out of range for $TASK_COUNT tasks; nothing to do." >&2
    exit 0
fi

echo "Processing array index $SLURM_ARRAY_TASK_ID"
run_task "$SLURM_ARRAY_TASK_ID"

echo "Done."
