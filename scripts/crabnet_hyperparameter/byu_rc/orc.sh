#!/bin/bash
# Login-node entry point for the v2 rerun. .github/workflows/byu-rc.yml calls it over
# SSH; it also works by hand or from a login-node crontab.
#   bash orc.sh setup                  # conda environment, v1 CSV, Matbench data (once)
#   bash orc.sh manifest decision      # pick the runs and split them into array tasks
#   bash orc.sh smoke                  # smoke manifest plus one task on the test QOS
#   bash orc.sh submit decision        # submit every task with runs to do that is not queued
#   bash orc.sh status decision        # queue, GPU types and collect
# submit is safe to repeat and is how preempted tasks get rerun: a task that was
# cancelled rather than requeued has runs to do and is no longer queued, so it goes
# back in. Settings come from the environment: QOS (default standby), GPUS (default
# l40s:1; see "Pick a GPU type" in README.md), THROTTLE (max tasks running at once),
# TIME (per task, default 12:00:00), plus REPO, RERUN_DIR and ENV_NAME as in setup_env.sh.

set -eo pipefail

ACTION="$1"
DESIGN="${2:-decision}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO="${REPO:-$HOME/matsci-opt-benchmarks}"
export RERUN_DIR="${RERUN_DIR:-$HOME/crabnet_rerun}"
export ENV_NAME="${ENV_NAME:-crabnet-v2}"
QOS="${QOS:-standby}"
GPUS="${GPUS:-l40s:1}"
TIME="${TIME:-12:00:00}"

if [ "$ACTION" = setup ]; then
    if [ -d "$HOME/.conda/envs/$ENV_NAME" ]; then
        echo "environment $ENV_NAME exists; remove it to rebuild: conda env remove -n $ENV_NAME"
    else
        bash "$HERE/setup_env.sh"
    fi
    exit
fi

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
mkdir -p "$RERUN_DIR/logs"
cd "$RERUN_DIR"  # sbatch writes logs/ relative to here
rerun() { python "$HERE/rerun.py" "$1" --design "$DESIGN" "${@:2}"; }
JOB="crabnet-v2-$DESIGN"

submit() {  # submit the tasks in $1 with any extra sbatch flags
    sbatch --job-name="$JOB" --array="$1" --qos="$QOS" --gpus="$GPUS" --time="$TIME" \
        --export=ALL,DESIGN="$DESIGN" "${@:2}" "$HERE/rerun.sbatch"
}

case "$ACTION" in
manifest)
    rerun manifest
    ;;
smoke)
    DESIGN=smoke JOB=crabnet-v2-smoke
    [ -f smoke/manifest.csv ] || rerun manifest
    QOS="${QOS_SMOKE:-test}" TIME=00:30:00 submit 0
    ;;
submit)
    if [ ! -f "$DESIGN/manifest.csv" ]; then
        echo "no manifest for $DESIGN yet; run: bash orc.sh manifest $DESIGN"
        exit 0
    fi
    # array task ids of this design that are pending, running or requeued
    queued="$(squeue --me -h -r -n "$JOB" -o %K | grep -E '^[0-9]+$' | paste -sd, -)"
    array="$(rerun todo --exclude "$queued")"
    if [ -z "$array" ]; then
        echo "$DESIGN: nothing to submit (queued: ${queued:-none})"
    else
        echo "$DESIGN: submitting tasks $array"
        submit "$array${THROTTLE:+%$THROTTLE}"
    fi
    ;;
status)
    squeue --me -o "%.18i %.20j %.8T %.10M %.12l %.10q %R"
    echo "GPU types (for GPUS=<type>:1):"
    sinfo -h -o "%G" | tr ',' '\n' | grep -o 'gpu:[a-z0-9_]*' | sort -u | sed 's/^gpu:/  /'
    echo "orcquota:"; orcquota || true
    [ -f "$DESIGN/manifest.csv" ] && rerun collect || echo "no results for $DESIGN yet"
    ;;
*)
    echo "usage: bash orc.sh {setup|manifest|smoke|submit|status} [design]" >&2
    exit 2
    ;;
esac
