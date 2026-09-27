#!/bin/bash
# Login-node entry point for the v2 rerun. A @claude-orc session calls it over SSH;
# it also works by hand or from a login-node crontab.
#   bash orc.sh setup                  # conda environment, v1 CSV, Matbench data (once)
#   bash orc.sh manifest decision      # pick the runs and split them into array tasks
#   bash orc.sh manifest dummy --sobol 100 --simplex 100   # extra options go to rerun.py
#   bash orc.sh smoke                  # smoke manifest plus one task on the test QOS
#   bash orc.sh submit decision        # submit every task with runs to do that is not queued
#   bash orc.sh status decision        # queue, GPU types and collect
# submit is safe to repeat and is how preempted tasks get rerun: a task that was
# cancelled rather than requeued has runs to do and is no longer queued, so it goes
# back in. Settings come from the environment: QOS (default standby), GPUS (default
# l40s:1; see "Pick a GPU type" in README.md), THROTTLE (max tasks running at once),
# TIME (per task, default 12:00:00), MAX_QUEUED (max tasks pending or running, default
# 1000), plus REPO, RERUN_DIR and ENV_NAME as in setup_env.sh.

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
MAX_QUEUED="${MAX_QUEUED:-1000}"

if [ "$ACTION" = setup ]; then
    if [ -d "$HOME/.conda/envs/$ENV_NAME" ]; then
        echo "environment $ENV_NAME exists; remove it to rebuild: conda env remove -n $ENV_NAME"
    else
        bash "$HERE/setup_env.sh"
    fi
    exit
fi

mkdir -p "$RERUN_DIR/logs"
cd "$RERUN_DIR"  # sbatch writes logs/ relative to here
rerun() {  # the conda environment is only needed here, so status works before setup
    if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
        module load miniforge3
        eval "$(conda shell.bash hook)"
        conda activate "$ENV_NAME"
    fi
    python "$HERE/rerun.py" "$1" --design "$DESIGN" "${@:2}"
}
JOB="crabnet-v2-$DESIGN"

submit() {  # submit array tasks $1 (task ids minus TASK_OFFSET) with any extra sbatch flags
    local offset="${TASK_OFFSET:-0}"
    sbatch --job-name="$JOB-$offset" --array="$1" --qos="$QOS" --gpus="$GPUS" --time="$TIME" \
        --export=ALL,DESIGN="$DESIGN",TASK_OFFSET="$offset" "${@:2}" "$HERE/rerun.sbatch"
}

case "$ACTION" in
manifest)
    rerun manifest "${@:3}"
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
    # task ids of this design that are pending, running or requeued: the array task id
    # plus the offset that ends the job name (awk rather than grep, which exits 1 when
    # nothing is queued and so stops the script)
    queued="$(squeue --me -h -r -o "%j %K" | awk -v p="$JOB-" \
        'index($1, p) == 1 && $2 ~ /^[0-9]+$/ {print substr($1, length(p) + 1) + $2}' |
        paste -sd, -)"
    # at most MAX_QUEUED tasks pending or running (the account allows 5,000 jobs in
    # the queue across its users), in one array per 5,000 tasks (the array task id
    # limit), each with its own offset
    n_queued="$(awk -F, '{print NF}' <<< "$queued")"
    room=$((MAX_QUEUED - n_queued))
    todo="$(rerun todo --exclude "$queued" --limit $((room > 0 ? room : 0)))"
    [ -n "$todo" ] || echo "$DESIGN: nothing to submit ($n_queued tasks queued, MAX_QUEUED=$MAX_QUEUED)"
    while read -r offset array; do
        [ -n "$array" ] || continue
        echo "$DESIGN: submitting tasks $offset + ($array)"
        TASK_OFFSET="$offset" submit "$array${THROTTLE:+%$THROTTLE}"
    done <<< "$todo"
    ;;
status)
    squeue --me -o "%.18i %.20j %.8T %.10M %.12l %.10q %R"
    echo "GPU types (for GPUS=<type>:1):"
    sinfo -h -o "%G" | tr ',' '\n' | grep -oE 'gpu:[a-z][a-z0-9_]*' | sort -u | sed 's/^gpu:/  /'
    echo "orcquota:"; orcquota || true
    [ -f "$DESIGN/manifest.csv" ] && rerun collect || echo "no results for $DESIGN yet"
    ;;
*)
    echo "usage: bash orc.sh {setup|manifest|smoke|submit|status} [design]" >&2
    exit 2
    ;;
esac
