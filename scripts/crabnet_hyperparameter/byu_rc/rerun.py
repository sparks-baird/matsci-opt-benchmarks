"""Rerun v1 CrabNet hyperparameter sets with a new network per Matbench fold (v2).

The v1 dataset (Zenodo 10.5281/zenodo.7694268) fit one CrabNet instance on the five
matbench_expt_gap folds in turn. CrabNet.fit() only builds a network when none exists,
so folds 1 to 4 were scored on compositions the network had already trained on.
submitit_evaluate now builds a new network for every fold and also returns each fold's
scores. This script reruns rows of the v1 CSV with it, one Slurm array task at a time
(rerun.sbatch), on BYU Research Computing. See README.md for the full workflow.

    python rerun.py manifest --design decision   # once, on a login node
    sbatch --array=0-<last task> rerun.sbatch    # the manifest step prints the range
    python rerun.py collect --design decision    # any time; merges and compares with v1

Designs, using v1 rows with train_frac >= 0.01 (the 16 rows from two test sessions,
with train_frac = 0.003, are left out):

- smoke: the 2 hyperparameter sets with the shortest v1 runtime, for a first test job
- decision: 1,000 random sets plus the 200 with the lowest repeat-averaged v1 MAE, one
  run each (about 3 GPU-days at 2080 Ti speed)
- one-per-set: one run for each of the 41,543 sets (93 GPU-days)
- full: one run per v1 run (173,203 runs), so v2 keeps the v1 repeat structure
  (387 GPU-days)

Runs are assigned to array tasks, longest first, so that each task holds about --hours
of v1 (RTX 2080 Ti) runtime. Each finished run is appended to results/task_<id>.jsonl,
and a task skips runs already there, so a preempted, requeued or resubmitted task
continues where it stopped. Failed runs are recorded with their error message and are
not retried. collect lists the tasks that still have runs to do.

Files live in RERUN_DIR (default ~/crabnet_rerun), which must hold sobol_regression.csv
from the Zenodo record (setup_env.sh downloads it). Each design gets its own
subdirectory with manifest.csv, results/ and results_v2.csv.
"""

import argparse
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("mode", choices=["manifest", "run", "collect"])
parser.add_argument(
    "--design", default="decision", choices=["smoke", "decision", "one-per-set", "full"]
)
parser.add_argument("--hours", type=float, default=4.0, help="v1 runtime per task")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

rerun_dir = Path(os.environ.get("RERUN_DIR", Path.home() / "crabnet_rerun"))
manifest_path = rerun_dir / args.design / "manifest.csv"
results_dir = rerun_dir / args.design / "results"
hp = [
    "N", "alpha", "d_model", "dim_feedforward", "dropout", "emb_scaler", "eps",
    "epochs_step", "fudge", "heads", "k", "lr", "pe_resolution", "ple_resolution",
    "pos_scaler", "weight_decay", "batch_size", "out_hidden4", "betas1", "betas2",
    "bias", "criterion", "elem_prop", "train_frac",
]  # fmt: skip

if args.mode == "manifest":
    assert not any(results_dir.glob("*.jsonl")), f"{results_dir} has results; move them first"
    v1 = pd.read_csv(rerun_dir / "sobol_regression.csv")
    v1 = v1[v1["train_frac"] >= 0.01].reset_index(names="v1_row")
    v1["set_id"] = v1.groupby(hp, sort=False).ngroup()
    sets = v1.groupby("set_id").agg(
        v1_mae=("mae", "mean"),
        v1_rmse=("rmse", "mean"),
        v1_runtime=("runtime", "mean"),
        v1_model_size=("model_size", "first"),
        v1_repeats=("mae", "size"),
    )
    rng = np.random.default_rng(args.seed)
    if args.design == "full":
        runs = v1
    else:
        if args.design == "smoke":
            ids = sets["v1_runtime"].nsmallest(2).index
        elif args.design == "decision":
            ids = np.union1d(
                rng.choice(sets.index, 1000, replace=False),
                sets["v1_mae"].nsmallest(200).index,
            )
        else:
            ids = sets.index
        runs = v1.drop_duplicates("set_id")
        runs = runs[runs["set_id"].isin(ids)]
    runs = runs[["v1_row", "set_id", *hp]].merge(sets, left_on="set_id", right_index=True)
    runs = runs.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    runs.insert(0, "run_id", np.arange(len(runs)))
    runs["sample_seed"] = rng.integers(0, 1000, len(runs))  # same range as v1

    # longest first, each to the least loaded task
    seconds = runs["v1_runtime"].to_numpy()
    load = np.zeros(max(1, int(np.ceil(seconds.sum() / 3600 / args.hours))))
    task = np.empty(len(runs), dtype=int)
    for i in np.argsort(-seconds):
        task[i] = load.argmin()
        load[task[i]] += seconds[i]
    runs["task"] = task
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(manifest_path, index=False)
    print(
        f"{args.design}: {len(runs)} runs of {runs['set_id'].nunique()} sets, "
        f"{seconds.sum() / 86400:.1f} GPU-days at 2080 Ti speed, {len(load)} tasks "
        f"(v1 hours per task: median {np.median(load) / 3600:.1f}, "
        f"max {load.max() / 3600:.1f})\n"
        f"submit from {rerun_dir} with: sbatch --array=0-{len(load) - 1} "
        f"--export=ALL,DESIGN={args.design} <path to rerun.sbatch>"
    )

if args.mode == "run":
    import torch

    from matsci_opt_benchmarks.crabnet_hyperparameter.utils.parameters import (
        submitit_evaluate,
    )

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    runs = pd.read_csv(manifest_path)
    runs = runs[runs["task"] == task_id]
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"task_{task_id:05d}.jsonl"
    done = [json.loads(line)["run_id"] for line in out.open()] if out.exists() else []
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    print(f"task {task_id}: {len(runs)} runs, {len(done)} already done, on {gpu}")
    for run in runs[~runs["run_id"].isin(done)].to_dict("records"):
        parameters = {k: run[k] for k in hp}
        parameters["bias"] = bool(parameters["bias"])
        parameters["sample_seed"] = int(run["sample_seed"])
        result = submitit_evaluate(parameters)
        record = {
            "run_id": run["run_id"],
            "set_id": run["set_id"],
            "v1_row": run["v1_row"],
            **result,
            "gpu": gpu,
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "finished": datetime.now(timezone.utc).isoformat(),
            "torch": torch.__version__,
        }
        with out.open("a") as f:
            f.write(json.dumps(record, default=float) + "\n")
            f.flush()
            os.fsync(f.fileno())

if args.mode == "collect":
    from scipy.stats import spearmanr

    runs = pd.read_csv(manifest_path)
    records = [
        json.loads(line)
        for path in sorted(results_dir.glob("task_*.jsonl"))
        for line in path.open()
    ]
    if not records:
        raise SystemExit(f"no results in {results_dir} yet")
    res = pd.json_normalize(records).drop_duplicates("run_id", keep="last")
    todo = sorted(runs.loc[~runs["run_id"].isin(res["run_id"]), "task"].unique())
    failed = res["error"].notna() if "error" in res else pd.Series(False, res.index)
    res = res[~failed].rename(
        columns={"scores.mae.mean": "mae", "scores.rmse.mean": "rmse"}
    )
    fold_cols = [c for c in res if c.startswith("fold_scores.")]
    table = runs.merge(
        res[["run_id", "mae", "rmse", "model_size", "runtime", "gpu", *fold_cols]],
        on="run_id",
    )
    table.to_csv(manifest_path.parent / "results_v2.csv", index=False)
    print(
        f"{len(table)} of {len(runs)} runs done, {failed.sum()} failed, "
        f"{len(runs) - len(table) - failed.sum()} to go"
    )
    if todo:
        print(f"tasks with runs to do: --array={','.join(map(str, todo))}")
    if len(table):
        folds = [table[f"fold_scores.fold_{i}.mae"].median() for i in range(5)]
        print("median MAE by fold (no downward trend expected):", np.round(folds, 3))
        per_set = table.groupby("set_id")[["v1_mae", "mae", "v1_rmse", "rmse"]].mean()
        print(f"median v2 / v1 MAE: {(per_set['mae'] / per_set['v1_mae']).median():.3f}")
        print(f"median v2 / v1 RMSE: {(per_set['rmse'] / per_set['v1_rmse']).median():.3f}")
        if len(per_set) > 2:
            rho = spearmanr(per_set["v1_mae"], per_set["mae"])[0]
            print(f"Spearman(v1 MAE, v2 MAE) over {len(per_set)} sets: {rho:.3f}")
            k = max(1, len(per_set) // 20)
            top = len(
                set(per_set["v1_mae"].nsmallest(k).index)
                & set(per_set["mae"].nsmallest(k).index)
            )
            print(f"best {k} sets by v1 MAE that are also best {k} by v2: {top}")
            best_v1 = per_set["v1_mae"].idxmin()
            print(
                f"v1's best set ranks {int(per_set['mae'].rank()[best_v1])} "
                f"of {len(per_set)} in v2"
            )
        ratio = (table["runtime"] / table["v1_runtime"]).groupby(table["gpu"]).median()
        print("median runtime ratio v2 / v1 (2080 Ti) by GPU:", ratio.round(2).to_dict())
