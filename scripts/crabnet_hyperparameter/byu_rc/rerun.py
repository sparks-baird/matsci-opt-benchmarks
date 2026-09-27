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
    python rerun.py todo --design decision       # offset and --array list of tasks to do

Designs, using v1 rows with train_frac >= 0.01 (the 16 rows from two test sessions,
with train_frac = 0.003, are left out):

- smoke: the 2 hyperparameter sets with the shortest v1 runtime, for a first test job
- decision: 1,000 random sets plus the 200 with the lowest repeat-averaged v1 MAE, one
  run each (about 3 GPU-days at 2080 Ti speed)
- one-per-set: one run for each of the 41,543 sets (93 GPU-days)
- full: one run per v1 run (173,203 runs), so v2 keeps the v1 repeat structure
  (387 GPU-days)

One design draws new points instead, the way v1 was generated, as a small end-to-end
test of a fresh dataset:

- dummy: --sobol Sobol points (default 100) over the v1 search space (get_parameters,
  with its two constraints), plus --simplex points (default 100) on the simplex where
  the 20 min-max scaled numeric hyperparameters sum to 1, which no v1 run comes near.
  The simplex block holds every vertex (one hyperparameter at its top, the rest at
  their bottom), points on faces with 2 to 4 active hyperparameters, and interior
  points. Each point runs --repeats times (default 2) with different sample seeds.

Runs are assigned to array tasks, longest first, so that each task holds about --hours
of v1 (RTX 2080 Ti) runtime. Each finished run is appended to results/task_<id>.jsonl,
and a task skips runs already there, so a preempted, requeued or resubmitted task
continues where it stopped. Failed runs are recorded with their error message and are
not retried. collect and todo list the tasks that still have runs to do; orc.sh submit
resubmits the ones that are not queued, which is how preempted tasks get rerun.

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
parser.add_argument("mode", choices=["manifest", "run", "collect", "todo"])
parser.add_argument(
    "--design",
    default="decision",
    choices=["smoke", "decision", "one-per-set", "full", "dummy"],
)
parser.add_argument("--hours", type=float, default=4.0, help="v1 runtime per task")
parser.add_argument("--sobol", type=int, default=100, help="dummy: Sobol points")
parser.add_argument("--simplex", type=int, default=100, help="dummy: simplex points")
parser.add_argument("--repeats", type=int, default=2, help="dummy: runs per point")
parser.add_argument("--runs-per-task", type=int, default=20, help="dummy")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--exclude", default="", help="todo: task ids to leave out, e.g. 3,7")
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


def read_results(paths=None):
    # a task killed mid-write can leave a partial last line; skip it
    records = []
    for path in sorted(results_dir.glob("task_*.jsonl")) if paths is None else paths:
        for line in path.open() if path.exists() else []:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def array_spec(ids):
    # 0,1,2,5,7,8 -> 0-2,5,7-8
    spans = []
    for i in sorted(ids):
        if spans and i == spans[-1][1] + 1:
            spans[-1][1] = i
        else:
            spans.append([i, i])
    return ",".join(f"{a}-{b}" if b > a else f"{a}" for a, b in spans)


def todo_tasks(runs, records):
    done = {r["run_id"] for r in records}
    return sorted(int(t) for t in runs.loc[~runs["run_id"].isin(done), "task"].unique())


if args.mode == "manifest":
    assert not any(results_dir.glob("*.jsonl")), f"{results_dir} has results; move them first"

if args.mode == "manifest" and args.design != "dummy":
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
        f"submit with: bash orc.sh submit {args.design}"
    )

if args.mode == "manifest" and args.design == "dummy":
    from scipy.stats import qmc

    from matsci_opt_benchmarks.crabnet_hyperparameter.utils.parameters import (
        get_parameters,
    )

    space = {p["name"]: p for p in get_parameters()[0]}
    numeric = [n for n, p in space.items() if p["type"] == "range" and n != "train_frac"]
    other = [n for n in space if n not in numeric]  # train_frac and the categoricals

    def to_values(names, u):
        # unit-cube coordinates to parameter values, as Ax does: ranges scaled, and
        # rounded when both bounds are integers; choices split into equal bins
        out = {}
        for name, x in zip(names, u.T):
            p = space[name]
            if p["type"] == "choice":
                bins = np.minimum((x * len(p["values"])).astype(int), len(p["values"]) - 1)
                out[name] = [p["values"][b] for b in bins]
            else:
                lo, hi = p["bounds"]
                v = lo + x * (hi - lo)
                integer = all(isinstance(b, int) for b in p["bounds"])
                out[name] = np.rint(v).astype(int) if integer else v
        return pd.DataFrame(out)

    names = list(space)
    n_draw = 2 ** int(np.ceil(np.log2(8 * args.sobol)))  # about 1 in 4 meets both constraints
    sobol = to_values(names, qmc.Sobol(len(names), seed=args.seed).random(n_draw))
    ok = (sobol["betas1"] <= sobol["betas2"]) & (sobol["emb_scaler"] + sobol["pos_scaler"] <= 1)
    sobol = sobol[ok].head(args.sobol).assign(block="sobol", active=len(numeric))

    # simplex: every vertex, then half faces (2, 3, 4 active in turn), half interior
    n = len(numeric)
    n_face = (args.simplex - n) // 2
    ks = [1] * n + [2 + i % 3 for i in range(n_face)] + [n] * (args.simplex - n - n_face)
    rng = np.random.default_rng(args.seed)
    cube = qmc.Sobol(n - 1, seed=args.seed + 1).random(len(ks))
    w = np.zeros((len(ks), n))
    for i, k in enumerate(ks):
        active = [i] if k == 1 else rng.choice(n, k, replace=False)
        # spacings of sorted Sobol coordinates are uniform on the face
        w[i, active] = np.diff(np.r_[0, np.sort(cube[i, : k - 1]), 1])
    # betas1 <= betas2 (the betas1 vertex becomes a second betas2 vertex);
    # emb_scaler + pos_scaler <= 1 holds on the simplex already
    b = [numeric.index("betas1"), numeric.index("betas2")]
    w[:, b] = np.sort(w[:, b], axis=1)
    simplex = pd.concat(
        [
            to_values(numeric, w),
            to_values(other, qmc.Sobol(len(other), seed=args.seed + 2).random(len(ks))),
        ],
        axis=1,
    ).assign(block="simplex", active=ks)

    points = pd.concat([sobol, simplex], ignore_index=True)[[*hp, "block", "active"]]
    points.insert(0, "set_id", np.arange(len(points)))
    runs = points.loc[points.index.repeat(args.repeats)]
    runs = runs.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    runs.insert(0, "run_id", np.arange(len(runs)))
    runs["sample_seed"] = rng.integers(0, 1000, len(runs))
    # no runtime estimate for new points, so each task gets an equal share
    n_tasks = int(np.ceil(len(runs) / args.runs_per_task))
    runs["task"] = runs["run_id"] % n_tasks
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(manifest_path, index=False)
    print(
        f"dummy: {len(sobol)} Sobol and {len(simplex)} simplex points, "
        f"{args.repeats} repeats, {len(runs)} runs in {n_tasks} tasks\n"
        f"submit from {rerun_dir} with: sbatch --array=0-{n_tasks - 1} "
        f"--export=ALL,DESIGN=dummy <path to rerun.sbatch>"
    )

if args.mode == "run":
    import torch

    from matsci_opt_benchmarks.crabnet_hyperparameter.utils.parameters import (
        submitit_evaluate,
    )

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"]) + int(os.environ.get("TASK_OFFSET", 0))
    runs = pd.read_csv(manifest_path)
    runs = runs[runs["task"] == task_id]
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"task_{task_id:05d}.jsonl"
    done = [r["run_id"] for r in read_results([out])]
    if out.exists() and out.read_bytes()[-1:] not in (b"", b"\n"):
        with out.open("a") as f:  # end a partial line left by a kill mid-write
            f.write("\n")
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    if "SLURM_JOB_ID" in os.environ:
        # Stop before any run is recorded: a GPU this torch build has no kernels for
        # (Hopper, Blackwell) would otherwise record every run as a failure, and failed
        # runs are not retried. orc.sh submit requeues the task later.
        if not torch.cuda.is_available():
            raise SystemExit(f"task {task_id}: no GPU visible, stopping")
        major, minor = torch.cuda.get_device_capability()
        sms = [a[3:] for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
        if not any(int(s[:-1]) == major and int(s[-1]) <= minor for s in sms):
            raise SystemExit(
                f"task {task_id}: torch {torch.__version__} cannot run on {gpu} "
                f"(sm_{major}{minor}; built for {', '.join(sms)}), stopping"
            )
    print(f"task {task_id}: {len(runs)} runs, {len(done)} already done, on {gpu}")
    for run in runs[~runs["run_id"].isin(done)].to_dict("records"):
        parameters = {k: run[k] for k in hp}
        parameters["bias"] = bool(parameters["bias"])
        parameters["sample_seed"] = int(run["sample_seed"])
        result = submitit_evaluate(parameters)
        record = {
            "run_id": run["run_id"],
            "set_id": run["set_id"],
            "v1_row": run.get("v1_row"),
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
    records = read_results()
    if not records:
        raise SystemExit(f"no results in {results_dir} yet")
    res = pd.json_normalize(records).drop_duplicates("run_id", keep="last")
    todo = todo_tasks(runs, records)
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
        print(f"{len(todo)} tasks with runs to do: {array_spec(todo)}")
    if len(table):
        folds = [table[f"fold_scores.fold_{i}.mae"].median() for i in range(5)]
        print("median MAE by fold (no downward trend expected):", np.round(folds, 3))
    if len(table) and "v1_mae" not in table:
        cols = ["mae", "rmse", "runtime", "model_size"]
        print("medians by block:\n", table.groupby("block")[cols].median().round(3))
        sd = table.groupby("set_id")["mae"].std().median()
        print(f"median repeat SD of MAE: {sd:.4f} eV")
        print("median runtime by GPU [s]:", table.groupby("gpu")["runtime"].median().round(1).to_dict())
    if len(table) and "v1_mae" in table:
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

if args.mode == "todo":
    # one line per job array, "<offset> <--array list>": array task ids only go up to
    # 5000 (MaxArraySize on ORC), so each array covers 5,000 tasks and rerun.sbatch
    # adds TASK_OFFSET to the array task id
    exclude = {int(t) for t in args.exclude.replace(" ", ",").split(",") if t.strip()}
    todo = todo_tasks(pd.read_csv(manifest_path), read_results())
    todo = [t for t in todo if t not in exclude]
    for offset in sorted({t // 5000 * 5000 for t in todo}):
        print(offset, array_spec(t - offset for t in todo if offset <= t < offset + 5000))
