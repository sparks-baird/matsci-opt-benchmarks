"""Compare optimizers on the released CrabNet hyperparameter surrogate.

Single-objective minimization of the surrogate MAE over the 20 numeric and 3
categorical CrabNet hyperparameters, with ``train_frac`` fixed at 1.0 and both
native constraints (``betas1 <= betas2``, ``emb_scaler + pos_scaler <= 1``)
enforced. Every evaluation seen by an optimizer uses a fresh noise percentile
``u ~ U(0, 1)`` as the ``mae_rank`` feature. The noise-free value (rank 0.5) is
stored for reporting. Integer hyperparameters are searched as continuous values
and rounded before evaluation for every method.

The RandomForest in ``surrogate_models.pkl`` (Zenodo 10.5281/zenodo.7694268)
only unpickles with scikit-learn 1.0.x, so its trees are first exported to
``.npy`` arrays and evaluated with a numpy traversal that reproduces
``model.predict`` exactly. Campaigns can then run in any environment.

Usage (from the repository root)::

    # once, with scikit-learn 1.0.x: export and verify the MAE forest
    python scripts/crabnet_hyperparameter/optimizer_benchmark.py export \
        surrogate_models.pkl FOREST_DIR
    # one campaign of 50 evaluations -> OUT_DIR/METHOD_seedSEED.csv
    python scripts/crabnet_hyperparameter/optimizer_benchmark.py run \
        METHOD SEED FOREST_DIR OUT_DIR

METHOD is one of random, grid, sobol, lhs, ga (pymoo), ax (ax-platform),
ax_saasbo (ax-platform + pyro-ppl), bofire (bofire[optimization]).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

N_EVAL, N_INIT = 50, 10
N_RESTARTS, RAW_SAMPLES = 4, 256  # acquisition optimizer effort (Ax, BoFire)
# bounds from matsci_opt_benchmarks.crabnet_hyperparameter.utils.parameters
NUMERIC = {
    "N": (1, 10),
    "alpha": (0.0, 1.0),
    "d_model": (100, 1024),
    "dim_feedforward": (1024, 4096),
    "dropout": (0.0, 1.0),
    "emb_scaler": (0.0, 1.0),
    "eps": (1e-7, 1e-4),
    "epochs_step": (5, 20),
    "fudge": (0.0, 0.1),
    "heads": (1, 10),
    "k": (2, 10),
    "lr": (1e-4, 6e-3),
    "pe_resolution": (2500, 10000),
    "ple_resolution": (2500, 10000),
    "pos_scaler": (0.0, 1.0),
    "weight_decay": (0.0, 1.0),
    "batch_size": (32, 256),
    "out_hidden4": (32, 512),
    "betas1": (0.5, 0.9999),
    "betas2": (0.5, 0.9999),
}
INTEGER = {"N", "d_model", "dim_feedforward", "epochs_step", "heads", "k"}
INTEGER |= {"pe_resolution", "ple_resolution", "batch_size", "out_hidden4"}
CATEGORICAL = {
    "bias": [False, True],
    "criterion": ["RobustL1", "RobustL2"],
    "elem_prop": ["mat2vec", "magpie", "onehot"],
}
NAMES = list(NUMERIC) + list(CATEGORICAL)
LO = np.array([b[0] for b in NUMERIC.values()], dtype=float)
HI = np.array([b[1] for b in NUMERIC.values()], dtype=float)
ARRAYS = ["children_left", "children_right", "feature", "threshold", "value", "roots"]

mode = sys.argv[1]
forest_dir = Path(sys.argv[3] if mode == "export" else sys.argv[4])

if mode == "export":
    import joblib

    model = joblib.load(sys.argv[2])["mae"]
    trees = [e.tree_ for e in model.estimators_]
    roots = np.cumsum([0] + [t.node_count for t in trees])[:-1]
    arrays = {k: [] for k in ARRAYS[:-1]}
    for root, t in zip(roots, trees):
        leaf = t.children_left == -1
        arrays["children_left"].append(np.where(leaf, -1, t.children_left + root))
        arrays["children_right"].append(np.where(leaf, -1, t.children_right + root))
        arrays["feature"].append(t.feature)
        arrays["threshold"].append(t.threshold)
        arrays["value"].append(t.value[:, 0, 0])
    dtypes = [np.int32, np.int32, np.int8, np.float64, np.float64]
    forest_dir.mkdir(parents=True, exist_ok=True)
    for (k, v), dt in zip(arrays.items(), dtypes):
        np.save(forest_dir / f"{k}.npy", np.concatenate(v).astype(dt))
    np.save(forest_dir / "roots.npy", roots.astype(np.int64))

forest = {k: np.load(forest_dir / f"{k}.npy", mmap_mode="r") for k in ARRAYS}


def predict(X):
    """Mean over trees; sklearn compares float32 inputs against thresholds."""
    X = np.asarray(X, dtype=np.float32).astype(np.float64)
    node = np.repeat(np.asarray(forest["roots"])[:, None], len(X), axis=1)
    rows = np.arange(len(X))[None, :]
    while True:
        left = forest["children_left"][node]
        leaf = left < 0
        if leaf.all():
            return forest["value"][node].mean(axis=0)
        feat = np.where(leaf, 0, forest["feature"][node]).astype(np.intp)
        go_left = X[rows, feat] <= forest["threshold"][node]
        node = np.where(leaf, node, np.where(go_left, left, forest["children_right"][node]))


if mode == "export":
    X = np.random.default_rng(0).uniform(size=(500, 29))
    X[:, :20] = LO + X[:, :20] * (HI - LO)
    X[:, 21:28] = np.round(X[:, 21:28])
    print("max |numpy - sklearn|:", np.abs(predict(X) - model.predict(X)).max())
    sys.exit()

def features(p, rank):
    x = [float(np.round(p[k])) if k in INTEGER else float(p[k]) for k in NUMERIC]
    bias = p["bias"] in (True, "True")
    x += [1.0, bias, p["criterion"] == "RobustL1", p["criterion"] == "RobustL2"]
    x += [p["elem_prop"] == e for e in ("magpie", "mat2vec", "onehot")]
    return np.array(x + [1.0, rank], dtype=float)  # hardware_2080ti, mae_rank



if mode == "plot":
    import json

    import matplotlib.pyplot as plt

    out = Path(__file__).resolve().parents[2] / "reports" / "crabnet_hyperparameter_immi"
    labels = {
        "random": "Random",
        "grid": "Grid (3 levels)",
        "sobol": "Sobol",
        "lhs": "LHS",
        "ga": "GA (pymoo)",
        "ax": "Ax (GP, qLogNEI)",
        "ax_saasbo": "Ax + SAASBO",
        "bofire": "BoFire (GP, qLogEI)",
    }
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    colors = dict(zip(labels, palette))
    runs = pd.concat([pd.read_csv(f) for f in sorted(Path(sys.argv[2]).glob("*_seed*.csv"))])
    # campaigns stopped early (compute time) are kept and truncated per method
    # to the number of evaluations reached by all of its seeds
    res = runs.sort_values(["method", "seed", "iteration"])
    curves = {
        m: res[res["method"] == m].pivot(index="iteration", columns="seed", values="best_so_far_noise_free").dropna()
        for m in labels
        if m in set(res["method"])
    }
    common = min(c.index.max() for c in curves.values())
    cols = ["method", "seed", "iteration", "noisy_mae_observed", "mae_noise_free"]
    cols += ["best_so_far_noise_free", "wall_time_s"]
    (out / "revision" / "analysis").mkdir(parents=True, exist_ok=True)
    res[cols].to_csv(out / "revision" / "analysis" / "optimizer_benchmark_results.csv", index=False)

    # reference: surrogate (rank 0.5) on the unique deposited hyperparameter sets
    raw = pd.read_csv(sys.argv[3])
    hp = list(NUMERIC) + ["train_frac"] + list(CATEGORICAL)
    groups = raw.groupby([raw[c].round(6) if raw[c].dtype.kind == "f" else raw[c] for c in hp], sort=False)
    sets = groups[hp].first().reset_index(drop=True)
    X = np.array([features(p, 0.5) for p in sets.to_dict("records")])
    X_recorded = X.copy()
    X_recorded[:, 20] = sets["train_frac"]
    ref = np.concatenate([predict(X_recorded[i : i + 2000]) for i in range(0, len(X), 2000)])
    ref_tf1 = np.concatenate([predict(X[i : i + 2000]) for i in range(0, len(X), 2000)])
    raw_mean = groups["mae"].mean().reset_index(drop=True)
    summary = {
        "task": "minimize surrogate MAE (eV), train_frac = 1.0, noise via random mae_rank",
        "budget": N_EVAL,
        "initial_points_model_based": N_INIT,
        "reference": {
            "n_unique_deposited_sets": len(sets),
            "best_surrogate_mae_rank05_train_frac_as_recorded": ref.min(),
            "best_surrogate_mae_rank05_train_frac_1": ref_tf1.min(),
            "best_repeat_averaged_raw_mae_any_train_frac": raw_mean.min(),
            "best_repeat_averaged_raw_mae_train_frac_ge_0.9": raw_mean[sets.train_frac >= 0.9].min(),
        },
        "methods": {},
        "common_budget_all_methods": int(common),
        "incomplete_campaigns_evaluations_reached": {
            f"{m}_seed{s}": int(n)
            for (m, s), n in runs.groupby(["method", "seed"])["iteration"].max().items()
            if n < N_EVAL
        },
    }
    versions = Path(sys.argv[2]) / "versions.json"
    if versions.exists():
        summary["settings_and_versions"] = json.loads(versions.read_text())

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(17 / 2.54, 8.5 / 2.54), width_ratios=[1.5, 1], constrained_layout=True
    )
    methods = list(curves)
    for i, m in enumerate(methods):
        best = curves[m]
        n, reached = best.shape[1], int(best.index.max())
        mean, se = best.mean(axis=1), best.std(axis=1, ddof=1) / np.sqrt(n)
        style = "--" if m in ("random", "grid", "sobol", "lhs") else "-"
        ax1.plot(best.index, mean, style, color=colors[m], lw=1.4, label=f"{labels[m]} (n = {n})")
        ax1.fill_between(best.index, mean - se, mean + se, color=colors[m], alpha=0.18, lw=0)
        final, at_common = best.loc[reached], best.loc[common]
        wall = res[(res["method"] == m) & (res["iteration"] == reached)]["wall_time_s"]
        summary["methods"][m] = {
            "label": labels[m],
            "n_seeds": n,
            "seeds": [int(s) for s in best.columns],
            "evaluations_reached_by_all_seeds": reached,
            "complete_budget": reached == N_EVAL,
            "final_best_mean": final.mean(),
            "final_best_se": final.std(ddof=1) / np.sqrt(n) if n > 1 else None,
            "final_best_median": final.median(),
            "final_best_min": final.min(),
            "final_best_max": final.max(),
            "best_at_common_budget_mean": at_common.mean(),
            "best_at_common_budget_se": at_common.std(ddof=1) / np.sqrt(n) if n > 1 else None,
            "best_at_eval_10_mean": best.loc[10].mean(),
            "mean_wall_time_to_last_eval_s": wall.mean(),
        }
        final = at_common
        y = len(methods) - 1 - i
        ax2.scatter(final, np.full(n, y), s=12, color=colors[m], alpha=0.8, lw=0, zorder=3)
        ax2.errorbar(final.mean(), y, xerr=final.std(ddof=1) / np.sqrt(n), fmt="|", ms=9, color="0.15", capsize=2, lw=1)
    best_ref = summary["reference"]["best_surrogate_mae_rank05_train_frac_as_recorded"]
    for ax in (ax1, ax2):
        line = ax.axhline if ax is ax1 else ax.axvline
        line(best_ref, color="0.4", lw=0.9, ls=":", zorder=1)
    ax1.text(N_EVAL, best_ref, "best of 41,550 deposited sets", ha="right", va="bottom", fontsize=6.5, color="0.3")
    ax1.set(xlabel="Evaluation number", ylabel="Best noise-free MAE so far (eV)", xlim=(1, N_EVAL), ylim=(0.15, 0.5))
    fig.legend(*ax1.get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False, fontsize=7)
    ax2.set_yticks(range(len(methods))[::-1], [labels[m] for m in methods])
    ax2.set_xlabel(f"Best MAE after {common} evaluations (eV)")
    ax1.set_title("(a)", loc="left", fontsize=8)
    ax2.set_title("(b)", loc="left", fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(out / "figures" / f"optimizer_benchmark.{ext}", dpi=300)
    summary["random_search_median_at_budget"] = summary["methods"].get("random", {}).get("final_best_median")
    text = json.dumps(summary, indent=2, default=float)
    (out / "revision" / "analysis" / "optimizer_benchmark_summary.json").write_text(text)
    sys.exit()

method, seed = sys.argv[2], int(sys.argv[3])
out_file = Path(sys.argv[5]) / f"{method}_seed{seed}.csv"
out_file.parent.mkdir(parents=True, exist_ok=True)
noise_rng = np.random.default_rng([seed, 2023])
rows = []
t0 = time.perf_counter()


def evaluate(p):
    assert p["betas1"] <= p["betas2"] + 1e-6, p  # optimizer tolerance
    assert p["emb_scaler"] + p["pos_scaler"] <= 1.0 + 1e-6, p
    noisy, clean = predict([features(p, noise_rng.uniform()), features(p, 0.5)])
    rows.append(
        {"method": method, "seed": seed, "iteration": len(rows) + 1}
        | {"noisy_mae_observed": noisy, "mae_noise_free": clean}
        | {"wall_time_s": time.perf_counter() - t0}
        | {k: p[k] for k in NAMES}
    )
    df = pd.DataFrame(rows)
    df.insert(5, "best_so_far_noise_free", df["mae_noise_free"].cummin())
    df.to_csv(out_file, index=False)
    return noisy


def from_unit(u):
    """Map a point in [0, 1]^23 to hyperparameters (categoricals by binning)."""
    p = dict(zip(NUMERIC, LO + u[:20] * (HI - LO)))
    for j, (k, vals) in enumerate(CATEGORICAL.items()):
        p[k] = vals[min(int(u[20 + j] * len(vals)), len(vals) - 1)]
    return p


def feasible(p):
    return p["betas1"] <= p["betas2"] and p["emb_scaler"] + p["pos_scaler"] <= 1.0


rng = np.random.default_rng(seed)
if method in ("random", "sobol", "lhs", "grid"):
    from scipy.stats import qmc

    if method == "random":
        U = rng.uniform(size=(1000, 23))
    elif method == "sobol":
        U = qmc.Sobol(23, scramble=True, seed=seed).random(512)
    elif method == "lhs":
        U = qmc.LatinHypercube(23, seed=seed).random(256)
    else:  # 3 levels per numeric HP x all categorical levels, random order
        levels = [rng.integers(0, 3, (5000, 20)) / 2]
        levels += [(rng.integers(0, len(v), (5000, 1)) + 0.5) / len(v) for v in CATEGORICAL.values()]
        U = pd.DataFrame(np.hstack(levels)).drop_duplicates().to_numpy()
    points = [p for p in map(from_unit, U) if feasible(p)][:N_EVAL]
    assert len(points) == N_EVAL
    for p in points:
        evaluate(p)

elif method == "ga":
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.problem import Problem
    from pymoo.core.repair import Repair
    from pymoo.optimize import minimize

    ib = [NAMES.index("betas1"), NAMES.index("betas2")]
    ie = [NAMES.index("emb_scaler"), NAMES.index("pos_scaler")]

    class ConstraintRepair(Repair):
        # both betas share bounds and both scalers live in [0, 1], so the
        # repair is done directly in unit coordinates
        def _do(self, problem, X, **kwargs):
            X = np.array(X, dtype=float)
            X[:, ib] = np.sort(X[:, ib], axis=1)
            total = X[:, ie].sum(axis=1, keepdims=True)
            X[:, ie] = np.where(total > 1.0, X[:, ie] / total, X[:, ie])
            return X

    class SurrogateProblem(Problem):
        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = [evaluate(from_unit(x)) if len(rows) < N_EVAL else np.inf for x in X]

    algorithm = GA(pop_size=10, repair=ConstraintRepair(), eliminate_duplicates=True)
    problem = SurrogateProblem(n_var=23, n_obj=1, xl=0.0, xu=1.0)
    minimize(problem, algorithm, ("n_eval", N_EVAL), seed=seed)

elif method in ("ax", "ax_saasbo"):
    import torch
    from ax import ChoiceParameterConfig, Client, RangeParameterConfig

    torch.set_num_threads(1)
    client = Client(random_seed=seed)
    params = [
        RangeParameterConfig(name=k, bounds=(float(a), float(b)), parameter_type="float")
        for k, (a, b) in NUMERIC.items()
    ]
    params += [ChoiceParameterConfig(name="bias", values=[False, True], parameter_type="bool")]
    params += [
        ChoiceParameterConfig(name=k, values=CATEGORICAL[k], parameter_type="str", is_ordered=False)
        for k in ("criterion", "elem_prop")
    ]
    client.configure_experiment(
        parameters=params,
        parameter_constraints=["betas1 <= betas2", "emb_scaler + pos_scaler <= 1.0"],
    )
    client.configure_optimization(objective="-mae")
    from ax.adapter.registry import Generators
    from ax.generation_strategy.generation_node import GenerationStep
    from ax.generation_strategy.generation_strategy import GenerationStrategy
    from ax.generators.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

    # default BoTorch generator (SingleTaskGP, qLogNEI) or SAAS GP with reduced
    # NUTS settings; fewer acquisition restarts than Ax's default (20 restarts,
    # 1024 raw samples) because the 12 categorical combinations are enumerated
    # and each is optimized with SLSQP, which was too slow on shared CPUs
    bo_kwargs = {}
    if method == "ax_saasbo":
        nuts = {"warmup_steps": 128, "num_samples": 64, "thinning": 16}
        saas = ModelConfig(botorch_model_class=SaasFullyBayesianSingleTaskGP, mll_options=nuts)
        bo_kwargs = {"surrogate_spec": SurrogateSpec(model_configs=[saas])}
    acqf_opt = {"optimizer_kwargs": {"num_restarts": N_RESTARTS, "raw_samples": RAW_SAMPLES}}
    steps = [
        GenerationStep(generator=Generators.SOBOL, num_trials=N_INIT, generator_kwargs={"seed": seed}),
        GenerationStep(
            generator=Generators.BOTORCH_MODULAR,
            num_trials=-1,
            generator_kwargs=bo_kwargs,
            generator_gen_kwargs={"model_gen_options": acqf_opt},
        ),
    ]
    client.set_generation_strategy(GenerationStrategy(name=method, nodes=steps))
    while len(rows) < N_EVAL:
        for idx, p in client.get_next_trials(max_trials=1).items():
            client.complete_trial(trial_index=idx, raw_data={"mae": evaluate(p)})

elif method == "bofire":
    import bofire.strategies.api as strategies
    import torch
    from bofire.data_models.acquisition_functions.api import qLogEI
    from bofire.data_models.constraints.api import LinearInequalityConstraint
    from bofire.data_models.domain.api import Constraints, Domain, Inputs, Outputs
    from bofire.data_models.features.api import CategoricalInput, ContinuousInput, ContinuousOutput
    from bofire.data_models.objectives.api import MinimizeObjective
    from bofire.data_models.strategies.api import BotorchOptimizer, RandomStrategy, SoboStrategy

    torch.set_num_threads(1)
    inputs = [ContinuousInput(key=k, bounds=(a, b)) for k, (a, b) in NUMERIC.items()]
    inputs += [CategoricalInput(key=k, categories=[str(v) for v in c]) for k, c in CATEGORICAL.items()]
    domain = Domain(
        inputs=Inputs(features=inputs),
        outputs=Outputs(features=[ContinuousOutput(key="mae", objective=MinimizeObjective())]),
        constraints=Constraints(
            constraints=[
                LinearInequalityConstraint(features=["betas1", "betas2"], coefficients=[1.0, -1.0], rhs=0.0),
                LinearInequalityConstraint(features=["emb_scaler", "pos_scaler"], coefficients=[1.0, 1.0], rhs=1.0),
            ]
        ),
    )
    candidates = strategies.map(RandomStrategy(domain=domain, seed=seed)).ask(N_INIT)
    acqf_opt = BotorchOptimizer(n_restarts=N_RESTARTS, n_raw_samples=RAW_SAMPLES)
    sobo = SoboStrategy(domain=domain, acquisition_function=qLogEI(), acquisition_optimizer=acqf_opt, seed=seed)
    sobo = strategies.map(sobo)
    while len(rows) < N_EVAL:
        X = candidates[NAMES].reset_index(drop=True)
        sobo.tell(X.assign(mae=[evaluate(p) for p in X.to_dict("records")], valid_mae=1))
        if len(rows) < N_EVAL:
            candidates = sobo.ask(1)
