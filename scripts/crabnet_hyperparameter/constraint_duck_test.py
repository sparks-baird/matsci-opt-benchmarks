"""Duck test for the x1 to x20 composition constraint on the CrabNet benchmark.

The HuggingFace Space AccelerationConsortium/crabnet-hyperparameter exposes the 20
numerical hyperparameters as x1 to x20, each min-max scaled to [0, 1]. The composition
variant checked here treats them like element fractions: x1 + ... + x20 = 1, at least
3 parameters are active (nonzero), and every inactive parameter sits at 0, the lowest
value of that hyperparameter. The Space's own constraints still apply (x19 <= x20,
x6 + x15 <= 1). fidelity1 = 1 and the categoricals stay at c1_0, c2_0, c3_0.

The question is whether the surrogate, restricted to that region, behaves like a real
composition-property problem. The reference problems are Materials Project (MP)
compositions with at least 3 elements, each drawn from a fixed 20-element set. The
comparison covers input cross-correlations and sparsity, SHAP feature importances and
effects, and response-surface statistics (predictability, linear mixing, smoothness,
noise).

Inputs (not tracked in git):

- ``models/surrogate_models_hgbr_opt.pkl`` from the Space, saved to
  ``models/crabnet_hyperparameter``. It reproduces the live Space API (checked below).
- ``sobol_regression.csv`` from Zenodo 10.5281/zenodo.7694268 (identical to the copy on
  the Space) saved to ``data/external/crabnet_hyperparameter``
- MP summary build 2026.04.13 from the MP AWS Open Data bucket (no API key needed):
  ``s3://materialsproject-build/collections/summary/version=2026-04-13/`` saved as
  ``data/external/materials_project/summary_2026-04-13.parquet``

The surrogate was pickled with scikit-learn 1.4.1.post1. Run with Python 3.12,
scikit-learn 1.4.1.post1, numpy 1.26, pandas, pyarrow, shap 0.45, matplotlib, and
gradio_client.
"""

# %% imports
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shap
from gradio_client import Client
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

rng = np.random.default_rng(42)
fig_dir = Path("reports/crabnet_hyperparameter_immi/figures")
n_samples = 5_000  # points per design; also rows per dataset for the model fits
n_explain, n_background = 300, 100  # SHAP computed directly on the surrogate
min_active = 3

# %% search space, in the order of x1..x20 on the Space (surrogate.PARAM_BOUNDS)
bounds = {
    "N": (1, 10),
    "alpha": (0.0, 1.0),
    "d_model": (100, 1024),
    "dim_feedforward": (1024, 4096),
    "dropout": (0.0, 1.0),
    "emb_scaler": (0.0, 1.0),
    "epochs_step": (5, 20),
    "eps": (1e-7, 1e-4),
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
hp = list(bounds)
labels = [f"x{i + 1} {p}" for i, p in enumerate(hp)]
lo = np.array([b[0] for b in bounds.values()])
hi = np.array([b[1] for b in bounds.values()])
i_alpha = hp.index("alpha")
objectives = ["mae", "rmse", "runtime", "model_size"]
fixed = {  # fidelity1 = 1, c1_0 (no bias), c2_0 (RobustL1), c3_0 (mat2vec)
    "train_frac": 1.0,
    "bias": 0,
    "use_RobustL1": 1,
    "elem_prop_magpie": 0,
    "elem_prop_mat2vec": 1,
    "elem_prop_onehot": 0,
}
surrogate = joblib.load("models/crabnet_hyperparameter/surrogate_models_hgbr_opt.pkl")


def features(x, obj, u=0.5):
    """Surrogate inputs for scaled points x, built as in the Space's surrogate.py."""
    frame = pd.DataFrame(lo + x * (hi - lo), columns=hp).assign(**fixed)
    if obj != "model_size":  # noise percentile; the Space uses 1 - u for runtime
        frame[f"{obj}_rank"] = 1 - u if obj == "runtime" else u
    return frame[surrogate[obj].feature_names_in_].astype(float)


def predict(x, u=0.5):
    return pd.DataFrame({o: surrogate[o].predict(features(x, o, u)) for o in objectives})


def composition(n, k_choices):
    """Sum-to-one points with k active parameters and the rest at their lowest value."""
    x = np.zeros((n, 20))
    for row, k in zip(x, rng.choice(k_choices, n)):
        row[rng.choice(20, k, replace=False)] = rng.dirichlet(np.ones(k))
    swap = x[:, 18] > x[:, 19]  # x19 <= x20
    x[swap, 18], x[swap, 19] = x[swap, 19], x[swap, 18]
    return x


# %% the local surrogate reproduces the live Space
client = Client("AccelerationConsortium/crabnet-hyperparameter")
u_grid = np.linspace(0, 1, 2001)
for x in composition(8, np.arange(min_active, 21)):
    space = client.predict(*x.tolist(), "c1_0", "c2_0", "c3_0", 1.0, api_name="/predict")
    space = np.array(space["data"][0], float)
    # each Space call draws one percentile u: MAE and RMSE at u, runtime at 1 - u
    local = predict(np.repeat(x[None], len(u_grid), 0), u_grid)
    assert np.isclose(local, space, rtol=1e-6).all(1).any(), space

# %% CrabNet Sobol data (real training runs)
sobol = pd.read_csv("data/external/crabnet_hyperparameter/sobol_regression.csv")
x_sobol = (sobol[hp].to_numpy() - lo) / (hi - lo)
group_cols = hp + ["train_frac", "bias", "criterion", "elem_prop"]
sobol["group"] = sobol[group_cols].round(6).groupby(group_cols).ngroup()
x_sobol_unique = x_sobol[~sobol["group"].duplicated().to_numpy()]

# %% Materials Project compositions with at least 3 elements
mp_cols = [
    "formula_pretty", "composition", "elements", "nsites", "formation_energy_per_atom",
    "energy_above_hull", "band_gap", "density", "total_magnetization", "deprecated",
]  # fmt: skip
mp = pq.read_table(
    "data/external/materials_project/summary_2026-04-13.parquet", columns=mp_cols
).to_pandas()
mp = mp[~mp["deprecated"] & mp["formation_energy_per_atom"].notna()]
mp["magnetization"] = mp["total_magnetization"] / mp["nsites"]
mp["element_set"] = mp["elements"].apply(frozenset)
mp = mp[mp["element_set"].apply(len) >= min_active]
mp = mp.rename(
    columns={
        "formation_energy_per_atom": "E_form",
        "energy_above_hull": "E_hull",
        "band_gap": "E_gap",
    }
)
element_sets = {
    "MP alloys": "Al Co Cr Cu Fe Hf Mg Mn Mo Nb Ni Re Si Sn Ta Ti V W Zn Zr",
    "MP oxides": "Al Ba Bi Ca Co Cr Cu Fe Li Mg Mn Na Nb Ni O Sr Ti V W Zn",
    "MP mixed": "C Ca Co Cr Cu F Fe H Li Mg Mn N Na Ni O P S Si Ti V",
}
mp_targets = {
    "MP alloys": ["E_form", "E_hull", "density", "magnetization"],
    "MP oxides": ["E_gap", "E_form", "E_hull", "density"],
    "MP mixed": ["E_form", "E_gap", "E_hull", "density"],
}
datasets, noise_rows = {}, []
for name, els in element_sets.items():
    els = els.split()
    sub = mp[mp["element_set"].apply(lambda s: s <= frozenset(els))]
    if name == "MP oxides":
        sub = sub[sub["element_set"].apply(lambda s: "O" in s)]
    # polymorphs share a reduced formula; their spread is the MP analogue of repeats
    for t in mp_targets[name]:
        s = sub.dropna(subset=[t])
        grp = s.groupby("formula_pretty")[t]
        multi = grp.transform("size") > 1
        dev = (s[t] - grp.transform("mean"))[multi]
        noise_rows.append(
            {
                "dataset": name,
                "target": t,
                "within_share": (dev**2).sum()
                / ((s[t][multi] - s[t][multi].mean()) ** 2).sum(),
            }
        )
    # one row per composition: the lowest-energy polymorph
    gs = sub.sort_values("E_hull").drop_duplicates("formula_pretty")
    comp = pd.DataFrame([dict(c) for c in gs["composition"]]).reindex(columns=els)
    comp = comp.fillna(0.0).to_numpy()
    datasets[name] = {
        "X": comp / comp.sum(1, keepdims=True),
        "Y": gs[mp_targets[name]].reset_index(drop=True),
        "features": els,
        "family": "MP",
    }
mp_x = np.vstack([ds["X"] for ds in datasets.values()])
k_mp = (mp_x > 0).sum(1)  # number of elements, used as the number of active parameters

# CrabNet replicate noise from the real Sobol runs
for t in ["mae", "rmse", "runtime"]:
    grp = sobol.groupby("group")[t]
    multi = grp.transform("size") > 1
    dev = (sobol[t] - grp.transform("mean"))[multi]
    noise_rows.append(
        {
            "dataset": "CrabNet Sobol runs",
            "target": t,
            "within_share": (dev**2).sum()
            / ((sobol[t][multi] - sobol[t][multi].mean()) ** 2).sum(),
        }
    )

# %% point sets on which the surrogate is evaluated
x_cube = rng.random((n_samples, 20))
flip = x_cube[:, 5] + x_cube[:, 14] > 1  # x6 + x15 <= 1
x_cube[flip, 5], x_cube[flip, 14] = 1 - x_cube[flip, 14], 1 - x_cube[flip, 5]
x_cube[:, 18:20] = np.sort(x_cube[:, 18:20], axis=1)  # x19 <= x20
x_comp = composition(n_samples, k_mp)  # as many active parameters as MP has elements
x_by_k = {k: composition(200, [k]) for k in [3, 4, 6, 10, 20]}

# MP alloy compositions mapped onto x1..x20 (same inputs, other function)
perms = [rng.permutation(20) for _ in range(3)]
x_alloy_perms = []
for perm in perms:
    x = np.empty_like(datasets["MP alloys"]["X"])
    x[:, perm] = datasets["MP alloys"]["X"]
    swap = x[:, 18] > x[:, 19]
    x[swap, 18], x[swap, 19] = x[swap, 19], x[swap, 18]
    x_alloy_perms.append(x)

pred_cube, pred_comp = predict(x_cube), predict(x_comp)
datasets["CrabNet no constraint"] = {"X": x_cube, "Y": pred_cube}
datasets["CrabNet sum-to-one"] = {"X": x_comp, "Y": pred_comp}
for j, (perm, x) in enumerate(zip(perms, x_alloy_perms)):
    datasets[f"CrabNet on MP alloys {j}"] = {
        "X": x,
        "Y": predict(x),
        "features": [
            f"{e}→x{perm[i] + 1}"
            for i, e in enumerate(element_sets["MP alloys"].split())
        ],
    }
for name, ds in datasets.items():
    ds["noise_key"] = name
    if name.startswith("CrabNet"):
        ds.setdefault("features", labels)
        ds["family"] = "CrabNet"
        ds["Y"] = ds["Y"][["mae", "runtime", "model_size"]]
        if name != "CrabNet no constraint":
            ds["noise_key"] = "CrabNet sum-to-one"

# surrogate noise at a fixed input: spread over the percentile (rank) input
u_noise = np.linspace(0.05, 0.95, 19)
for key, x in {"no constraint": x_cube, "sum-to-one": x_comp}.items():
    x = x[:500]
    reps = pd.concat([predict(x, u).assign(point=range(len(x))) for u in u_noise])
    for t in ["mae", "rmse", "runtime"]:
        dev = reps[t] - reps.groupby("point")[t].transform("mean")
        noise_rows.append(
            {
                "dataset": f"CrabNet {key}",
                "target": t,
                "within_share": (dev**2).sum() / ((reps[t] - reps[t].mean()) ** 2).sum(),
            }
        )
noise = pd.DataFrame(noise_rows)

# %% SHAP directly on the surrogate (interventional, background from the same design)
def surrogate_shap(model, X, n_bg):
    explainer = shap.TreeExplainer(
        model, data=X.iloc[:n_bg], feature_perturbation="interventional"
    )
    sv = explainer.shap_values(X.iloc[n_bg:], check_additivity=False)
    return pd.DataFrame(sv, columns=X.columns)[hp].to_numpy()


tasks = [
    (key, obj, x[: n_background + n_explain], n_background)
    for key, x in [("no constraint", x_cube), ("sum-to-one", x_comp)]
    for obj in objectives
] + [(k, obj, x[:200], 50) for k, x in x_by_k.items() for obj in objectives]
shap_surrogate = dict(
    zip(
        [t[:2] for t in tasks],
        Parallel(n_jobs=4)(
            delayed(surrogate_shap)(surrogate[obj], features(x, obj), n_bg)
            for _, obj, x, n_bg in tasks
        ),
    )
)
imp_surrogate = {
    key: np.abs(sv).mean(0) / np.abs(sv).mean(0).sum()
    for key, sv in shap_surrogate.items()
}


def n_eff(imp):
    return 1 / (imp**2).sum()


# %% landscape metrics for every dataset and target (same model pipeline for all)
kf = KFold(5, shuffle=True, random_state=0)
rows, curves = [], {}
for name, ds in datasets.items():
    for t in ds["Y"].columns:
        y_all = ds["Y"][t].to_numpy(float)
        ok = np.flatnonzero(np.isfinite(y_all))
        idx = rng.choice(ok, min(n_samples, len(ok)), replace=False)
        X, y = ds["X"][idx], y_all[idx]
        hgb = HistGradientBoostingRegressor(max_iter=300, random_state=0)
        r2 = r2_score(y, cross_val_predict(hgb, X, y, cv=kf))
        r2_lin = r2_score(y, cross_val_predict(RidgeCV(np.logspace(-6, 3, 19)), X, y, cv=kf))
        hgb.fit(X, y)
        s_idx = rng.choice(len(X), min(1_500, len(X)), replace=False)
        sv = shap.TreeExplainer(hgb).shap_values(X[s_idx])
        imp = np.abs(sv).mean(0) / np.abs(sv).mean(0).sum()
        nbr = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X[s_idx])[1][:, 1]
        # composition-style effect: mean change when a component is present
        present = X > 0
        effect = np.array(
            [
                y[p].mean() - y[~p].mean() if 0 < p.sum() < len(p) else np.nan
                for p in present.T
            ]
        )
        rows.append(
            {
                "dataset": name,
                "family": ds["family"],
                "target": t,
                "n": len(y),
                "r2_hgb": r2,
                "r2_linear": r2_lin,
                "lin_share": max(r2_lin, 0) / r2 if r2 > 0.2 else np.nan,
                "n_eff": n_eff(imp),
                "top_share": imp.max(),
                "nn_autocorr": np.corrcoef(y[s_idx], y[nbr])[0, 1],
                "top_features": ", ".join(
                    np.array(ds["features"])[np.argsort(-imp)[:3]]
                ),
            }
        )
        curves[(name, t)] = {
            "imp": imp,
            "effect": effect / y.std(),
            "shap": sv,
            "x": X[s_idx],
        }
metrics = pd.DataFrame(rows)
noise_share = noise.set_index(["dataset", "target"])["within_share"]
metrics["within_share"] = [
    noise_share.get((datasets[d]["noise_key"], t))
    for d, t in metrics[["dataset", "target"]].itertuples(index=False)
]
surrogate_rows = [
    {"design": str(key), "objective": obj, "n_eff_surrogate": n_eff(imp)}
    | dict(zip(labels, imp))
    for (key, obj), imp in imp_surrogate.items()
]
metrics.to_csv(fig_dir / "constraint_duck_test_metrics.csv", index=False)
pd.DataFrame(surrogate_rows).to_csv(
    fig_dir / "constraint_duck_test_importance.csv", index=False
)
input_corr = {name: np.corrcoef(ds["X"], rowvar=False) for name, ds in datasets.items()}

# %% figure style: categorical slots, ink, and ramps from the dataviz reference palette
blue, orange, aqua, light = "#2a78d6", "#eb6834", "#1baf7a", "#86b6ef"
ink, ink2, muted, rule = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": rule,
        "axes.linewidth": 0.8,
        "xtick.color": ink2,
        "ytick.color": ink2,
        "axes.labelcolor": ink,
        "text.color": ink,
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "savefig.dpi": 200,
    }
)
diverging = LinearSegmentedColormap.from_list(
    "blue_red", ["#256abf", "#6da7ec", "#f0efec", "#e66767", "#e34948"]
)
nice = {
    "mae": "MAE",
    "rmse": "RMSE",
    "runtime": "runtime",
    "model_size": "model size",
    "E_form": "formation energy",
    "E_hull": "energy above hull",
    "E_gap": "band gap",
    "density": "density",
    "magnetization": "magnetization",
}
mp_names = list(element_sets)
mp_imp = [curves[(d, t)]["imp"] for d in mp_names for t in mp_targets[d]]
mp_neff = [n_eff(imp) for imp in mp_imp]
alpha_on = x_comp[:, i_alpha] > 0
suptitle = dict(x=0.01, ha="left", fontsize=9.5, fontweight="bold")

# %% figure 1: feature importances with and without the constraint
fig = plt.figure(figsize=(12.5, 10.5), layout="constrained")
top, bottom = fig.subfigures(2, 1, height_ratios=[1.35, 1])
axs = top.subplots(1, 4, sharey=True)
ypos = np.arange(20)
for ax, obj, letter in zip(axs, objectives, "abcd"):
    for dy, key, c in [(-0.2, "no constraint", light), (0.2, "sum-to-one", blue)]:
        ax.barh(ypos + dy, imp_surrogate[(key, obj)], height=0.38, color=c, label=key)
    ax.set_title(
        f"{letter}  {nice[obj]}: effective inputs"
        f" {n_eff(imp_surrogate[('no constraint', obj)]):.1f} → "
        f"{n_eff(imp_surrogate[('sum-to-one', obj)]):.1f}"
    )
    ax.set_xlabel("share of mean |SHAP|")
    ax.set_xlim(0, 0.8)
    ax.grid(axis="y", visible=False)
axs[0].set_yticks(ypos, labels)
axs[0].invert_yaxis()
axs[0].legend(
    handles=[
        plt.Rectangle((0, 0), 1, 1, color=light, label="no constraint (uniform in the box)"),
        plt.Rectangle((0, 0), 1, 1, color=blue,
                      label=f"sum-to-one, ≥{min_active} active, inactive at lowest value"),
    ],
    loc="lower right",
)  # fmt: skip
top.suptitle(
    "Feature importance computed directly on the HuggingFace Space surrogate"
    " (interventional SHAP, fidelity1 = 1, c1_0, c2_0, c3_0)",
    **suptitle,
)

axs = bottom.subplots(1, 4)
ax = axs[0]
ks = list(x_by_k)
for obj, c, mk in zip(
    objectives, [blue, light, aqua, orange], ["o", "s", "^", "D"]
):  # fmt: skip
    ax.plot(ks, [n_eff(imp_surrogate[(k, obj)]) for k in ks], color=c, marker=mk,
            ms=5, lw=2, label=nice[obj])  # fmt: skip
ax.axhspan(min(mp_neff), max(mp_neff), color=orange, alpha=0.1, lw=0)
ax.text(20, max(mp_neff) - 0.3, "range for MP properties", ha="right", va="top",
        color=ink2, fontsize=7)  # fmt: skip
ax.set_xlabel("number of active parameters (sum-to-one)")
ax.set_ylabel("effective inputs, 1 / Σ share²")
ax.set_ylim(0, 20)
ax.legend(loc="upper left", ncol=2)
ax.set_title("e  Spread vs. number of active inputs")

ax = axs[1]
ranks = np.arange(1, 21)
for imp in mp_imp:
    ax.plot(ranks, np.sort(imp)[::-1], color=orange, lw=1, alpha=0.6)
for key, c in [("no constraint", light), ("sum-to-one", blue)]:
    ax.plot(ranks, np.sort(imp_surrogate[(key, "mae")])[::-1], color=c, lw=2.2)
ax.set_yscale("log")
ax.set_ylim(1e-3, 1)
ax.set_xlabel("input rank")
ax.set_ylabel("share of mean |SHAP|")
ax.legend(
    handles=[
        plt.Line2D([], [], color=light, lw=2.2, label="CrabNet MAE, no constraint"),
        plt.Line2D([], [], color=blue, lw=2.2, label="CrabNet MAE, sum-to-one"),
        plt.Line2D([], [], color=orange, lw=1, label="MP properties (12)"),
    ],
    loc="lower left",
)
ax.set_title("f  Sorted importance profiles")

ax = axs[2]
bins = np.linspace(0.15, 1.0, 69)
for mask, c, lab in [(alpha_on, blue, "alpha (x2) active"),
                     (~alpha_on, muted, "alpha (x2) at 0")]:  # fmt: skip
    ax.hist(pred_comp["mae"][mask], bins=bins, color=c, alpha=0.8,
            label=f"{lab} ({mask.mean():.0%} of points)")  # fmt: skip
ax.set_xlabel("surrogate MAE (eV), sum-to-one points")
ax.set_ylabel("points")
ax.legend(loc="upper left")
ax.set_title("g  Without alpha the model does not learn")

ax = axs[3]
edges = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
mid = (edges[:-1] + edges[1:]) / 2
for sel, c, lab in [
    (sobol["train_frac"] > 0.8, blue, "Sobol runs, train_frac > 0.8"),
    (sobol["train_frac"] > 0, muted, "all Sobol runs"),
]:
    b = pd.cut(sobol.loc[sel, "alpha"], edges, include_lowest=True)
    q = sobol.loc[sel, "mae"].groupby(b, observed=False).quantile([0.25, 0.5, 0.75])
    q = q.unstack()
    ax.fill_between(mid, q[0.25], q[0.75], color=c, alpha=0.15, lw=0)
    ax.plot(mid, q[0.5], color=c, lw=2, marker="o", ms=4, label=lab)
ax.set_xlabel("alpha (Lookahead step size, scaled)")
ax.set_ylabel("MAE of real training runs (eV)")
ax.legend(loc="upper right", title="median and IQR", title_fontsize=7)
ax.set_title("h  Same effect in the real runs")
fig.suptitle(
    "Are importances spread out? Yes without the constraint, no with it",
    **suptitle,
)
fig.savefig(fig_dir / "constraint_duck_test_importance.png")

# %% figure 2: inputs, and where the constrained points sit relative to the data
fig = plt.figure(figsize=(12, 9.6), layout="constrained")
top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.85])
heat = {
    "CrabNet no constraint": "CrabNet, no constraint",
    "CrabNet sum-to-one": f"CrabNet, sum-to-one (≥{min_active} active)",
    "MP alloys": f"MP alloys (≥{min_active} elements)",
    "MP oxides": f"MP oxides (≥{min_active} elements)",
}
axs = top.subplots(1, 4)
for ax, (key, title) in zip(axs, heat.items()):
    C = input_corr[key].copy()
    np.fill_diagonal(C, np.nan)
    im = ax.imshow(C, cmap=diverging, vmin=-0.5, vmax=0.5)
    ax.set_title(title, loc="center")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines[:].set_visible(False)
    ax.set_xlabel(f"mean off-diagonal r = {np.nanmean(C):+.3f}", color=ink2)
axs[0].set_xlabel(
    axs[0].get_xlabel() + "\n(±0.5 cells: x6 + x15 ≤ 1, x19 ≤ x20)", color=ink2
)
cbar = top.colorbar(im, ax=axs, shrink=0.8, label="Pearson r between two inputs")
cbar.outline.set_visible(False)
top.suptitle("a  Input cross-correlations (20 × 20, diagonal blank)", **suptitle)

axs = bottom.subplots(2, 4)
ax = axs[0, 0]
closure = {
    "CrabNet no constraint": (light, "CrabNet, no constraint"),
    "CrabNet sum-to-one": (blue, "CrabNet, sum-to-one"),
    "MP alloys": (orange, "MP alloys"),
    "MP oxides": (orange, "MP oxides"),
    "MP mixed": (orange, "MP mixed"),
}
for i, (key, (c, label)) in enumerate(closure.items()):
    C = input_corr[key]
    off = C[~np.eye(len(C), dtype=bool)]
    ax.scatter(off, i + rng.uniform(-0.2, 0.2, len(off)), s=5, color=c, alpha=0.4, lw=0)
    ax.plot([off.mean()] * 2, [i - 0.32, i + 0.32], color=ink, lw=2)
    ax.plot(-1 / (len(C) - 1), i - 0.45, "v", ms=5, mfc="white", mec=ink2)
ax.set_yticks(range(len(closure)), [v[1] for v in closure.values()])
ax.invert_yaxis()
ax.set_xlim(-0.55, 0.55)
ax.set_xlabel("pairwise input correlation")
ax.legend(
    handles=[
        plt.Line2D([], [], color=ink, lw=2, label="mean"),
        plt.Line2D([], [], ls="", marker="v", mfc="white", mec=ink2, label="−1/(d−1)"),
    ],
    loc="lower right",
)
ax.set_title("b  Closure: mean r vs. −1/(d−1)")

ax = axs[0, 1]
ax.hist(
    [k_mp, (x_comp > 0).sum(1)],
    bins=np.arange(0.5, 21.5),
    color=[orange, blue],
    label=["MP, three 20-element sets", "CrabNet, sum-to-one"],
    density=True,
    rwidth=0.9,
)
ax.set_xlim(0.5, 12.5)
ax.set_xlabel("active components per point (of 20)")
ax.set_ylabel("fraction of points")
ax.legend(loc="upper right")
ax.set_title("c  Sparsity (matched by design)")

ax = axs[0, 2]
for x, c, label in [(mp_x, orange, "MP"), (x_comp, blue, "CrabNet, sum-to-one")]:
    kw = dict(bins=np.linspace(0, 1, 41), density=True, color=c)
    ax.hist(x.max(1), histtype="stepfilled", alpha=0.12, **kw)
    ax.hist(x.max(1), histtype="step", lw=1.8, label=label, **kw)
ax.set_xlabel("largest component fraction")
ax.set_ylabel("density")
ax.legend(loc="upper left")
ax.set_title("d  Dominant component")

ax = axs[0, 3]
at_floor = {
    "Sobol runs": ((x_sobol_unique < 0.05).sum(1), muted),
    "CrabNet, sum-to-one": ((x_comp < 0.05).sum(1), blue),
}
ax.hist(
    [v[0] for v in at_floor.values()],
    bins=np.arange(-0.5, 21.5),
    color=[v[1] for v in at_floor.values()],
    label=list(at_floor),
    density=True,
    rwidth=0.9,
)
ax.set_xlabel("inputs within 0.05 of lowest value")
ax.set_ylabel("fraction of points")
ax.legend(loc="upper center")
ax.set_title("e  No training run looks like this")

nn = NearestNeighbors(n_neighbors=1).fit(x_sobol_unique)
domain = {
    "no constraint": (x_cube, light),
    "sum-to-one": (x_comp, blue),
    "on MP alloy compositions": (x_alloy_perms[0], aqua),
}
ax = axs[1, 0]
bins = np.linspace(0.3, 1.7, 57)
for label, (x, c) in domain.items():
    d = nn.kneighbors(x[rng.choice(len(x), 2000, replace=False)])[0][:, 0]
    ax.hist(d, bins=bins, histtype="stepfilled", color=c, alpha=0.12)
    ax.hist(d, bins=bins, histtype="step", color=c, lw=1.8, label=label)
ax.set_xlabel("distance to nearest Sobol run (scaled units)")
ax.set_ylabel("points")
ax.set_ylim(top=ax.get_ylim()[1] * 1.35)
ax.legend(loc="upper right")
ax.set_title("f  Distance from the training runs")

for ax, obj, xlabel, letter in [
    (axs[1, 1], "mae", "MAE (eV)", "g"),
    (axs[1, 2], "model_size", "model size (parameters)", "h"),
]:
    ecdf = {"Sobol runs (all fidelities)": (sobol[obj], muted),
            "no constraint": (pred_cube[obj], light),
            "sum-to-one": (pred_comp[obj], blue)}  # fmt: skip
    for label, (v, c) in ecdf.items():
        v = np.sort(v)
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, lw=2, label=label)
    ax.set_xlabel(xlabel + "; surrogate at median noise")
    ax.set_ylabel("cumulative fraction")
    ax.set_title(f"{letter}  {nice[obj]} values reached")
axs[1, 2].set_xscale("log")
axs[1, 1].legend(loc="upper left", bbox_to_anchor=(0, -0.16))

ax = axs[1, 3]
pairs = [("mae", "rmse"), ("mae", "runtime"), ("mae", "model_size"),
         ("runtime", "model_size")]  # fmt: skip
short = {"mae": "MAE", "rmse": "RMSE", "runtime": "runtime", "model_size": "size"}
bars = {
    "Sobol runs (all fidelities)": (sobol[objectives], muted),
    "no constraint": (pred_cube, light),
    "sum-to-one": (pred_comp, blue),
}
for j, (label, (frame, c)) in enumerate(bars.items()):
    C = frame.corr("spearman")
    ax.barh(np.arange(len(pairs)) + (j - 1) * 0.27, [C.loc[a, b] for a, b in pairs],
            height=0.25, color=c, label=label)  # fmt: skip
ax.set_yticks(range(len(pairs)), [f"{short[a]} vs {short[b]}" for a, b in pairs])
ax.invert_yaxis()
ax.axvline(0, color=rule, lw=0.8)
ax.set_xlim(-0.7, 1.05)
ax.set_xlabel("Spearman correlation between objectives")
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16))
ax.set_title("i  Objective trade-offs")
fig.suptitle(
    "Looks like a duck? Inputs under the x1 to x20 composition constraint, and where"
    " they sit relative to the training runs",
    **suptitle,
)
fig.savefig(fig_dir / "constraint_duck_test_inputs.png")

# %% figure 3: response-surface scorecard
card_rows = [
    ("CrabNet no constraint", "CrabNet, no constraint", ["mae", "runtime", "model_size"], "cube"),
    ("CrabNet sum-to-one", "CrabNet, sum-to-one", ["mae", "runtime", "model_size"], "comp"),
    ("CrabNet on MP alloys", "CrabNet on MP alloy comps.", ["mae", "runtime", "model_size"], "mpin"),
    ("MP alloys", "MP alloys", mp_targets["MP alloys"], "mp"),
    ("MP oxides", "MP oxides", mp_targets["MP oxides"], "mp"),
    ("MP mixed", "MP mixed", mp_targets["MP mixed"], "mp"),
]  # fmt: skip
marker = {
    "cube": dict(mfc="white", mec=blue),
    "comp": dict(mfc=blue, mec=blue),
    "mpin": dict(mfc=aqua, mec=aqua),
    "mp": dict(mfc=orange, mec=orange),
}
m = metrics.assign(dataset=metrics["dataset"].str.replace(r" \d$", "", regex=True))
cols = ["r2_hgb", "lin_share", "n_eff", "nn_autocorr", "within_share"]
agg = m.groupby(["dataset", "target"])[cols].agg(["mean", "min", "max"])
card = pd.DataFrame(
    [
        {"key": key, "label": f"{label} · {nice[t]}", "kind": kind, "target": t}
        for key, label, targets, kind in card_rows
        for t in targets
    ]
)
columns = [
    ("r2_hgb", "Predictable from inputs\n(CV R², boosted trees)", (-0.1, 1.05)),
    ("lin_share", "Captured by linear mixing\n(R² linear / R² trees)", (0, 1.05)),
    ("n_eff", "Effective inputs of 20\n(from mean |SHAP|)", (0, 20)),
    ("nn_autocorr", "Smoothness (corr. with\nnearest composition)", (-0.1, 1.05)),
    ("within_share", "Variance at fixed input\n(noise / polymorphs)", (0, 1)),
]
fig, axs = plt.subplots(1, len(columns), figsize=(12.5, 6.6), sharey=True,
                        layout="constrained")  # fmt: skip
mp_rows = card["kind"].eq("mp").to_numpy()
for ax, (col, title, lim) in zip(axs, columns):
    vals, lo_, hi_ = (
        np.array([agg.loc[(r.key, r.target), (col, s)] for r in card.itertuples()])
        for s in ("mean", "min", "max")
    )
    band = vals[mp_rows & np.isfinite(vals)]
    ax.axvspan(band.min(), band.max(), color=orange, alpha=0.1, lw=0)
    for i, r in enumerate(card.itertuples()):
        if r.kind == "mpin":
            ax.plot([lo_[i], hi_[i]], [i, i], color=aqua, lw=2, solid_capstyle="round")
        ax.plot(vals[i], i, "o", ms=6, mew=1.5, **marker[r.kind])
    for b in card.index[card["key"].ne(card["key"].shift())][1:]:
        ax.axhline(b - 0.5, color=rule, lw=0.6)
    ax.set_xlim(*lim)
    ax.set_title(title, fontsize=7.5)
    ax.grid(axis="y", visible=False)
axs[0].set_yticks(range(len(card)), card["label"])
axs[0].invert_yaxis()
fig.legend(
    handles=[
        plt.Line2D([], [], ls="", marker="o", mew=1.5, **marker["cube"],
                   label="CrabNet surrogate, no constraint"),
        plt.Line2D([], [], ls="", marker="o", mew=1.5, **marker["comp"],
                   label=f"CrabNet surrogate, sum-to-one (≥{min_active} active)"),
        plt.Line2D([], [], color=aqua, marker="o", mew=1.5, **marker["mpin"],
                   label="CrabNet surrogate on MP alloy compositions"
                         " (mean, range of 3 mappings)"),
        plt.Line2D([], [], ls="", marker="o", mew=1.5, **marker["mp"],
                   label=f"Materials Project, ≥{min_active} elements,"
                         " lowest-energy polymorph"),
        plt.Rectangle((0, 0), 1, 1, color=orange, alpha=0.2,
                      label="range of the MP datasets"),
    ],
    loc="outside lower center",
    ncol=3,
)  # fmt: skip
fig.suptitle(
    "Quacks like a duck? Response-surface behavior under the composition constraint"
    " vs. Materials Project composition-property data",
    **suptitle,
)
fig.savefig(fig_dir / "constraint_duck_test_scorecard.png")

# %% figure 4: effect of each input (SHAP dependence and presence effects)
shap_rows = [
    (shap_surrogate[("sum-to-one", "mae")], x_comp[n_background:][:n_explain],
     imp_surrogate[("sum-to-one", "mae")], labels, "CrabNet, sum-to-one\nMAE (eV)", blue),
    (shap_surrogate[("sum-to-one", "runtime")], x_comp[n_background:][:n_explain],
     imp_surrogate[("sum-to-one", "runtime")], labels,
     "CrabNet, sum-to-one\nruntime (s)", blue),
    (curves[("MP alloys", "E_form")]["shap"], curves[("MP alloys", "E_form")]["x"],
     curves[("MP alloys", "E_form")]["imp"], datasets["MP alloys"]["features"],
     "MP alloys, formation\nenergy (eV/atom)", orange),
    (curves[("MP oxides", "E_gap")]["shap"], curves[("MP oxides", "E_gap")]["x"],
     curves[("MP oxides", "E_gap")]["imp"], datasets["MP oxides"]["features"],
     "MP oxides\nband gap (eV)", orange),
]  # fmt: skip
fig = plt.figure(figsize=(12, 11), layout="constrained")
top, bottom = fig.subfigures(2, 1, height_ratios=[len(shap_rows), 2.2])
axs = top.subplots(len(shap_rows), 5)
for row, (sv, x, imp, feats, label, c) in zip(axs, shap_rows):
    order = np.argsort(-imp)[:5]
    lim = np.abs(sv[:, order]).max() * 1.05
    for ax, j in zip(row, order):
        ax.scatter(x[:, j], sv[:, j], s=5, color=c, alpha=0.35, lw=0)
        ax.axhline(0, color=rule, lw=0.8)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"{feats[j]} ({imp[j]:.0%} of mean |SHAP|)", fontsize=7.5,
                     fontweight="normal")  # fmt: skip
        ax.set_xlabel("component value", labelpad=1)
    row[0].set_ylabel(f"{label}\nSHAP value", fontsize=7.5)
top.suptitle(
    "a  SHAP dependence of the five most important inputs (y-scale shared within a"
    " row; CrabNet rows use SHAP on the Space surrogate)",
    **suptitle,
)
axs = bottom.subplots(1, 3)
effects = [
    ("CrabNet sum-to-one", "mae", labels, blue, "CrabNet, sum-to-one · MAE"),
    ("CrabNet sum-to-one", "runtime", labels, blue, "CrabNet, sum-to-one · runtime"),
    ("MP alloys", "E_form", datasets["MP alloys"]["features"], orange,
     "MP alloys · formation energy"),
]  # fmt: skip
for ax, (key, t, feats, c, title) in zip(axs, effects):
    eff = curves[(key, t)]["effect"]
    order = np.argsort(-np.abs(np.nan_to_num(eff)))[:10]
    ax.barh(np.arange(10), eff[order], color=c, height=0.7)
    ax.set_yticks(np.arange(10), [feats[j] for j in order])
    ax.invert_yaxis()
    ax.axvline(0, color=rule, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean(y | present) − mean(y | absent), in SD of y")
    ax.set_title(title, fontweight="normal")
bottom.suptitle(
    "b  Presence effects: how much the output moves when a component is active"
    " (10 largest)",
    **suptitle,
)
fig.savefig(fig_dir / "constraint_duck_test_shap.png")
