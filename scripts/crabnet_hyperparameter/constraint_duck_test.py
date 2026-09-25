"""Duck test for the sum-to-one constraint on the CrabNet hyperparameter benchmark.

The formulation-style variant of the benchmark min-max scales each numerical
hyperparameter to [0, 1] and requires the scaled values to sum to one. This script
checks whether the published surrogate, restricted to that simplex, behaves like a
real composition-property problem. The reference problems are composition-property
datasets built from the Materials Project (MP). The comparison covers input
cross-correlations, sparsity, distance from the training data, SHAP feature
importances and effects, the share captured by linear mixing, smoothness, and
replicate/polymorph noise.

Inputs (not tracked in git):

- Zenodo 10.5281/zenodo.7694268: ``sobol_regression.csv`` saved to
  ``data/external/crabnet_hyperparameter`` and ``surrogate_models.pkl`` saved to
  ``models/crabnet_hyperparameter``
- MP summary build 2026.04.13 from the MP AWS Open Data bucket (no API key needed):
  ``s3://materialsproject-build/collections/summary/version=2026-04-13/`` saved as
  ``data/external/materials_project/summary_2026-04-13.parquet``

The surrogate was pickled with scikit-learn 1.0.1. Run with Python 3.10,
scikit-learn 1.0.2, numpy<1.24, pandas, pyarrow, shap 0.42.1, and matplotlib.
"""

# %% imports
import gc
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shap
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

rng = np.random.default_rng(42)
fig_dir = Path("reports/crabnet_hyperparameter_immi/figures")
n_samples = 20_000  # surrogate evaluations per point set
n_max = 5_000  # rows per dataset used for model fits, so R2 values are comparable
n_shap = 1_500

# %% CrabNet search space, in the order of the blinded x1..x20 on the HF Space
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
lo = np.array([b[0] for b in bounds.values()])
hi = np.array([b[1] for b in bounds.values()])
i_b1, i_b2, i_emb, i_pos = (
    hp.index(p) for p in ["betas1", "betas2", "emb_scaler", "pos_scaler"]
)

# Feature layout of the Zenodo surrogate (notebook 1.2 as of the 2023-03-02 upload):
# 21 numeric columns, one-hot categoricals, a constant hardware column, and the
# percentile rank for the noisy objectives.
rf_numeric = [
    "N", "alpha", "d_model", "dim_feedforward", "dropout", "emb_scaler", "eps",
    "epochs_step", "fudge", "heads", "k", "lr", "pe_resolution", "ple_resolution",
    "pos_scaler", "weight_decay", "batch_size", "out_hidden4", "betas1", "betas2",
    "train_frac",
]  # fmt: skip
rf_fixed = {  # highest fidelity and the default categoricals
    "train_frac": 1.0,
    "bias": 0,
    "criterion_RobustL1": 1,
    "criterion_RobustL2": 0,
    "elem_prop_magpie": 0,
    "elem_prop_mat2vec": 1,
    "elem_prop_onehot": 0,
    "hardware_2080ti": 1,
    "rank": 0.5,  # median of the heteroskedastic noise distribution
}
objectives = ["mae", "rmse", "runtime", "model_size"]

# PseudoCrabModerate (core.py): 5 hyperparameters picked by default_rng(50), the rest
# fixed at the defaults in matbench_metric_calculator
moderate_hp = list(np.random.default_rng(50).choice(hp + ["train_frac"], 5, False))
moderate_defaults = {
    "N": 3, "alpha": 0.5, "d_model": 512, "dim_feedforward": 2048, "dropout": 0.1,
    "emb_scaler": 1.0, "epochs_step": 10, "eps": 1e-6, "fudge": 0.02, "heads": 4,
    "k": 6, "lr": 0.001, "pe_resolution": 5000, "ple_resolution": 5000,
    "pos_scaler": 1.0, "weight_decay": 0.0, "batch_size": 32, "out_hidden4": 128,
    "betas1": 0.9, "betas2": 0.999,
}  # fmt: skip

# %% CrabNet Sobol data (real training runs)
sobol = pd.read_csv("data/external/crabnet_hyperparameter/sobol_regression.csv")
x_sobol = (sobol[hp].to_numpy() - lo) / (hi - lo)
group_cols = hp + ["train_frac", "bias", "criterion", "elem_prop"]
sobol["group"] = sobol[group_cols].round(6).groupby(group_cols).ngroup()
x_sobol_unique = x_sobol[~sobol["group"].duplicated().to_numpy()]
sum_sobol = x_sobol_unique.sum(1)
sum_sobol_moderate = x_sobol_unique[:, [hp.index(p) for p in moderate_hp]].sum(1)

# %% Materials Project composition-property datasets
mp_cols = [
    "material_id", "formula_pretty", "composition", "elements", "nsites",
    "formation_energy_per_atom", "energy_above_hull", "band_gap", "density",
    "total_magnetization", "bulk_modulus", "deprecated",
]  # fmt: skip
mp = pq.read_table(
    "data/external/materials_project/summary_2026-04-13.parquet", columns=mp_cols
).to_pandas()
mp = mp[~mp["deprecated"] & mp["formation_energy_per_atom"].notna()]
mp["magnetization"] = mp["total_magnetization"] / mp["nsites"]
mp["bulk_modulus"] = [
    np.nan if b is None else dict(b).get("vrh") for b in mp["bulk_modulus"]
]
mp.loc[~mp["bulk_modulus"].between(0, 1000), "bulk_modulus"] = np.nan  # bad fits
mp["element_set"] = mp["elements"].apply(frozenset)
mp_props = {
    "formation_energy_per_atom": "E_form",
    "energy_above_hull": "E_hull",
    "band_gap": "E_gap",
    "density": "density",
    "magnetization": "magnetization",
    "bulk_modulus": "K_VRH",
}
element_sets = {
    "MP alloys": "Al Co Cr Cu Fe Hf Mg Mn Mo Nb Ni Re Si Sn Ta Ti V W Zn Zr",
    "MP oxides": "Al Ba Bi Ca Co Cr Cu Fe Li Mg Mn Na Nb Ni O Sr Ti V W Zn",
    "MP mixed": "C Ca Co Cr Cu F Fe H Li Mg Mn N Na Ni O P S Si Ti V",
    "MP Li-Co-Mn-Ni-O": "Co Li Mn Ni O",
}
mp_targets = {
    "MP alloys": ["E_form", "E_hull", "density", "magnetization", "K_VRH"],
    "MP oxides": ["E_gap", "E_form", "E_hull", "density"],
    "MP mixed": ["E_form", "E_gap", "E_hull", "density"],
    "MP Li-Co-Mn-Ni-O": ["E_form", "E_hull", "E_gap", "density", "magnetization"],
}

datasets = {}
noise_rows = []
for name, els in element_sets.items():
    els = els.split()
    sub = mp[mp["element_set"].apply(lambda s: s <= frozenset(els))]
    if name == "MP oxides":
        sub = sub[sub["element_set"].apply(lambda s: "O" in s)]
    sub = sub.rename(columns=mp_props)
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
                "within_sd": np.sqrt(
                    (dev**2).sum() / (multi.sum() - (grp.size() > 1).sum())
                ),
                "n_groups": int((grp.size() > 1).sum()),
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

# CrabNet replicate noise from the real Sobol runs
for t in ["mae", "rmse", "runtime", "model_size"]:
    grp = sobol.groupby("group")[t]
    multi = grp.transform("size") > 1
    dev = sobol[t] - grp.transform("mean")
    noise_rows.append(
        {
            "dataset": "CrabNet Sobol runs",
            "target": t,
            "within_share": (dev[multi] ** 2).sum()
            / ((sobol[t][multi] - sobol[t][multi].mean()) ** 2).sum(),
            "within_sd": np.sqrt((dev[multi] ** 2).sum() / (multi.sum() - grp.ngroups)),
            "n_groups": int((grp.size() > 1).sum()),
        }
    )
noise = pd.DataFrame(noise_rows)

# %% point sets on which the surrogate is evaluated
x_cube = rng.random((n_samples, len(hp)))
flip = x_cube[:, i_emb] + x_cube[:, i_pos] > 1  # Sobol design: emb + pos <= 1
x_cube[flip, i_emb], x_cube[flip, i_pos] = (
    1 - x_cube[flip, i_pos],
    1 - x_cube[flip, i_emb],
)
x_simplex = rng.dirichlet(np.ones(len(hp)), n_samples)
for x in (x_cube, x_simplex):  # Sobol design: betas1 <= betas2
    x[:, [i_b1, i_b2]] = np.sort(x[:, [i_b1, i_b2]], axis=1)

# MP alloy compositions mapped onto the 20 hyperparameters (same inputs, other function)
x_alloys = datasets["MP alloys"]["X"]
perms = [rng.permutation(len(hp)) for _ in range(3)]
x_alloy_perms = []
for perm in perms:
    x = np.empty_like(x_alloys)
    x[:, perm] = x_alloys
    x_alloy_perms.append(x)

x_mod_cube = rng.random((n_samples, 5))
x_mod_simplex = rng.dirichlet(np.ones(5), n_samples)

raw_sets = {"cube": x_cube, "simplex": x_simplex}
raw_sets.update({f"alloys_perm{k}": x for k, x in enumerate(x_alloy_perms)})
raw = [pd.DataFrame(lo + x * (hi - lo), columns=hp) for x in raw_sets.values()]
for key, x in {"mod_cube": x_mod_cube, "mod_simplex": x_mod_simplex}.items():
    frame = pd.DataFrame([moderate_defaults] * len(x))
    for j, p in enumerate(moderate_hp):
        b = bounds[p]
        frame[p] = b[0] + x[:, j] * (b[1] - b[0])
        if isinstance(b[0], int):  # PseudoCrab rounds integer hyperparameters
            frame[p] = frame[p].round()
    raw.append(frame)
    raw_sets[key] = x
raw = pd.concat(raw, ignore_index=True).assign(**rf_fixed)
split = np.cumsum([len(x) for x in raw_sets.values()])[:-1]

# %% evaluate the published surrogate (about 9 GB in memory)
rf = joblib.load("models/crabnet_hyperparameter/surrogate_models.pkl")
pred = {}
for obj in objectives:
    cols = rf_numeric + list(rf_fixed)[1:-1] + ([] if obj == "model_size" else ["rank"])
    pred[obj] = rf[obj].predict(raw[cols].to_numpy())
pred = pd.DataFrame(pred)
pred_sets = dict(zip(raw_sets, np.split(pred, split)))

# spread across the 100 trees as a model-uncertainty proxy
spread = {}
cols = rf_numeric + list(rf_fixed)[1:]
for key in ["cube", "simplex", "alloys_perm0"]:
    start = ([0] + list(split))[list(raw_sets).index(key)]
    idx = start + rng.choice(len(raw_sets[key]), 2000, replace=False)
    trees = np.stack(
        [t.predict(raw.iloc[idx][cols].to_numpy()) for t in rf["mae"].estimators_]
    )
    spread[key] = trees.std(0)
del rf
gc.collect()

# %% support: distance from each point set to the nearest real Sobol run
nn = NearestNeighbors(n_neighbors=1).fit(x_sobol_unique)
nn_dist = {}
for key in ["cube", "simplex", "alloys_perm0"]:
    x = raw_sets[key][rng.choice(len(raw_sets[key]), 2000, replace=False)]
    nn_dist[key] = nn.kneighbors(x)[0][:, 0]

# %% assemble CrabNet datasets
datasets["CrabNet cube"] = {
    "X": x_cube, "Y": pred_sets["cube"].reset_index(drop=True),
    "features": hp, "family": "CrabNet",
}  # fmt: skip
datasets["CrabNet simplex"] = {
    "X": x_simplex, "Y": pred_sets["simplex"].reset_index(drop=True),
    "features": hp, "family": "CrabNet",
}  # fmt: skip
for k, perm in enumerate(perms):
    datasets[f"CrabNet on MP alloys {k}"] = {
        "X": x_alloys,  # identical inputs to "MP alloys"
        "Y": pred_sets[f"alloys_perm{k}"].reset_index(drop=True),
        "features": [
            f"{e}→{hp[perm[j]]}"
            for j, e in enumerate(element_sets["MP alloys"].split())
        ],
        "family": "CrabNet",
    }
datasets["CrabNet 5-D cube"] = {
    "X": x_mod_cube, "Y": pred_sets["mod_cube"].reset_index(drop=True),
    "features": moderate_hp, "family": "CrabNet",
}  # fmt: skip
datasets["CrabNet 5-D simplex"] = {
    "X": x_mod_simplex, "Y": pred_sets["mod_simplex"].reset_index(drop=True),
    "features": moderate_hp, "family": "CrabNet",
}  # fmt: skip


# %% landscape metrics for every dataset and target
kf = KFold(5, shuffle=True, random_state=0)
rows, curves = [], {}
for name, ds in datasets.items():
    for t in ds["Y"].columns:
        y_all = ds["Y"][t].to_numpy(float)
        ok = np.flatnonzero(np.isfinite(y_all))
        idx = rng.choice(ok, min(n_max, len(ok)), replace=False)
        X, y = ds["X"][idx], y_all[idx]
        hgb = HistGradientBoostingRegressor(max_iter=300, random_state=0)
        r2 = {"r2_hgb": r2_score(y, cross_val_predict(hgb, X, y, cv=kf))}
        for deg in (1, 2, 3):
            poly = make_pipeline(  # unscaled: rare mixture terms stay penalized
                PolynomialFeatures(deg, include_bias=False),
                VarianceThreshold(),
                RidgeCV(np.logspace(-6, 3, 19)),
            )
            r2[f"r2_poly{deg}"] = r2_score(y, cross_val_predict(poly, X, y, cv=kf))
        hgb.fit(X, y)
        s_idx = rng.choice(len(X), min(n_shap, len(X)), replace=False)
        sv = shap.TreeExplainer(hgb).shap_values(X[s_idx])
        imp = np.abs(sv).mean(0) / np.abs(sv).mean(0).sum()
        nbr = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X[s_idx])[1][:, 1]
        rows.append(
            {
                "dataset": name,
                "family": ds["family"],
                "target": t,
                "n": len(y),
                "d": X.shape[1],
                "y_sd": y.std(),
                **r2,
                "n_eff_frac": 1 / (imp**2).sum() / X.shape[1],
                "top_share": imp.max(),
                "nn_autocorr": np.corrcoef(y[s_idx], y[nbr])[0, 1],
                "top_features": ", ".join(
                    np.array(ds["features"])[np.argsort(-imp)[:3]]
                ),
            }
        )
        curves[(name, t)] = {"imp": imp, "shap": sv, "x": X[s_idx]}
metrics = pd.DataFrame(rows)
noise_share = noise.set_index(["dataset", "target"])["within_share"]
metrics["within_share"] = [
    noise_share.get(("CrabNet Sobol runs" if f == "CrabNet" else d, t))
    for d, f, t in metrics[["dataset", "family", "target"]].itertuples(index=False)
]
metrics.to_csv(fig_dir / "constraint_duck_test_metrics.csv", index=False)
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
domain = {  # surrogate point sets: color and label
    "cube": (light, "no constraint"),
    "simplex": (blue, "sum-to-one"),
    "alloys_perm0": (aqua, "on MP alloy compositions"),
}

# %% figure 1: inputs, data support, and what the constraint does to the surrogate
fig = plt.figure(figsize=(12, 9.6), layout="constrained")
top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.85])
heat = {
    "CrabNet cube": "CrabNet, no constraint",
    "CrabNet simplex": "CrabNet, sum-to-one",
    "MP alloys": "MP alloys (20 elements)",
    "MP oxides": "MP oxides (20 elements)",
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
    axs[0].get_xlabel() + "\n(±0.5 cells: emb+pos ≤ 1, betas1 ≤ betas2)", color=ink2
)
cbar = top.colorbar(im, ax=axs, shrink=0.8, label="Pearson r between two inputs")
cbar.outline.set_visible(False)
top.suptitle(
    "a  Input cross-correlations (20 × 20, diagonal blank)",
    x=0.01,
    ha="left",
    fontsize=8.5,
    fontweight="bold",
)

axs = bottom.subplots(2, 4)
ax = axs[0, 0]
closure = {
    "CrabNet cube": (light, "CrabNet, no constraint"),
    "CrabNet simplex": (blue, "CrabNet, sum-to-one"),
    "MP alloys": (orange, "MP alloys"),
    "MP oxides": (orange, "MP oxides"),
    "MP mixed": (orange, "MP mixed"),
    "CrabNet 5-D simplex": (blue, "CrabNet 5-D, sum-to-one"),
    "MP Li-Co-Mn-Ni-O": (orange, "MP Li-Co-Mn-Ni-O"),
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

mp20 = np.vstack([datasets[k]["X"] for k in ["MP alloys", "MP oxides", "MP mixed"]])
ax = axs[0, 1]
ax.hist(
    [(mp20 > 0).sum(1), (x_simplex > 0).sum(1)],
    bins=np.arange(0.5, 21.5),
    color=[orange, blue],
    label=["MP, three 20-element sets", "CrabNet, sum-to-one"],
    density=True,
    rwidth=0.9,
)
ax.set_xlabel("nonzero components per point (of 20)")
ax.set_ylabel("fraction of points")
ax.legend(loc="upper center")
ax.set_title("c  Sparsity")

ax = axs[0, 2]
for x, c, label in [(mp20, orange, "MP"), (x_simplex, blue, "CrabNet, sum-to-one")]:
    kw = dict(bins=np.linspace(0, 1, 41), density=True, color=c)
    ax.hist(x.max(1), histtype="stepfilled", alpha=0.12, **kw)
    ax.hist(x.max(1), histtype="step", lw=1.8, label=label, **kw)
ax.set_xlabel("largest component fraction")
ax.set_ylabel("density")
ax.legend()
ax.set_title("d  Dominant component")

ax = axs[0, 3]
kw = dict(bins=np.linspace(0, 14, 71))
ax.hist(sum_sobol, color=muted, label=f"20 inputs (min {sum_sobol.min():.1f})", **kw)
ax.hist(
    sum_sobol_moderate,
    color=rule,
    label=f"5 PseudoCrabModerate inputs\n({(sum_sobol_moderate <= 1).mean():.1%} ≤ 1)",
    **kw,
)
ax.axvline(1, color=blue, lw=2, label="sum-to-one constraint")
ax.set_xlabel("sum of min-max scaled hyperparameters")
ax.set_ylabel("unique Sobol settings")
ax.set_ylim(top=ax.get_ylim()[1] * 1.5)
ax.legend(loc="upper right")
ax.set_title("e  Where the training runs are")

for ax, values, xlabel, bins, title in [
    (
        axs[1, 0],
        nn_dist,
        "distance to nearest Sobol run (scaled units)",
        np.linspace(0.4, 1.7, 53),
        "f  Distance from the training runs",
    ),
    (
        axs[1, 1],
        spread,
        "MAE std. across the 100 trees (eV)",
        np.linspace(0, 0.2, 51),
        "g  Surrogate uncertainty",
    ),
]:
    for key, (c, label) in domain.items():
        ax.hist(values[key], bins=bins, histtype="stepfilled", color=c, alpha=0.12)
        ax.hist(values[key], bins=bins, histtype="step", color=c, lw=1.8, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("points")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.35)
    ax.legend(loc="upper right")
    ax.set_title(title)

ax = axs[1, 2]
ecdf = {"Sobol runs (all fidelities)": (sobol["mae"], muted)}
ecdf.update({v[1]: (pred_sets[k]["mae"], v[0]) for k, v in domain.items()})
for label, (v, c) in ecdf.items():
    v = np.sort(v)
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, lw=2, label=label)
ax.set_xlabel("MAE (eV); surrogate at median noise")
ax.set_ylabel("cumulative fraction")
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16))
ax.set_title("h  Objective values reached")

ax = axs[1, 3]
pairs = [
    ("mae", "rmse"),
    ("mae", "runtime"),
    ("mae", "model_size"),
    ("runtime", "model_size"),
]
short = {"mae": "MAE", "rmse": "RMSE", "runtime": "runtime", "model_size": "size"}
bars = {
    "Sobol runs (all fidelities)": (sobol[objectives], muted),
    "no constraint": (pred_sets["cube"], light),
    "sum-to-one": (pred_sets["simplex"], blue),
}
for j, (label, (frame, c)) in enumerate(bars.items()):
    C = frame.corr("spearman")
    ax.barh(
        np.arange(len(pairs)) + (j - 1) * 0.27,
        [C.loc[a, b] for a, b in pairs],
        height=0.25,
        color=c,
        label=label,
    )
ax.set_yticks(range(len(pairs)), [f"{short[a]} vs {short[b]}" for a, b in pairs])
ax.invert_yaxis()
ax.axvline(0, color=rule, lw=0.8)
ax.set_xlim(-0.7, 1.05)
ax.set_xlabel("Spearman correlation between objectives")
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=1)
ax.set_title("i  Objective trade-offs")
fig.suptitle(
    "Looks like a duck? Inputs, data support, and what the sum-to-one constraint does"
    " to the published surrogate",
    x=0.01,
    ha="left",
    fontsize=9.5,
    fontweight="bold",
)
fig.savefig(fig_dir / "constraint_duck_test_inputs.png")

# %% figure 2: response-surface scorecard
nice = {
    "mae": "MAE",
    "runtime": "runtime",
    "model_size": "model size",
    "E_form": "formation energy",
    "E_hull": "energy above hull",
    "E_gap": "band gap",
    "density": "density",
    "magnetization": "magnetization",
    "K_VRH": "bulk modulus",
}
card_rows = [
    ("CrabNet cube", "no constraint", ["mae", "runtime", "model_size"], "cube"),
    ("CrabNet simplex", "sum-to-one", ["mae", "runtime", "model_size"], "simplex"),
    ("CrabNet on MP alloys", "on MP alloy comps.", ["mae", "runtime", "model_size"],
     "mpin"),
    ("CrabNet 5-D cube", "5-D, no constraint", ["mae", "runtime"], "cube"),
    ("CrabNet 5-D simplex", "5-D, sum-to-one", ["mae", "runtime"], "simplex"),
    ("MP alloys", "MP alloys", mp_targets["MP alloys"], "mp"),
    ("MP oxides", "MP oxides", mp_targets["MP oxides"], "mp"),
    ("MP mixed", "MP mixed", mp_targets["MP mixed"], "mp"),
    ("MP Li-Co-Mn-Ni-O", "MP Li-Co-Mn-Ni-O", mp_targets["MP Li-Co-Mn-Ni-O"], "mp"),
]  # fmt: skip
marker = {
    "cube": dict(mfc="white", mec=blue),
    "simplex": dict(mfc=blue, mec=blue),
    "mpin": dict(mfc=aqua, mec=aqua),
    "mp": dict(mfc=orange, mec=orange),
}
best_r2 = metrics[["r2_poly1", "r2_poly2", "r2_poly3", "r2_hgb"]].max(axis=1)
m = metrics.assign(
    dataset=metrics["dataset"].str.replace(r" \d$", "", regex=True),
    lin_share=(metrics["r2_poly1"].clip(lower=0) / best_r2).where(best_r2 > 0.2),
)
cols = ["r2_hgb", "lin_share", "n_eff_frac", "nn_autocorr", "within_share"]
agg = m.groupby(["dataset", "target"])[cols + ["d"]].agg(["mean", "min", "max"])
card = pd.DataFrame(
    [
        {"key": key, "label": f"{label} · {nice[t]}", "kind": kind, "target": t}
        for key, label, targets, kind in card_rows
        for t in targets
    ]
)
columns = [
    ("r2_hgb", "Predictable from inputs\n(CV R², boosted trees)", (-0.1, 1.05)),
    ("lin_share", "Captured by linear mixing\n(R² linear / best R²)", (0, 1.05)),
    ("n_eff_frac", "Effective inputs / d\n(from mean |SHAP|)", (0, 1)),
    ("nn_autocorr", "Smoothness (corr. with\nnearest composition)", (-0.1, 1.05)),
    ("within_share", "Variance at fixed input\n(repeats / polymorphs)", (0, 1)),
]
fig, axs = plt.subplots(
    1, len(columns), figsize=(12.5, 8.4), sharey=True, layout="constrained"
)
mp20_rows = card["kind"].eq("mp") & ~card["key"].eq("MP Li-Co-Mn-Ni-O")
for ax, (col, title, lim) in zip(axs, columns):
    vals = np.array(
        [agg.loc[(r.key, r.target), (col, "mean")] for r in card.itertuples()]
    )
    lo_, hi_ = (
        np.array([agg.loc[(r.key, r.target), (col, s)] for r in card.itertuples()])
        for s in ("min", "max")
    )
    band = vals[mp20_rows.to_numpy() & np.isfinite(vals)]
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
        plt.Line2D([], [], ls="", marker="o", mew=1.5, **marker["simplex"],
                   label="CrabNet surrogate, sum-to-one (uniform on the simplex)"),
        plt.Line2D([], [], color=aqua, marker="o", mew=1.5, **marker["mpin"],
                   label="CrabNet surrogate on MP alloy compositions"
                         " (mean, range of 3 mappings)"),
        plt.Line2D([], [], ls="", marker="o", mew=1.5, **marker["mp"],
                   label="Materials Project, lowest-energy polymorph per composition"),
        plt.Rectangle((0, 0), 1, 1, color=orange, alpha=0.2,
                      label="range of the three 20-element MP datasets"),
    ],
    loc="outside lower center",
    ncol=3,
)  # fmt: skip
fig.suptitle(
    "Quacks like a duck? Response-surface behavior of the constrained benchmark vs."
    " Materials Project composition-property data",
    x=0.01,
    ha="left",
    fontsize=9.5,
    fontweight="bold",
)
fig.savefig(fig_dir / "constraint_duck_test_scorecard.png")

# %% figure 3: SHAP dependence of the most important inputs
shap_rows = [
    ("CrabNet simplex", "mae", "CrabNet, sum-to-one\nMAE (eV)", blue),
    ("CrabNet simplex", "runtime", "CrabNet, sum-to-one\nruntime (s)", blue),
    ("CrabNet on MP alloys 0", "mae", "CrabNet on MP alloy\ncompositions, MAE (eV)",
     aqua),
    ("MP alloys", "E_form", "MP alloys, formation\nenergy (eV/atom)", orange),
    ("MP alloys", "density", "MP alloys\ndensity (g/cm³)", orange),
    ("MP oxides", "E_gap", "MP oxides\nband gap (eV)", orange),
]  # fmt: skip
fig, axs = plt.subplots(len(shap_rows), 5, figsize=(12, 12.5), layout="constrained")
for row, (key, t, label, c) in zip(axs, shap_rows):
    cv = curves[(key, t)]
    order = np.argsort(-cv["imp"])[:5]
    lim = np.abs(cv["shap"][:, order]).max() * 1.05
    for ax, j in zip(row, order):
        ax.scatter(cv["x"][:, j], cv["shap"][:, j], s=5, color=c, alpha=0.35, lw=0)
        ax.axhline(0, color=rule, lw=0.8)
        ax.set_ylim(-lim, lim)
        ax.set_title(
            f"{datasets[key]['features'][j]} ({cv['imp'][j]:.0%} of mean |SHAP|)",
            fontsize=7.5,
            fontweight="normal",
        )
        ax.set_xlabel("component value", labelpad=1)
    row[0].set_ylabel(f"{label}\nSHAP value", fontsize=7.5)
fig.suptitle(
    "Effect on the model: SHAP dependence of the five most important inputs"
    " (one point per composition; y-scale shared within a row)",
    x=0.01,
    ha="left",
    fontsize=9.5,
    fontweight="bold",
)
fig.savefig(fig_dir / "constraint_duck_test_shap.png")
