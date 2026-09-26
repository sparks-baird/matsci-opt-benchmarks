"""What happens to the sum-to-one duck test if ``alpha`` is put on a log scale.

The HF Space surrogate (``models/surrogate_models_hgbr_opt.pkl``) is evaluated on
the same random compositions with ``alpha`` mapped from its scaled value x2 in five
ways: the current linear [0, 1], log [1e-3, 1], log [1e-2, 1], log [0.1, 1], and
linear [0.1, 1] as a control that raises the floor without the log. Every other
hyperparameter keeps its linear min-max scaling. Two designs are compared:

- sum-to-one: x1 + ... + x20 = 1 with 3 to 6 active inputs (uniform on their
  simplex) and inactive inputs at 0, i.e. at their lowest value
- box: x uniform on [0, 1]^20, as on the HF Space

Both designs keep the Space constraints x19 <= x20 and x6 + x15 <= 1. The
fidelity and categoricals are fixed at train_frac = 1, bias = False, RobustL1,
mat2vec, and the noise percentile at the median. SHAP values are interventional,
computed on the Space surrogate itself with a background drawn from the same
design and mapping.

Run with Python 3.12, scikit-learn==1.4.1.post1, numpy<2, pandas, shap, joblib,
matplotlib, and huggingface_hub.
"""

# %% imports
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from huggingface_hub import hf_hub_download
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rng = np.random.default_rng(0)
fig_dir = Path("reports/crabnet_hyperparameter_immi/figures")
n_samples = 20_000
n_background, n_explain = 100, 1_000

# %% HF Space surrogate and Sobol data
space = "AccelerationConsortium/crabnet-hyperparameter"
models = joblib.load(
    hf_hub_download(space, "models/surrogate_models_hgbr_opt.pkl", repo_type="space")
)
sobol = pd.read_csv(hf_hub_download(space, "sobol_regression.csv", repo_type="space"))

# x1..x20 in PARAM_BOUNDS order
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
i_alpha, i_emb, i_pos, i_b1, i_b2 = (
    hp.index(p) for p in ["alpha", "emb_scaler", "pos_scaler", "betas1", "betas2"]
)
fixed = {
    "train_frac": 1.0,
    "bias": 0,
    "use_RobustL1": 1,
    "elem_prop_magpie": 0,
    "elem_prop_mat2vec": 1,
    "elem_prop_onehot": 0,
    "mae_rank": 0.5,
    "rmse_rank": 0.5,
}
objectives = ["mae", "rmse"]
alpha_bin_edge = models["mae"]._bin_mapper.bin_thresholds_[
    list(models["mae"].feature_names_in_).index("alpha")
][0]

# (scale, floor): alpha = floor ** (1 - x) on a log scale, floor + (1 - floor) * x
# on a linear scale
mappings = {
    "linear [0, 1] (current)": ("linear", 0.0),
    "log [0.001, 1]": ("log", 1e-3),
    "log [0.01, 1]": ("log", 1e-2),
    "log [0.1, 1]": ("log", 0.1),
    "linear [0.1, 1]": ("linear", 0.1),
}

# %% designs (same random compositions for every mapping)
n_active = rng.integers(3, 7, n_samples)
x_simplex = np.zeros((n_samples, len(hp)))
for row, k in zip(x_simplex, n_active):
    row[rng.choice(len(hp), k, replace=False)] = rng.dirichlet(np.ones(k))
x_box = rng.random((n_samples, len(hp)))
flip = x_box[:, i_emb] + x_box[:, i_pos] > 1
x_box[flip, i_emb], x_box[flip, i_pos] = 1 - x_box[flip, i_pos], 1 - x_box[flip, i_emb]
for x in (x_simplex, x_box):
    x[:, [i_b1, i_b2]] = np.sort(x[:, [i_b1, i_b2]], axis=1)
designs = {"sum-to-one": x_simplex, "box": x_box}

# %% evaluate the surrogate for every design and mapping
frames, preds = {}, {}
for design, x in designs.items():
    for name, (scale, floor) in mappings.items():
        frame = pd.DataFrame(lo + x * (hi - lo), columns=hp).assign(**fixed)
        xa = x[:, i_alpha]
        frame["alpha"] = floor ** (1 - xa) if scale == "log" else floor + (1 - floor) * xa
        frames[design, name] = frame
        preds[design, name] = pd.DataFrame(
            {
                t: models[t].predict(frame[models[t].feature_names_in_])
                for t in objectives
            }
        ).assign(alpha=frame["alpha"], alpha_active=xa > 0)

# %% interventional SHAP on the surrogate, in parallel
keys = [(d, m, t) for d in designs for m in mappings for t in objectives]
jobs = []
for design, name, t in keys:
    cols = models[t].feature_names_in_
    frame = frames[design, name][cols].astype(float)
    explainer = shap.TreeExplainer(
        models[t], frame.iloc[:n_background], feature_perturbation="interventional"
    )
    jobs.append(
        delayed(explainer.shap_values)(
            frame.iloc[n_background : n_background + n_explain]
        )
    )
shap_values = dict(zip(keys, Parallel(n_jobs=-1)(jobs)))

# %% summary table
rows, shares = [], {}
for design, name, t in keys:
    cols = list(models[t].feature_names_in_)
    imp = pd.Series(np.abs(shap_values[design, name, t]).mean(0), index=cols)[hp]
    share = shares[design, name, t] = imp / imp.sum()
    p = preds[design, name]
    rows.append(
        {
            "design": design,
            "alpha_mapping": name,
            "objective": t,
            "alpha_share": share["alpha"],
            "n_eff": 1 / (share**2).sum(),
            "top3": ", ".join(
                f"{k} {v:.0%}" for k, v in share.nlargest(3).items()
            ),
            "median": p[t].median(),
            "q10": p[t].quantile(0.1),
            "best": p[t].min(),
            "median_alpha_active": p.loc[p.alpha_active, t].median(),
            "median_alpha_inactive": p.loc[~p.alpha_active, t].median(),
            "frac_alpha_below_bin_edge": (p["alpha"] < alpha_bin_edge).mean(),
            "frac_alpha_below_0.02": (p["alpha"] < 0.02).mean(),
        }
    )
summary = pd.DataFrame(rows)
summary.to_csv(fig_dir / "alpha_log_scale_metrics.csv", index=False)
print(summary.round(3).to_string())

# %% figure
blue, orange, aqua, grey = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
ink, muted, band = "#0b0b0b", "#898781", "#f0efec"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": muted,
        "axes.labelcolor": ink,
        "axes.titlesize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": muted,
        "ytick.color": muted,
        "xtick.labelcolor": ink,
        "ytick.labelcolor": ink,
    }
)
plateau = 0.02  # median MAE of the real runs is above 0.85 eV below this alpha
names = list(mappings)
alpha_ticks = ([0, 1e-3, 1e-2, 1e-1, 1], ["0", "0.001", "0.01", "0.1", "1"])

# repeat-averaged Sobol runs, median and IQR of MAE per alpha bin
sets = (
    sobol.groupby(hp + ["train_frac", "bias", "criterion", "elem_prop"])["mae"]
    .mean()
    .reset_index()
)
edges = np.r_[0, np.logspace(-3, 0, 13)]
curve = (
    sets.groupby(pd.cut(sets["alpha"], edges, include_lowest=True), observed=True)
    ["mae"].quantile([0.25, 0.5, 0.75]).unstack()
)  # fmt: skip
centers = np.r_[5e-4, np.sqrt(edges[1:-1] * edges[2:])]

fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.2), layout="constrained", sharex="col")
ax = axs[0, 0]
ax.axvspan(0, plateau, color=band, lw=0)
ax.fill_between(centers, curve[0.25], curve[0.75], color=blue, alpha=0.18, lw=0)
ax.plot(centers, curve[0.5], color=blue, lw=2, marker="o", ms=4)
ax.axvline(alpha_bin_edge, color=muted, lw=0.8)
ax.text(0.0003, 0.3, "model barely\ntrains", color=ink, va="bottom")
ax.text(
    alpha_bin_edge * 1.12, 1.18, "surrogate's first split\non α (0.005)", color=ink,
    va="top",
)  # fmt: skip
ax.set_ylim(0.25, 1.25)
ax.set_ylabel("MAE of real runs (eV)")
ax.set_title("(a) Sobol runs: median and IQR of MAE per α bin", loc="left")

ax = axs[1, 0]
ax.axvspan(0, plateau, color=band, lw=0)
for y, name in enumerate(names):
    p = preds["sum-to-one", name]
    active = p.loc[p.alpha_active, "alpha"]
    ax.plot(active.quantile([0.1, 0.9]), [y, y], color=blue, lw=2, alpha=0.5)
    ax.plot(active.median(), y, "o", color=blue, ms=8, mec="white", mew=1.5)
    ax.plot(
        p.loc[~p.alpha_active, "alpha"].iloc[0], y, "o", mfc="white", mec=blue,
        mew=2, ms=8, clip_on=False,
    )  # fmt: skip
    ax.annotate(
        f"{(p['alpha'] < plateau).mean():.0%}", (1, y), xytext=(8, 0),
        xycoords=("axes fraction", "data"), textcoords="offset points", va="center",
    )  # fmt: skip
ax.annotate(
    "share with\nα < 0.02", (1, -0.75), xytext=(8, 0), xycoords=("axes fraction", "data"),
    textcoords="offset points", va="center", color=muted,
)  # fmt: skip
ax.set_xscale("symlog", linthresh=1e-3, linscale=0.3)
ax.set_xlim(0, 1)
ax.set_xticks(*alpha_ticks)
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_yticks(range(len(names)), names)
ax.set_ylim(len(names) - 0.5, -0.9)
ax.tick_params(axis="y", length=0, pad=8)
ax.set_xlabel("α (linear below 0.001, log above)")
ax.set_title(
    "(b) Sum-to-one: α when inactive (○) and active (●)",
    loc="left",
)

feats = ["alpha", "emb_scaler", "epochs_step"]
for ax, design, title in zip(
    axs[:, 1],
    designs,
    ["(c) SHAP share of MAE, sum-to-one (3 to 6 active)", "(d) SHAP share of MAE, box"],
):
    for y, name in enumerate(names):
        s = shares[design, name, "mae"]
        left = 0.0
        for v, c, label in zip(
            [*s[feats], 1 - s[feats].sum()], [blue, orange, aqua, grey],
            [*feats, "other 17"],
        ):  # fmt: skip
            ax.barh(
                y, v, left=left, height=0.62, color=c, edgecolor="white", lw=1.5,
                label=label if y == 0 else None,
            )  # fmt: skip
            left += v
        n_eff = summary.set_index(["design", "alpha_mapping", "objective"]).loc[
            (design, name, "mae"), "n_eff"
        ]
        ax.annotate(
            f"{n_eff:.1f}", (1, y), xytext=(8, 0), xycoords=("axes fraction", "data"),
            textcoords="offset points", va="center",
        )  # fmt: skip
    ax.annotate(
        "effective\ninputs", (1, -0.75), xytext=(8, 0), color=muted, va="center",
        xycoords=("axes fraction", "data"), textcoords="offset points",
    )  # fmt: skip
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(len(names) - 0.5, -0.9)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title(title, loc="left")
axs[1, 1].set_xlabel("share of mean |SHAP| over x1 to x20")
fig.legend(
    *axs[0, 1].get_legend_handles_labels(), loc="outside upper right", ncols=4,
    frameon=False, handlelength=1.2, columnspacing=1.2,
)  # fmt: skip
fig.savefig(fig_dir / "alpha_log_scale.png", dpi=200, facecolor="white")
