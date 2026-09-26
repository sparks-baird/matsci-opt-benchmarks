"""What happens to the sum-to-one duck test if alpha, and other inputs whose lowest
value is harmful, are inverted so that an inactive input sits at the good end.

In the sum-to-one design (x1 + ... + x20 = 1, 3 to 6 active inputs, inactive inputs
at 0), an inactive input takes the lowest value of its hyperparameter. For alpha (x2)
that is alpha = 0, where CrabNet barely trains. An inverted input counts down from
the top of its range, so x = 0 gives the highest value:

- neg_alpha: alpha = 1 - x2. alpha = 1 is the plain inner optimizer (Lookahead with
  no damping).
- neg_emb_scaler: emb_scaler = 1 - x6 - x15, i.e. x6 becomes CrabNet's third scaler
  (pos_scaler_log = 1 - emb_scaler - pos_scaler). With x15 = 0 this is 1 - x6, and
  the Space constraint x6 + x15 <= 1 keeps its form, so it holds on the simplex.
- neg_epochs_step: epochs_step = 20 - 15 x7, the most epochs when inactive.

The candidates come from a screen of all 20 inputs: the Space surrogate's partial
dependence at x = 0 and x = 1 over the box, and the median MAE of the real Sobol
runs (repeat averaged, alpha > 0.1 except for alpha itself) in the lowest and
highest tenth of each range.

Setup as in alpha_log_scale.py: HF Space surrogate, train_frac = 1, bias = False,
RobustL1, mat2vec, median noise percentile, x19 <= x20 kept by sorting, the same
random compositions for every orientation, and interventional SHAP computed on the
surrogate with a background drawn from the same design. The MP range of effective
inputs is read from constraint_duck_test_metrics.csv.

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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

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
i_emb, i_pos, i_b1, i_b2 = (
    hp.index(p) for p in ["emb_scaler", "pos_scaler", "betas1", "betas2"]
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
    "runtime_rank": 0.5,
}
objectives = ["mae", "rmse", "runtime", "model_size"]
# each orientation adds to the previous one
orientations = {
    "current": [],
    "neg_alpha": ["alpha"],
    "+ neg_emb_scaler": ["alpha", "emb_scaler"],
    "+ neg_epochs_step": ["alpha", "emb_scaler", "epochs_step"],
}


def to_frame(x, inverted=()):
    """Surrogate inputs for scaled points x, with the inverted inputs counting down."""
    u = x.copy()
    for p in inverted:
        i = hp.index(p)
        u[:, i] = 1 - x[:, i] - (x[:, i_pos] if p == "emb_scaler" else 0)
    return pd.DataFrame(lo + u * (hi - lo), columns=hp).assign(**fixed)


def predict(frame):
    return pd.DataFrame(
        {t: models[t].predict(frame[models[t].feature_names_in_]) for t in objectives}
    )


# %% designs (same random compositions as alpha_log_scale.py)
n_active = rng.integers(3, 7, n_samples)
x_simplex = np.zeros((n_samples, len(hp)))
for row, k in zip(x_simplex, n_active):
    row[rng.choice(len(hp), k, replace=False)] = rng.dirichlet(np.ones(k))
x_box = rng.random((n_samples, len(hp)))
flip = x_box[:, i_emb] + x_box[:, i_pos] > 1
x_box[flip, i_emb], x_box[flip, i_pos] = 1 - x_box[flip, i_pos], 1 - x_box[flip, i_emb]
for x in (x_simplex, x_box):
    x[:, [i_b1, i_b2]] = np.sort(x[:, [i_b1, i_b2]], axis=1)


# %% screen: each input at its lowest vs its highest value
def set_input(x, i, v):
    """Set input i to v for every point, adjusting partners to keep the constraints."""
    x = x.copy()
    x[:, i] = v
    if i in (i_emb, i_pos):  # x6 + x15 <= 1
        j = i_pos if i == i_emb else i_emb
        x[:, j] = np.minimum(x[:, j], 1 - v)
    if i == i_b1:  # x19 <= x20
        x[:, i_b2] = np.maximum(x[:, i_b2], v)
    if i == i_b2:
        x[:, i_b1] = np.minimum(x[:, i_b1], v)
    return x


x_pd = x_box[:3_000]
screen = {}
for i, p in enumerate(hp):
    at_lo, at_hi = (predict(to_frame(set_input(x_pd, i, v))).mean() for v in (0, 1))
    screen[p] = at_lo / at_hi - 1
screen = pd.DataFrame(screen).T

sets = (
    sobol.groupby(hp + ["train_frac", "bias", "criterion", "elem_prop"])["mae"]
    .mean()
    .reset_index()
)
x_sets = (sets[hp] - lo) / (hi - lo)
for p in hp:
    keep = (sets["alpha"] > 0.1) | (p == "alpha")
    low = sets.loc[keep & (x_sets[p] < 0.1), "mae"].median()
    high = sets.loc[keep & (x_sets[p] > 0.9), "mae"].median()
    screen.loc[p, "mae_runs"] = low / high - 1
screen.index.name = "input"
screen.to_csv(fig_dir / "invert_inputs_screen.csv")
print(screen.round(3).to_string())

# %% evaluate the surrogate for the box and every orientation on the simplex
frames = {"box, no constraint": to_frame(x_box)} | {
    name: to_frame(x_simplex, inverted) for name, inverted in orientations.items()
}
preds = {name: predict(frame) for name, frame in frames.items()}

# %% interventional SHAP on the surrogate, in parallel
keys = [(name, t) for name in frames for t in objectives]
jobs = []
for name, t in keys:
    frame = frames[name][models[t].feature_names_in_].astype(float)
    explainer = shap.TreeExplainer(
        models[t], frame.iloc[:n_background], feature_perturbation="interventional"
    )
    jobs.append(
        delayed(explainer.shap_values)(
            frame.iloc[n_background : n_background + n_explain], check_additivity=False
        )
    )
shap_values = dict(zip(keys, Parallel(n_jobs=-1)(jobs)))

# %% summary table
rows, shares = [], {}
for name, t in keys:
    cols = list(models[t].feature_names_in_)
    imp = pd.Series(np.abs(shap_values[name, t]).mean(0), index=cols)[hp]
    share = shares[name, t] = imp / imp.sum()
    p = preds[name]
    rows.append(
        {
            "orientation": name,
            "objective": t,
            "n_eff": 1 / (share**2).sum(),
            "top3": ", ".join(f"{k} {v:.0%}" for k, v in share.nlargest(3).items()),
            "median": p[t].median(),
            "q10": p[t].quantile(0.1),
            "q90": p[t].quantile(0.9),
            "best": p[t].min(),
            "frac_mae_below_0.5": (p["mae"] < 0.5).mean(),
            "spearman_mae_runtime": p["mae"].corr(p["runtime"], method="spearman"),
            "spearman_mae_model_size": p["mae"].corr(p["model_size"], method="spearman"),
        }
    )
summary = pd.DataFrame(rows)
summary.to_csv(fig_dir / "invert_inputs_metrics.csv", index=False)
print(summary.round(3).to_string())

# how often each input is active in the best 1% of compositions (by predicted MAE)
active_in_best = pd.DataFrame(
    {
        name: (x_simplex[preds[name]["mae"].nsmallest(n_samples // 100).index] > 0).mean(0)
        for name in orientations
    },
    index=hp,
)
print(active_in_best.round(2).to_string())

# %% figure
blue, orange, aqua, yellow, grey = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#c3c2b7"
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
inverted_all = orientations["+ neg_epochs_step"]
names = list(orientations) + ["box, no constraint"]
ypos = [0, 1, 2, 3, 4.4]  # gap before the reference row
mp_neff = pd.read_csv(fig_dir / "constraint_duck_test_metrics.csv").query(
    "family == 'MP'"
)["n_eff"]

fig = plt.figure(figsize=(12.5, 7.8), layout="constrained")
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35])
ax = fig.add_subplot(gs[:, 0])
cols = ["mae", "rmse", "runtime", "model_size", "mae_runs"]
vals = screen[cols].to_numpy() * 100
cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", band, "#e34948"])
ax.imshow(
    vals, cmap=cmap, norm=TwoSlopeNorm(0, -100, 100), aspect="auto",
    interpolation="none",
)  # fmt: skip
for (r, c), v in np.ndenumerate(vals):
    if abs(v) >= 5:
        ax.text(
            c, r, f"{v:+.0f}%", ha="center", va="center", fontsize=8.5,
            color="white" if abs(v) > 60 else ink,
        )  # fmt: skip
ax.set_xticks(
    range(len(cols)), ["MAE", "RMSE", "runtime", "model\nsize", "MAE,\nreal runs"]
)
ax.xaxis.tick_top()
ax.set_yticks(range(len(hp)), [f"x{i + 1} {p}" for i, p in enumerate(hp)])
for label in ax.get_yticklabels():
    if label.get_text().split()[1] in inverted_all:
        label.set_fontweight("bold")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.axvline(3.5, color="white", lw=4)
ax.set_title(
    "(a) Change at the lowest value vs the highest (surrogate\n"
    "over the box; red = lowest is worse; bold = inverted in b, c)",
    loc="left",
)

ax = fig.add_subplot(gs[0, 1])
top = []
for name in orientations:
    top += [k for k in shares[name, "mae"].nlargest(2).index if k not in top]
feats = top[:4]
for y, name in zip(ypos, names):
    s = shares[name, "mae"]
    left = 0.0
    for v, c, label in zip(
        [*s[feats], 1 - s[feats].sum()], [blue, orange, aqua, yellow][: len(feats)]
        + [grey], [*feats, f"other {20 - len(feats)}"],
    ):  # fmt: skip
        ax.barh(
            y, v, left=left, height=0.62, color=c, edgecolor="white", lw=1.5,
            label=label if y == 0 else None,
        )  # fmt: skip
        left += v
    n_eff = summary.set_index(["orientation", "objective"]).loc[(name, "mae"), "n_eff"]
    ax.annotate(
        f"{n_eff:.1f}", (1, y), xytext=(8, 0), xycoords=("axes fraction", "data"),
        textcoords="offset points", va="center",
    )  # fmt: skip
ax.annotate(
    "effective\ninputs", (1, -0.85), xytext=(8, 0), color=muted, va="center",
    xycoords=("axes fraction", "data"), textcoords="offset points",
)  # fmt: skip
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
ax.set_yticks(ypos, names)
ax.set_ylim(ypos[-1] + 0.5, -1)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlabel("share of mean |SHAP| over x1 to x20")
ax.legend(
    loc="lower left", bbox_to_anchor=(0, 1.02), ncols=len(feats) + 1, frameon=False,
    handlelength=1.2, columnspacing=1.2, borderaxespad=0,
)  # fmt: skip
ax.set_title(
    f"(b) SHAP share of MAE, sum-to-one (MP: {mp_neff.min():.1f} to "
    f"{mp_neff.max():.1f} effective inputs)",
    loc="left", pad=24,
)  # fmt: skip

ax = fig.add_subplot(gs[1, 1])
for y, name in zip(ypos, names):
    p = preds[name]
    ax.plot(p["mae"].quantile([0.1, 0.9]), [y, y], color=blue, lw=2, alpha=0.5)
    ax.plot(p["mae"].median(), y, "o", color=blue, ms=8, mec="white", mew=1.5)
    ax.plot(p["mae"].min(), y, "D", mfc="white", mec=blue, mew=2, ms=7)
    ax.annotate(
        f"{p['runtime'].median():.0f} s   {p['model_size'].median() / 1e6:.1f} M",
        (1, y), xytext=(8, 0), xycoords=("axes fraction", "data"),
        textcoords="offset points", va="center",
    )  # fmt: skip
ax.annotate(
    "median runtime,\nmodel size", (1, -0.85), xytext=(8, 0), color=muted,
    va="center", xycoords=("axes fraction", "data"), textcoords="offset points",
)  # fmt: skip
ax.set_yticks(ypos, names)
ax.set_ylim(ypos[-1] + 0.5, -1)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("predicted MAE (eV): best ◇, median ●, 10th to 90th percentile")
ax.set_title("(c) Predicted MAE on the same compositions", loc="left")
fig.savefig(fig_dir / "invert_inputs.png", dpi=200, facecolor="white")
