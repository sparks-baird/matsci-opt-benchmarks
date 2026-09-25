"""Noise-model ablation for the CrabNet benchmark surrogate.

Compares three predictive distributions for a new training run at an unseen
hyperparameter set, all built from HistGradientBoostingRegressor models:

* M1, deposited approach: one model on individual runs with the within-group
  percentile rank as an extra input; predictive ensemble = predictions at 21
  evenly spaced ranks (midpoints of (0, 1), as the released wrapper samples
  rank ~ U(0, 1)).
* M2, homoskedastic Gaussian: mean model on group means, constant sigma equal
  to the pooled within-group standard deviation.
* M3, heteroskedastic Gaussian: same mean model plus a model for the per-group
  log standard deviation.

One 80/20 split by unique hyperparameter set; groups with at least 3
successful runs; scored on held-out individual runs.
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

REPO = Path(__file__).resolve().parents[2]
FIG_DIR = REPO / "reports" / "crabnet_hyperparameter_immi" / "figures"
OUT_DIR = REPO / "reports" / "crabnet_hyperparameter_immi" / "revision" / "analysis"
CSV = os.environ.get("CRABNET_DATA", "/tmp/crabnet_data/sobol_regression.csv")

NUMERIC = [
    "N", "alpha", "d_model", "dim_feedforward", "dropout", "emb_scaler", "eps",
    "epochs_step", "fudge", "heads", "k", "lr", "pe_resolution", "ple_resolution",
    "pos_scaler", "weight_decay", "batch_size", "out_hidden4", "betas1", "betas2",
    "train_frac",
]  # fmt: skip
HP_COLUMNS = NUMERIC + ["bias", "criterion", "elem_prop"]
TARGETS = ["mae", "runtime"]
N_GRID = 21
SEED = 0

df = pd.read_csv(CSV)
df = df[df.train_frac >= 0.01]  # drop the 16 debug rows with train_frac = 0.003
df["group"] = df.groupby(HP_COLUMNS).ngroup()
df["n_runs"] = df.groupby("group")["mae"].transform("size")
df = df[df.n_runs >= 3].reset_index(drop=True)
X_all = pd.get_dummies(df[HP_COLUMNS], columns=["criterion", "elem_prop"], dtype=float)
X_all["bias"] = X_all["bias"].astype(float)

train_idx, test_idx = next(
    GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(
        X_all, groups=df["group"]
    )
)
is_train = np.zeros(len(df), bool)
is_train[train_idx] = True


def hgb():
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.1, random_state=SEED)


def gaussian_scores(y, mu, sd):
    z = (y - mu) / sd
    crps = sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    out = {"crps": crps, "lpd": stats.norm.logpdf(y, mu, sd)}
    for level in (0.8, 0.9):
        half = stats.norm.ppf(0.5 + level / 2) * sd
        out[f"cover{int(level * 100)}"] = np.abs(y - mu) <= half
        out[f"width{int(level * 100)}"] = 2 * half
    return out


def ensemble_scores(y, ens):
    """y: (n,), ens: (n, M) ensemble members for each observation."""
    m = ens.shape[1]
    term1 = np.abs(ens - y[:, None]).mean(1)
    srt = np.sort(ens, 1)
    # mean |X - X'| over all member pairs via the sorted-sample identity
    weights = 2 * np.arange(1, m + 1) - m - 1
    term2 = (srt * weights).sum(1) * 2 / m**2
    out = {"crps": term1 - 0.5 * term2}
    h = ens.std(1, ddof=1) * m ** (-1 / 5)  # scipy gaussian_kde default (Scott)
    z = (y[:, None] - ens) / h[:, None]
    out["lpd"] = logsumexp(-0.5 * z**2, axis=1) - np.log(m * h * np.sqrt(2 * np.pi))
    for level in (0.8, 0.9):
        lo, hi = np.quantile(ens, [0.5 - level / 2, 0.5 + level / 2], axis=1)
        out[f"cover{int(level * 100)}"] = (y >= lo) & (y <= hi)
        out[f"width{int(level * 100)}"] = hi - lo
    return out


summary = {
    "design": {
        "split": "one GroupShuffleSplit, 80/20 by unique hyperparameter set, seed 0",
        "learner": "sklearn HistGradientBoostingRegressor(max_iter=500, lr=0.1)",
        "features": "21 numeric hyperparameters (incl. train_frac), bias, one-hot criterion and elem_prop",
        "groups_used": "sets with at least 3 successful runs, debug rows (train_frac 0.003) removed",
        "m1_rank_grid": f"{N_GRID} midpoints of (0, 1); variant m1_support uses {N_GRID} points on [0.2, 1]",
        "m3_sd_model": "HGB on log of per-group SD (ddof=1), rescaled so the mean predicted variance equals the pooled variance on training groups",
        "lpd_m1": "Gaussian KDE over the 21 members, Scott bandwidth sd * 21^(-1/5)",
        "n_train_groups": int(df.loc[is_train, "group"].nunique()),
        "n_test_groups": int(df.loc[~is_train, "group"].nunique()),
        "n_train_runs": int(is_train.sum()),
        "n_test_runs": int((~is_train).sum()),
    }
}
rng = np.random.default_rng(SEED)
fig_data = {}
for target in TARGETS:
    y = df[target].to_numpy()
    rank = df[f"{target}_rank"].to_numpy()
    grp = df.groupby("group")[target]
    g_mean, g_sd, g_n = grp.mean(), grp.std(ddof=1), grp.size()
    g_train = df.loc[is_train, "group"].unique()
    g_test = df.loc[~is_train, "group"].unique()
    Xg = X_all.groupby(df["group"]).first()

    # heteroskedasticity evidence (all groups with >= 3 runs)
    resid = (y - df["group"].map(g_mean)) / df["group"].map(g_sd)
    sd_q = g_sd.quantile([0.05, 0.5, 0.95])
    hetero = {
        "per_group_sd_p5": sd_q[0.05],
        "per_group_sd_median": sd_q[0.5],
        "per_group_sd_p95": sd_q[0.95],
        "per_group_sd_p95_over_p5": sd_q[0.95] / sd_q[0.05],
        "spearman_sd_vs_mean": stats.spearmanr(g_sd, g_mean).statistic,
        "pooled_skewness_standardized_residuals": stats.skew(resid[np.isfinite(resid)]),
        "n_groups_zero_sd": int((g_sd == 0).sum()),
    }

    # M1: runs + rank
    X_rank = X_all.assign(rank=rank)
    m1 = hgb().fit(X_rank[is_train], y[is_train])
    X_test = X_all[~is_train]
    y_test = y[~is_train]
    grids = {
        "m1": (np.arange(N_GRID) + 0.5) / N_GRID,
        "m1_support": np.linspace(0.2, 1.0, N_GRID),
    }
    scores = {}
    for name, grid in grids.items():
        ens_g = np.column_stack([m1.predict(Xg.loc[g_test].assign(rank=r)) for r in grid])
        ens_g = pd.DataFrame(ens_g, index=g_test)
        ens = ens_g.loc[df.loc[~is_train, "group"]].to_numpy()
        scores[name] = ensemble_scores(y_test, ens)
        if name == "m1":
            m1_ens, m1_ens_g = ens, ens_g

    # M2 and M3: shared mean model on group means
    mean_model = hgb().fit(Xg.loc[g_train], g_mean.loc[g_train])
    mu_g = pd.Series(mean_model.predict(Xg.loc[g_test]), index=g_test)
    mu = mu_g.loc[df.loc[~is_train, "group"]].to_numpy()
    dof = g_n.loc[g_train] - 1
    pooled_sd = np.sqrt((dof * g_sd.loc[g_train] ** 2).sum() / dof.sum())
    scores["m2"] = gaussian_scores(y_test, mu, np.full_like(mu, pooled_sd))

    pos = g_sd.loc[g_train] > 0
    sd_model = hgb().fit(Xg.loc[g_train][pos], np.log(g_sd.loc[g_train][pos]))
    sd_train = np.exp(sd_model.predict(Xg.loc[g_train]))
    scale = np.sqrt((dof * g_sd.loc[g_train] ** 2).sum() / (dof * sd_train**2).sum())
    sd_g = pd.Series(scale * np.exp(sd_model.predict(Xg.loc[g_test])), index=g_test)
    sd = sd_g.loc[df.loc[~is_train, "group"]].to_numpy()
    scores["m3"] = gaussian_scores(y_test, mu, sd)
    # M1 spread shape around the shared mean model, to separate shape from accuracy
    scores["m1_recentered"] = ensemble_scores(y_test, m1_ens - m1_ens.mean(1, keepdims=True) + mu[:, None])
    obs_sd_test = g_sd.loc[g_test]
    hetero["m3_test_spearman_pred_vs_obs_group_sd"] = stats.spearmanr(sd_g, obs_sd_test).statistic

    # summaries and paired bootstrap over test groups
    test_groups = df.loc[~is_train, "group"].to_numpy()
    codes, uniq = pd.factorize(test_groups)
    res = {}
    crps_by_group = {}
    for name, s in scores.items():
        res[name] = {
            "crps_mean": s["crps"].mean(),
            "lpd_mean": s["lpd"].mean(),
            "lpd_median": np.median(s["lpd"]),
            "n_lpd_nonfinite": int((~np.isfinite(s["lpd"])).sum()),
            "coverage80": s["cover80"].mean(),
            "coverage90": s["cover90"].mean(),
            "width80_mean": s["width80"].mean(),
            "width90_mean": s["width90"].mean(),
        }
        crps_by_group[name] = np.bincount(codes, s["crps"]), np.bincount(codes)
    boot = rng.integers(0, len(uniq), (1000, len(uniq)))
    for a, b in [("m1", "m2"), ("m1", "m3"), ("m3", "m2")]:
        (sa, n), (sb, _) = crps_by_group[a], crps_by_group[b]
        diff = (sa[boot].sum(1) - sb[boot].sum(1)) / n[boot].sum(1)
        res[f"crps_diff_{a}_minus_{b}"] = {
            "mean": (sa.sum() - sb.sum()) / n.sum(),
            "ci95": np.quantile(diff, [0.025, 0.975]).tolist(),
        }
    res["pooled_sd"] = pooled_sd
    res["m1_ensemble_mean_rmse_vs_test_group_means"] = np.sqrt(
        ((m1_ens_g.mean(axis=1) - g_mean.loc[g_test]) ** 2).mean()
    )
    res["m1_ensemble_sd_median"] = m1_ens_g.std(axis=1, ddof=1).median()
    res["m3_predicted_sd_median"] = sd_g.median()
    res["mean_model_rmse_vs_test_group_means"] = np.sqrt(
        ((mu_g - g_mean.loc[g_test]) ** 2).mean()
    )
    summary[target] = {"heteroskedasticity": hetero, "scores": res}
    fig_data[target] = {"g_sd": g_sd, "g_mean": g_mean, "pooled_sd": pooled_sd, "res": res}

(OUT_DIR / "noise_ablation_summary.json").write_text(json.dumps(summary, indent=2, default=float))

# ---------------------------------------------------------------- figure
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)
models = {"m1": "M1 rank ensemble", "m2": "M2 constant Gaussian", "m3": "M3 hetero. Gaussian"}
colors = {"m1": "#2a78d6", "m2": "#eb6834", "m3": "#1baf7a"}
fig, axes = plt.subplots(2, 2, figsize=(17 / 2.54, 10 / 2.54), layout="constrained")
ax_a, ax_b, ax_c, ax_d = axes.ravel()
width = 0.25
for i, m in enumerate(models):
    rel = [fig_data[t]["res"][m]["crps_mean"] / fig_data[t]["res"]["m2"]["crps_mean"] for t in TARGETS]
    ax_a.bar(np.arange(len(TARGETS)) + (i - 1) * width, rel, width, color=colors[m], label=models[m])
    for j, t in enumerate(TARGETS):
        r = fig_data[t]["res"][m]
        ax_b.plot([80, 90][0:2], [r["coverage80"] * 100, r["coverage90"] * 100],
                  marker="os"[j], color=colors[m], lw=1, ms=4,
                  ls="-" if j == 0 else "--")  # fmt: skip
ax_a.set_xticks(range(len(TARGETS)), ["MAE", "runtime"])
ax_a.set_ylabel("CRPS relative to M2")
ax_a.axhline(1, color="#888888", lw=0.6)
ax_b.plot([75, 95], [75, 95], color="#888888", lw=0.6)
ax_b.set(xlabel="nominal coverage (%)", ylabel="observed coverage (%)", xticks=[80, 90])
ax_b.plot([], [], "ko-", ms=4, lw=1, label="MAE")
ax_b.plot([], [], "ks--", ms=4, lw=1, label="runtime")
ax_b.legend(frameon=False, loc="lower right")

sd = fig_data["mae"]["g_sd"]
ax_c.hist(sd[sd > 0], bins=np.logspace(np.log10(sd[sd > 0].min()), np.log10(sd.max()), 50), color="#2a78d6")
ax_c.axvline(fig_data["mae"]["pooled_sd"], color="#0b0b0b", lw=0.8, ls="--")
ax_c.annotate("pooled SD (M2)", (fig_data["mae"]["pooled_sd"], 0.95), xycoords=("data", "axes fraction"),
              xytext=(3, 0), textcoords="offset points", fontsize=7, va="top")  # fmt: skip
ax_c.set(xscale="log", xlabel="per-set SD of MAE (eV)", ylabel="number of hyperparameter sets")
ax_c.set_xlim(1e-4, 0.5)
ax_d.hexbin(fig_data["mae"]["g_mean"], sd.clip(lower=1e-4), yscale="log", gridsize=40, bins="log", cmap="Blues", mincnt=1, linewidths=0)
ax_d.set(xlabel="per-set mean MAE (eV)", ylabel="per-set SD of MAE (eV)")
rho = summary["mae"]["heteroskedasticity"]["spearman_sd_vs_mean"]
ax_d.text(0.02, 0.97, f"Spearman $\\rho$ = {rho:.2f}", transform=ax_d.transAxes, fontsize=7, va="top")
for ax, label in zip(axes.ravel(), "abcd"):
    ax.set_title(f"({label})", loc="left", fontweight="bold", fontsize=8)
fig.legend(*ax_a.get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False)
fig.savefig(FIG_DIR / "noise_ablation.pdf")
fig.savefig(FIG_DIR / "noise_ablation.png", dpi=300)
