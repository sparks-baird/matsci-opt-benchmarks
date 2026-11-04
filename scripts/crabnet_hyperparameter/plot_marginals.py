"""Marginal distributions of the numeric hyperparameters in the CrabNet dataset.

Compares the unique completed hyperparameter sets in ``sobol_regression.csv``
with (i) the uniform distribution on each parameter's bounds and (ii) a fresh
scrambled Sobol reference: 2**18 scipy Sobol points filtered by the two design
constraints (betas1 <= betas2, emb_scaler + pos_scaler <= 1), first 65,536
accepted points kept, integer parameters rounded as in Ax. Writes
``marginals.pdf/.png`` and ``revision/analysis/marginals_summary.json``.
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
from scipy.stats import qmc

REPO = Path(__file__).resolve().parents[2]
FIG_DIR = REPO / "reports" / "crabnet_hyperparameter_immi" / "figures"
OUT_DIR = REPO / "reports" / "crabnet_hyperparameter_immi" / "revision" / "analysis"
CSV = os.environ.get("CRABNET_DATA", "/tmp/crabnet_data/sobol_regression.csv")

# bounds from get_parameters() in utils/parameters.py; True marks integer type
BOUNDS = {
    "N": (1, 10, True),
    "alpha": (0.0, 1.0, False),
    "d_model": (100, 1024, True),
    "dim_feedforward": (1024, 4096, True),
    "dropout": (0.0, 1.0, False),
    "emb_scaler": (0.0, 1.0, False),
    "eps": (1e-7, 1e-4, False),
    "epochs_step": (5, 20, True),
    "fudge": (0.0, 0.1, False),
    "heads": (1, 10, True),
    "k": (2, 10, True),
    "lr": (1e-4, 6e-3, False),
    "pe_resolution": (2500, 10000, True),
    "ple_resolution": (2500, 10000, True),
    "pos_scaler": (0.0, 1.0, False),
    "weight_decay": (0.0, 1.0, False),
    "batch_size": (32, 256, True),
    "out_hidden4": (32, 512, True),
    "betas1": (0.5, 0.9999, False),
    "betas2": (0.5, 0.9999, False),
    "train_frac": (0.01, 1.0, False),
}
NAMES = list(BOUNDS)
HP_COLUMNS = NAMES + ["bias", "criterion", "elem_prop"]
# CDFs of the constrained-triangle marginals (uniform on each 2D triangle)
TRIANGLE_CDF = {
    "betas1": lambda u: 1 - (1 - u) ** 2,
    "betas2": lambda u: u**2,
    "emb_scaler": lambda u: 1 - (1 - u) ** 2,
    "pos_scaler": lambda u: 1 - (1 - u) ** 2,
}

df = pd.read_csv(CSV)
df = df[df.train_frac >= 0.01]  # drop the 16 debug rows with train_frac = 0.003
sets = df.drop_duplicates(HP_COLUMNS)[NAMES]
lo = np.array([BOUNDS[n][0] for n in NAMES], float)
hi = np.array([BOUNDS[n][1] for n in NAMES], float)
is_int = np.array([BOUNDS[n][2] for n in NAMES])
data_u = pd.DataFrame((sets.to_numpy() - lo) / (hi - lo), columns=NAMES)

# reference constrained Sobol design
raw = qmc.Sobol(d=len(NAMES), scramble=True, seed=0).random(2**18)
ref = pd.DataFrame(raw, columns=NAMES)
ok = (ref.betas1 <= ref.betas2) & (ref.emb_scaler + ref.pos_scaler <= 1)
ref = ref[ok].iloc[: 2**16]
native = lo + ref.to_numpy() * (hi - lo)
native[:, is_int] = np.round(native[:, is_int])
ref_u = pd.DataFrame((native - lo) / (hi - lo), columns=NAMES)

rows = []
for n in NAMES:
    d, r = data_u[n].to_numpy(), ref_u[n].to_numpy()
    row = {
        "hyperparameter": n,
        "integer": bool(BOUNDS[n][2]),
        "ks_data_vs_uniform": stats.kstest(d, "uniform").statistic,
        "ks_reference_vs_uniform": stats.kstest(r, "uniform").statistic,
        "ks_data_vs_reference": stats.ks_2samp(d, r).statistic,
        "ks_data_vs_reference_pvalue": stats.ks_2samp(d, r).pvalue,
        "data_mean_scaled": d.mean(),
        "reference_mean_scaled": r.mean(),
    }
    if n in TRIANGLE_CDF:
        row["ks_data_vs_constrained_triangle"] = stats.kstest(d, TRIANGLE_CDF[n]).statistic
    # completion fraction = completed sets / reference (attempted) points per decile
    edges = np.linspace(0, 1, 11)
    edges[0], edges[-1] = -1, 2
    cd = np.histogram(d, edges)[0]
    cr = np.histogram(r, edges)[0]
    row["completion_fraction_lowest_decile"] = cd[0] / cr[0]
    row["completion_fraction_highest_decile"] = cd[-1] / cr[-1]
    rows.append(row)
ks = pd.DataFrame(rows).sort_values("ks_data_vs_reference", ascending=False)

beta = {
    str(k): {
        "marginal": f"Beta(1, {k}), density {k}(1 - x)^{k - 1}",
        "mean": 1 / (k + 1),
        "median": 1 - 0.5 ** (1 / k),
        "p95": 1 - 0.05 ** (1 / k),
        "prob_x_gt_0.5": 0.5**k,
        "feasible_fraction_of_unit_cube": 1 / float(np.prod(np.arange(1, k + 1))),
    }
    for k in (5, 10, 20)
}
summary = {
    "n_unique_sets_used": len(sets),
    "note_sets": "41,550 unique sets minus 7 debug sets with train_frac = 0.003",
    "reference": "scipy qmc.Sobol(d=21, scramble=True, seed=0), 2**18 draws, constraint filter, first 65,536 kept, integers rounded",
    "reference_acceptance_rate": ok.mean(),
    "ks_table": ks.to_dict(orient="records"),
    "sum_constraint_marginals_uniform_on_simplex": beta,
}
(OUT_DIR / "marginals_summary.json").write_text(json.dumps(summary, indent=2, default=float))

# ---------------------------------------------------------------- figure
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)
fig, axes = plt.subplots(4, 6, figsize=(17 / 2.54, 13 / 2.54), layout="constrained", sharey=True)
for ax, n in zip(axes.ravel(), NAMES):
    lo_n, hi_n, integer = BOUNDS[n]
    if integer and hi_n - lo_n <= 20:
        bins = (np.arange(lo_n, hi_n + 2) - 0.5 - lo_n) / (hi_n - lo_n)
    else:
        bins = np.linspace(0, 1, 21)
    ax.hist(data_u[n], bins=bins, density=True, color="#2a78d6", label="deposited sets")
    ax.hist(ref_u[n], bins=bins, density=True, histtype="step", color="#eb6834", lw=1, label="constrained Sobol reference")
    ks_val = ks.set_index("hyperparameter").loc[n, "ks_data_vs_reference"]
    ax.set_title(f"{n}\nKS = {ks_val:.3f}", fontsize=7)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_ylim(0, 2.3)
for ax in axes.ravel()[len(NAMES):]:
    ax.axis("off")
for ax in axes[:, 0]:
    ax.set_ylabel("density")
fig.supxlabel("value scaled to [0, 1] by parameter bounds", fontsize=7)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower right", bbox_to_anchor=(0.98, 0.06), frameon=False)
fig.savefig(FIG_DIR / "marginals.pdf")
fig.savefig(FIG_DIR / "marginals.png", dpi=300)
