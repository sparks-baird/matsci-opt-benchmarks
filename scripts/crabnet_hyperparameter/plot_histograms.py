"""Consolidated histogram figure and data checks for the CrabNet benchmark dataset.

Reads ``sobol_regression.csv`` (Zenodo 10.5281/zenodo.7694268) and writes
``histograms_panel.pdf/.png`` plus ``revision/analysis/data_checks.json``.
Set ``CRABNET_DATA`` to the CSV path if it is not in /tmp/crabnet_data.
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import (
    FuncFormatter,
    LogFormatterSciNotation,
    LogLocator,
    NullFormatter,
)

REPO = Path(__file__).resolve().parents[2]
FIG_DIR = REPO / "reports" / "crabnet_hyperparameter_immi" / "figures"
OUT_DIR = REPO / "reports" / "crabnet_hyperparameter_immi" / "revision" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV = os.environ.get("CRABNET_DATA", "/tmp/crabnet_data/sobol_regression.csv")

NUMERIC = [
    "N", "alpha", "d_model", "dim_feedforward", "dropout", "emb_scaler", "eps",
    "epochs_step", "fudge", "heads", "k", "lr", "pe_resolution", "ple_resolution",
    "pos_scaler", "weight_decay", "batch_size", "out_hidden4", "betas1", "betas2",
    "train_frac",
]  # fmt: skip
HP_COLUMNS = NUMERIC + ["bias", "criterion", "elem_prop"]
N_SOBOL, N_INTENDED = 2**16, 5

df = pd.read_csv(CSV)
repeats = df.groupby(HP_COLUMNS).size()
rep_counts = repeats.value_counts().sort_index()

# ---------------------------------------------------------------- data checks
checks = {
    "n_rows": len(df),
    "n_unique_sets": len(repeats),
    "n_nan_values": int(df.isna().sum().sum()),
    "hardware_values": df["hardware"].unique().tolist(),
    "n_sessions": df["session_id"].nunique(),
    "rows_per_session": df["session_id"].value_counts().to_dict(),
    "repeats_per_set": {
        "mean": repeats.mean(),
        "median": repeats.median(),
        "min": int(repeats.min()),
        "max": int(repeats.max()),
        "histogram": {int(k): int(v) for k, v in rep_counts.items()},
    },
    "successful_runs_per_sobol_point": len(df) / N_SOBOL,
    "attempted_runs_assuming_5_repeats": N_SOBOL * N_INTENDED,
    "fraction_attempted_completed": len(df) / (N_SOBOL * N_INTENDED),
    "sobol_points_with_no_success": N_SOBOL - len(repeats),
    "fraction_sobol_points_with_at_least_one_success": len(repeats) / N_SOBOL,
    "constraints": {
        "betas1_le_betas2_violations": int((df.betas1 > df.betas2).sum()),
        "emb_plus_pos_le_1_violations": int((df.emb_scaler + df.pos_scaler > 1).sum()),
        "max_emb_plus_pos": (df.emb_scaler + df.pos_scaler).max(),
    },
    "train_frac_below_lower_bound_0.01": {
        "n_rows": int((df.train_frac < 0.01).sum()),
        "values": sorted(df.loc[df.train_frac < 0.01, "train_frac"].unique().tolist()),
        "sessions": df.loc[df.train_frac < 0.01, "session_id"].unique().tolist(),
    },
}
objectives = {}
for col in ["mae", "rmse", "runtime", "model_size"]:
    x = df[col]
    objectives[col] = {
        "min": x.min(),
        "p1": x.quantile(0.01),
        "median": x.median(),
        "mean": x.mean(),
        "p99": x.quantile(0.99),
        "max": x.max(),
        "log10_max_over_min": np.log10(x.max() / x.min()),
        "log10_p99_over_p1": np.log10(x.quantile(0.99) / x.quantile(0.01)),
    }
objectives["runtime"]["n_runs_above_3600_s"] = int((df.runtime > 3600).sum())
checks["objectives"] = objectives
checks["units"] = {
    "mae_rmse": "eV (matbench_expt_gap experimental band gap, 5-fold mean)",
    "runtime": "seconds of wall-clock time per run on one RTX 2080 Ti, time() - t0",
    "model_size": "number of trainable parameters, crabnet count_parameters()",
}
# numbers for comparison with the originally submitted plotly histograms
# (plotly used 1000 s runtime bins centered on multiples of 1000 s)
rt_bins = np.histogram(df.runtime, bins=np.arange(-500, 35500, 1000))[0]
checks["original_figure_comparison"] = {
    "runtime_counts_1000s_centered_bins": rt_bins.tolist(),
    "model_size_tallest_bar_bin_0.5M": int(np.histogram(df.model_size, np.arange(0, 1.3e8, 5e5))[0].max()),
    "mae_tallest_bar_bin_0.005": int(np.histogram(df.mae, np.arange(0.15, 1.25, 0.005))[0].max()),
    "rmse_tallest_bar_bin_0.005": int(np.histogram(df.rmse, np.arange(0.4, 1.75, 0.005))[0].max()),
}
sessions = pd.crosstab(df.groupby(HP_COLUMNS).ngroup(), df["session_id"])
checks["runs_per_set_by_session"] = {
    sid: {int(k): int(v) for k, v in sessions[sid].value_counts().sort_index().items()}
    for sid in sessions.columns
}
checks["notes"] = {
    "filtering": (
        "Notebook 1.0 pulls only MongoDB documents that contain 'hardware' and "
        "'scores', i.e. runs that finished. Runs that raised an exception were "
        "stored with an 'error' field and no 'scores', so they are absent. Runs "
        "lost to SLURM walltime or preemption never wrote a document. No NaNs, "
        "no threshold on MAE, RMSE, runtime or model size, and no explicit "
        "'bad hyperparameter' filter was found in notebooks 1.0 to 1.2 or in "
        "the submission script. NaN predictions were replaced by 0 before "
        "scoring (np.nan_to_num), so diverged models appear as high-MAE runs."
    ),
    "repeats": (
        "Session 6bf8f274 has at most one run per set and session 201d73be at most "
        "four (2 sample seeds x num_repeats=2 in "
        "crabnet_hyperparameter_submitit.py), giving 5 intended runs per point. "
        "173,219 / 65,536 = 2.64 is successful runs per attempted Sobol point, "
        "which is the 'approximately 2.6 repeats' of the submitted text; the "
        "mean among sets with at least one success is 4.17."
    ),
    "debug_rows": (
        "16 rows from two short test sessions (15ec2ebc, 5fd7e1c6) use "
        "train_frac = 0.003 (dummy setting), below the 0.01 lower bound. They "
        "are included in the CSV and in all figures."
    ),
    "original_figures": (
        "The submitted plotly histograms (image1 to image5) match this CSV: "
        "repeat counts 130, 1294, 6647, 16835, 16644; runtime bars with 1000 s "
        "bins give 159968, 12479, 694, 47, ...; tallest MAE, RMSE and model "
        "size bars (about 2600, 2000, 3100) and x-ranges agree."
    ),
    "units": (
        "model_size is a parameter count, so the 'model size [MB]' axis label "
        "in plot_pareto_fronts.py is wrong."
    ),
}
(OUT_DIR / "data_checks.json").write_text(json.dumps(checks, indent=2, default=float))

# ---------------------------------------------------------------- figure
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)
color = "#2a78d6"
fig, axes = plt.subplots(2, 3, figsize=(17 / 2.54, 10.5 / 2.54), layout="constrained")
ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()

ax_a.bar(rep_counts.index, rep_counts.values, width=0.7, color=color)
kfmt = FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}")
for k, v in rep_counts.items():
    ax_a.annotate(kfmt(round(v, -2) if v >= 1000 else v), (k, v), xytext=(0, 2),
                  textcoords="offset points", ha="center", fontsize=7)  # fmt: skip
ax_a.set(xlabel="successful repeats per hyperparameter set",
         ylabel="number of hyperparameter sets", xticks=range(1, 6))  # fmt: skip
ax_a.yaxis.set_major_formatter(kfmt)
ax_a.set_ylim(0, rep_counts.max() * 1.12)

ax_b.hist(df.mae, bins=100, color=color)
ax_b.set(xlabel="MAE (eV)", ylabel="number of runs")
ax_c.hist(df.rmse, bins=100, color=color)
ax_c.set(xlabel="RMSE (eV)", ylabel="number of runs")

# runtime spans about four decades, so both axes are logarithmic
log_bins = np.logspace(np.log10(df.runtime.min()), np.log10(df.runtime.max()), 61)
ax_d.hist(df.runtime, bins=log_bins, color=color)
ax_d.set(xscale="log", yscale="log", xlabel="GPU runtime (s)", ylabel="number of runs")
for axis in (ax_d.xaxis, ax_d.yaxis):
    axis.set_major_locator(LogLocator(base=10))
    axis.set_major_formatter(LogFormatterSciNotation(base=10))
    axis.set_minor_formatter(NullFormatter())
ax_d.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
ax_d.yaxis.set_minor_locator(LogLocator(base=10, subs=[]))
ax_d.set_ylim(0.7, None)

ax_e.hist(df.model_size / 1e6, bins=100, color=color)
ax_e.set(xlabel="model size (millions of trainable parameters)", ylabel="number of runs")
ax_f.axis("off")

for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e], "abcde"):
    ax.set_title(f"({label})", loc="left", fontweight="bold", fontsize=8)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.5)
    ax.set_axisbelow(True)

fig.savefig(FIG_DIR / "histograms_panel.pdf")
fig.savefig(FIG_DIR / "histograms_panel.png", dpi=300)
