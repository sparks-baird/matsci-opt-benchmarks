"""Check whether ``train_frac`` behaves like a fidelity parameter.

For a random subset of the deposited hyperparameter sets, the released surrogate
(``surrogate_models.pkl``, Zenodo 10.5281/zenodo.7694268) predicts MAE and GPU
runtime at several training-set fractions, with the noise percentile fixed at
the median (rank = 0.5). A useful low-fidelity setting is one that is much
cheaper than ``train_frac = 1`` while still ranking hyperparameter sets in the
same order, so we report the Spearman correlation with the full-fidelity MAE,
the overlap of the best 10% of sets, and the median runtime ratio.

The pickle was written with scikit-learn 1.0.1, so run this with a matching
environment (e.g. Python 3.10, scikit-learn 1.0.2, numpy < 1.24, pandas < 2).

Usage (from the repository root)::

    python scripts/crabnet_hyperparameter/fidelity_correlation.py DATA_DIR
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_pareto_fronts import build_features, repeat_average  # noqa: E402

plt.switch_backend("Agg")
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7})

data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
repo = Path(__file__).resolve().parents[2]
fig_dir = repo / "reports" / "crabnet_hyperparameter_immi" / "figures"
out_json = (
    repo
    / "reports"
    / "crabnet_hyperparameter_immi"
    / "revision"
    / "analysis"
    / "fidelity_correlation.json"
)
out_json.parent.mkdir(parents=True, exist_ok=True)

fractions = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
n_sets = 5000

df = pd.read_csv(data_dir / "sobol_regression.csv")
sets = repeat_average(df).sample(n_sets, random_state=0).reset_index(drop=True)
models = joblib.load(data_dir / "surrogate_models.pkl")

mae, runtime = {}, {}
for f in fractions:
    X = build_features(sets.assign(train_frac=f))
    mae[f] = models["mae"].predict(X.assign(mae_rank=0.5).to_numpy())
    runtime[f] = models["runtime"].predict(X.assign(runtime_rank=0.5).to_numpy())

top = int(0.1 * n_sets)
best_full = set(np.argsort(mae[1.0])[:top])
rows = []
for f in fractions:
    rows.append(
        {
            "train_frac": f,
            "spearman_mae_vs_full": float(spearmanr(mae[f], mae[1.0])[0]),
            "top10pct_overlap_with_full": len(best_full & set(np.argsort(mae[f])[:top]))
            / top,
            "median_mae_eV": float(np.median(mae[f])),
            "median_runtime_s": float(np.median(runtime[f])),
            "median_runtime_ratio_to_full": float(np.median(runtime[f] / runtime[1.0])),
        }
    )
summary = pd.DataFrame(rows)
json.dump(
    {"n_hyperparameter_sets": n_sets, "noise_rank": 0.5, "rows": rows},
    open(out_json, "w"),
    indent=2,
)
print(summary.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))
ax = axes[0]
ax.plot(summary.train_frac, summary.spearman_mae_vs_full, "o-", color="#0072B2", label="Spearman $\\rho$ with full-fidelity MAE")
ax.plot(summary.train_frac, summary.top10pct_overlap_with_full, "s--", color="#D55E00", label="overlap of best 10% of sets")
ax.set_xlabel("training-set fraction (train_frac)")
ax.set_ylim(0, 1.02)
ax.set_title("(a) agreement with train_frac = 1")
ax.legend(fontsize=6, loc="lower right")
ax.grid(True, ls=":", lw=0.4)
ax = axes[1]
ax.plot(summary.train_frac, summary.median_runtime_s, "o-", color="#009E73", label="median GPU runtime")
ax.set_xlabel("training-set fraction (train_frac)")
ax.set_ylabel("GPU runtime [s]")
ax2 = ax.twinx()
ax2.plot(summary.train_frac, summary.median_mae_eV, "s--", color="#CC79A7", label="median MAE")
ax2.set_ylabel("MAE [eV]")
ax.set_title("(b) cost and accuracy vs. fidelity")
lines = ax.get_lines() + ax2.get_lines()
ax.legend(lines, [ln.get_label() for ln in lines], fontsize=6, loc="upper center")
ax.grid(True, ls=":", lw=0.4)
fig.tight_layout()
fig.savefig(fig_dir / "fidelity_surrogate.pdf")
fig.savefig(fig_dir / "fidelity_surrogate.png", dpi=300)
