"""Plot Pareto fronts for the CrabNet hyperparameter benchmark dataset.

Two multi-panel figures are produced, one subplot per objective pair
(MAE, RMSE, runtime, model size -> six pairwise combinations):

* ``pareto_surrogate`` -- objectives predicted by the released RandomForest
  surrogate (``surrogate_models.pkl``) evaluated at the median noise
  percentile (rank = 0.5) for every unique hyperparameter set.
* ``pareto_rawdata`` -- objectives taken directly from the raw runs
  (``sobol_regression.csv``), averaged over the repeated evaluations of each
  identical hyperparameter set.

All four objectives are minimized, so the Pareto front is the lower-left
frontier of each scatter. Each subplot title reports the Spearman rank
correlation between the two objectives (this addresses reviewer questions
about whether the objectives exhibit genuine trade-offs or are simply
correlated).

A third figure, ``pareto_mae_rmse_zoom``, zooms into the low-error corner of
MAE vs RMSE for the repeat-averaged sets, the individual runs, and the
surrogate, with points colored by loss function (``criterion``) and the
region dominated by the Pareto front shaded.

The dataset and surrogate are the public Zenodo deposition
``10.5281/zenodo.7694268``. If the files are not found locally they are
downloaded from Zenodo into a cache directory.

Usage (from the repository root)::

    python scripts/crabnet_hyperparameter/plot_pareto_fronts.py

Optional arguments::

    --data-dir DIR   directory holding / caching the Zenodo files
    --out-dir DIR    directory to write the figures into
    --sample N       evaluate only N random hyperparameter sets (faster)
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from itertools import combinations
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")  # non-interactive backend for headless figure export
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7})

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "crabnet_hyperparameter_immi" / "figures"

ZENODO_RECORD = "7694268"
ZENODO_FILES = {
    "sobol_regression.csv": (
        f"https://zenodo.org/api/records/{ZENODO_RECORD}"
        "/files/sobol_regression.csv/content"
    ),
    "surrogate_models.pkl": (
        f"https://zenodo.org/api/records/{ZENODO_RECORD}"
        "/files/surrogate_models.pkl/content"
    ),
}

# Numeric hyperparameters, in the order the surrogate was fit on.
NUMERIC = [
    "N",
    "alpha",
    "d_model",
    "dim_feedforward",
    "dropout",
    "emb_scaler",
    "eps",
    "epochs_step",
    "fudge",
    "heads",
    "k",
    "lr",
    "pe_resolution",
    "ple_resolution",
    "pos_scaler",
    "weight_decay",
    "batch_size",
    "out_hidden4",
    "betas1",
    "betas2",
    "train_frac",
]
# Full feature order the surrogate models expect (28 features; the *_rank
# column is appended for the mae / rmse / runtime models -> 29 features).
COMMON28 = NUMERIC + [
    "bias",
    "criterion_RobustL1",
    "criterion_RobustL2",
    "elem_prop_magpie",
    "elem_prop_mat2vec",
    "elem_prop_onehot",
    "hardware_2080ti",
]
# Columns that define a unique hyperparameter set (excludes fixed hardware).
HP_COLUMNS = NUMERIC + ["bias", "criterion", "elem_prop"]

OBJECTIVES = ["mae", "rmse", "runtime", "model_size"]
OBJ_LABELS = {
    "mae": "MAE [eV]",
    "rmse": "RMSE [eV]",
    "runtime": "GPU runtime [s]",
    "model_size": "model size [$10^6$ parameters]",
}
RANK_COL = {"mae": "mae_rank", "rmse": "rmse_rank", "runtime": "runtime_rank"}

# Low-error corner shown in the MAE vs RMSE zoom figure [eV].
ZOOM_MAE = (0.165, 0.235)
ZOOM_RMSE = (0.43, 0.60)
LOSS_COLORS = {"RobustL1": "#2a78d6", "RobustL2": "#eb6834"}
INK = "#0b0b0b"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _resolve_file(name: str, data_dir: Path, repo_candidates: list[Path]) -> Path:
    """Return a local path to ``name``, downloading from Zenodo if needed."""
    for cand in repo_candidates:
        if cand.is_file():
            print(f"  using local {name}: {cand}")
            return cand
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / name
    if dest.is_file():
        print(f"  using cached {name}: {dest}")
        return dest
    url = ZENODO_FILES[name]
    print(f"  downloading {name} from {url} ...")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (trusted Zenodo URL)
    return dest


def load_inputs(data_dir: Path):
    """Load the raw dataframe and the surrogate model dictionary."""
    csv_path = _resolve_file(
        "sobol_regression.csv",
        data_dir,
        [
            REPO_ROOT
            / "data"
            / "processed"
            / "crabnet_hyperparameter"
            / "sobol_regression.csv"
        ],
    )
    pkl_path = _resolve_file(
        "surrogate_models.pkl",
        data_dir,
        [REPO_ROOT / "models" / "crabnet_hyperparameter" / "surrogate_models.pkl"],
    )
    df = pd.read_csv(csv_path)
    models = joblib.load(pkl_path)
    return df, models


# --------------------------------------------------------------------------- #
# Feature construction / prediction
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 28 surrogate input columns (in fit order) from raw rows."""
    X = pd.DataFrame(index=df.index)
    for col in NUMERIC:
        X[col] = df[col].astype(float)
    X["bias"] = df["bias"].astype(int)
    X["criterion_RobustL1"] = (df["criterion"] == "RobustL1").astype(int)
    X["criterion_RobustL2"] = (df["criterion"] == "RobustL2").astype(int)
    for elem in ("magpie", "mat2vec", "onehot"):
        X[f"elem_prop_{elem}"] = (df["elem_prop"] == elem).astype(int)
    X["hardware_2080ti"] = (df["hardware"] == "2080ti").astype(int)
    return X[COMMON28]


def surrogate_predict(
    models, features: pd.DataFrame, rank: float = 0.5
) -> pd.DataFrame:
    """Predict all four objectives at a fixed noise percentile ``rank``."""
    preds = {}
    for obj in OBJECTIVES:
        if obj in RANK_COL:
            X = features.assign(**{RANK_COL[obj]: rank})
        else:
            X = features
        preds[obj] = models[obj].predict(X.to_numpy())
    return pd.DataFrame(preds, index=features.index)


def repeat_average(df: pd.DataFrame) -> pd.DataFrame:
    """Average objectives over the repeats of each identical parameter set."""
    keys = [df[c].round(6) if df[c].dtype.kind in "fc" else df[c] for c in HP_COLUMNS]
    grouped = df.groupby(keys, sort=False)
    # one representative parameter row per group + averaged objectives
    reps = grouped[HP_COLUMNS + ["hardware"]].first().reset_index(drop=True)
    means = grouped[OBJECTIVES].mean().reset_index(drop=True)
    out = pd.concat([reps, means], axis=1)
    return out


# --------------------------------------------------------------------------- #
# Pareto front
# --------------------------------------------------------------------------- #
def pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean mask of Pareto-optimal points when minimizing both ``x`` and ``y``."""
    order = np.lexsort((y, x))  # sort by x asc, ties broken by y asc
    mask = np.zeros(len(x), dtype=bool)
    best_y = np.inf
    for idx in order:
        if y[idx] < best_y:
            mask[idx] = True
            best_y = y[idx]
    return mask


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def _use_log(values: np.ndarray) -> bool:
    v = values[values > 0]
    return v.size > 0 and (v.max() / v.min() > 20)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_panels(data: pd.DataFrame, title: str, out_stem: Path) -> None:
    """Draw a 2x3 grid of pairwise objective scatters with Pareto fronts."""
    pairs = list(combinations(OBJECTIVES, 2))
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.7))
    axes = axes.ravel()
    for ax, (ox, oy) in zip(axes, pairs):
        x = data[ox].to_numpy(dtype=float)
        y = data[oy].to_numpy(dtype=float)
        if ox == "model_size":
            x = x / 1e6
        if oy == "model_size":
            y = y / 1e6
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]

        ax.scatter(
            x,
            y,
            s=1.5,
            c="#9ecae1",
            alpha=0.35,
            edgecolors="none",
            rasterized=True,
            label="all sets",
        )
        mask = pareto_mask(x, y)
        fx, fy = x[mask], y[mask]
        line_order = np.argsort(fx)
        ax.plot(fx[line_order], fy[line_order], "-", color="#d62728", lw=0.9, zorder=5)
        ax.scatter(
            fx,
            fy,
            s=7,
            c="#d62728",
            edgecolors="k",
            linewidths=0.3,
            zorder=6,
            label=f"Pareto front (n={mask.sum()})",
        )

        if _use_log(x):
            ax.set_xscale("log")
        if _use_log(y):
            ax.set_yscale("log")
        ax.set_xlabel(OBJ_LABELS[ox])
        ax.set_ylabel(OBJ_LABELS[oy])
        rho = _spearman(x, y)
        ax.set_title(
            f"{OBJ_LABELS[ox].split(' [')[0]} vs "
            f"{OBJ_LABELS[oy].split(' [')[0]}  (Spearman "
            f"$\\rho$={rho:.2f})",
        )
        ax.legend(fontsize=6, loc="upper right", framealpha=0.9, markerscale=1.5)
        ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)

    fig.suptitle(title, fontsize=8, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)  # vector version for print-quality submission
    plt.close(fig)
    print(f"  wrote {png}")
    print(f"  wrote {pdf}")


def plot_mae_rmse_zoom(
    raw: pd.DataFrame, runs: pd.DataFrame, surrogate: pd.DataFrame, out_stem: Path
) -> None:
    """Zoom into the low-error corner of MAE vs RMSE, colored by loss function.

    Panel (a) is the full repeat-averaged scatter with the zoom window drawn on
    it; panels (b) to (d) show the window for the repeat-averaged sets, the
    individual runs and the surrogate. The Pareto front is computed over all
    points, and in the zoomed panels the region it dominates is shaded. The
    Spearman correlation is computed over the points shown in each panel.
    """
    panels = [
        ("(a) All sets, repeat-averaged", raw, False),
        ("(b) Zoom: repeat-averaged sets", raw, True),
        ("(c) Zoom: individual runs", runs, True),
        ("(d) Zoom: surrogate (median percentile)", surrogate, True),
    ]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.8))
    for ax, (title, data, zoom) in zip(axes.ravel(), panels):
        x = data["mae"].to_numpy(dtype=float)
        y = data["rmse"].to_numpy(dtype=float)
        loss = data["criterion"].to_numpy()
        front = pareto_mask(x, y)
        shown = np.ones(len(x), dtype=bool)
        if zoom:
            shown = (
                (x >= ZOOM_MAE[0])
                & (x <= ZOOM_MAE[1])
                & (y >= ZOOM_RMSE[0])
                & (y <= ZOOM_RMSE[1])
            )
        # shuffle so neither loss function is always drawn on top
        idx = rng.permutation(np.flatnonzero(shown))
        ax.scatter(
            x[idx],
            y[idx],
            s=6 if zoom else 1.5,
            c=[LOSS_COLORS[c] for c in loss[idx]],
            alpha=0.6 if zoom else 0.3,
            edgecolors="none",
            rasterized=not zoom,
            zorder=2,
        )
        if zoom:
            ax.set_xlim(ZOOM_MAE)
            ax.set_ylim(ZOOM_RMSE)
        else:
            ax.add_patch(
                plt.Rectangle(
                    (ZOOM_MAE[0], ZOOM_RMSE[0]),
                    ZOOM_MAE[1] - ZOOM_MAE[0],
                    ZOOM_RMSE[1] - ZOOM_RMSE[0],
                    fill=False,
                    ec=INK,
                    lw=0.8,
                    zorder=6,
                )
            )
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        order = np.argsort(x[front])
        fx, fy = x[front][order], y[front][order]
        if zoom:
            # staircase boundary of the region dominated by the front
            sx = np.concatenate([[fx[0]], np.repeat(fx, 2)[1:], [xlim[1]]])
            sy = np.concatenate([[ylim[1]], np.repeat(fy, 2)])
            ax.fill(
                np.append(sx, xlim[1]),
                np.append(sy, ylim[1]),
                color="#ebeae5",
                lw=0,
                zorder=0.5,
            )
            ax.plot(sx, sy, color=INK, lw=0.8, zorder=1.8)
        ax.scatter(
            fx, fy, s=16, c=INK, edgecolors="white", linewidths=0.6, zorder=5
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        rho = _spearman(x[shown], y[shown])
        ax.set_title(
            f"{title}\nPareto front n = {front.sum()}, Spearman $\\rho$ = {rho:.2f}"
        )
        ax.set_xlabel(OBJ_LABELS["mae"])
        ax.set_ylabel(OBJ_LABELS["rmse"])
        ax.grid(True, color="#e1e0d9", lw=0.4)

    handles = [
        plt.Line2D(
            [], [], ls="", marker="o", ms=4, mfc=color, mec="none",
            label=f"{name} loss",
        )
        for name, color in LOSS_COLORS.items()
    ] + [
        plt.Line2D(
            [], [], color=INK, lw=0.8, marker="o", ms=4, mec="white",
            label="Pareto front",
        ),
        plt.Rectangle((0, 0), 1, 1, fc="#ebeae5", ec="none", label="dominated region"),
        plt.Rectangle((0, 0), 1, 1, fill=False, ec=INK, lw=0.8, label="zoom window"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  wrote {png}")
    print(f"  wrote {pdf}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "CRABNET_DATA_DIR", Path.home() / ".cache" / "crabnet_hyperparameter"
            )
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="evaluate only N random hyperparameter sets",
    )
    args = parser.parse_args(argv)

    print("Loading inputs ...")
    df, models = load_inputs(args.data_dir)
    print(f"  raw runs: {len(df):,}")

    print("Averaging repeats ...")
    raw = repeat_average(df)
    if args.sample and args.sample < len(raw):
        raw = raw.sample(args.sample, random_state=0).reset_index(drop=True)
    print(f"  unique hyperparameter sets: {len(raw):,}")

    print("Evaluating surrogate at median percentile (rank=0.5) ...")
    feats = build_features(raw)
    surrogate = surrogate_predict(models, feats, rank=0.5)
    surrogate = pd.concat([raw[HP_COLUMNS], surrogate], axis=1)

    print("Plotting raw-data Pareto fronts ...")
    plot_panels(
        raw,
        "Raw data (objectives averaged over repeats)",
        args.out_dir / "pareto_rawdata",
    )
    print("Plotting surrogate-model Pareto fronts ...")
    plot_panels(
        surrogate,
        "Surrogate model (median noise percentile, rank = 0.5)",
        args.out_dir / "pareto_surrogate",
    )
    print("Plotting MAE vs RMSE low-error zoom ...")
    plot_mae_rmse_zoom(raw, df, surrogate, args.out_dir / "pareto_mae_rmse_zoom")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
