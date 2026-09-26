"""Ways to map a composition onto the CrabNet benchmark's x1 to x20, scored against
real composition-property data.

In the sum-to-one variant a composition w (w1 + ... + w20 = 1, 3 or more nonzero)
sets the Space inputs x1 to x20. So far the mapping has been the identity, x = w, so an
inactive element leaves its hyperparameter at the lowest value. The duck tests
(constraint_duck_test.py, invert_inputs.py) found that this looks like a composition
but does not behave like one: two inputs carry most of the MAE importance, the surface
is smoother, more predictable and less noisy than Materials Project (MP) data, and
every constrained point lies far from the training runs.

Mappings compared on the same compositions:

- identity: x = w (current)
- inverted: identity, but alpha = 1 - w2 and emb_scaler = 1 - w6 - w15
- midpoint: x = 0.5 + 0.5 (w - 1/20), so an inactive element leaves its hyperparameter
  near the middle of its range
- random mixing: x = w E, where row i of E is a random point in the box, so every
  element moves every hyperparameter; three seeds
- elemental mixing: x = w E, where row i of E holds 20 min-max scaled properties of a
  real element (Al, Co, ..., Zr), i.e. the composition-weighted means used by Magpie;
  three random assignments of properties to x1 to x20

Rows of E are made to satisfy the Space constraints (x6 + x15 <= 1, x19 <= x20), which
then hold for every composition because both are kept under convex combinations.

Metrics follow constraint_duck_test.py: boosted trees refit from composition to the
surrogate's prediction, SHAP on that fit, 5-fold CV R², ridge share, nearest-neighbour
correlation, and the noise share over the surrogate's percentile input. MP reference
values come from constraint_duck_test_metrics.csv. matbench_steels (yield strength of
312 steels, 14 elements) is a second, experimental reference.

The script ends with a walk test: random search and a random-forest search on CrabNet
MAE under four of the mappings and on emulators of MP alloy formation energy and
energy above hull.

Inputs as in constraint_duck_test.py (Space surrogate, Sobol CSV, MP parquet), plus
https://ml.materialsproject.org/projects/matbench_steels.json.gz saved to
data/external/matbench. Run with Python 3.12, scikit-learn 1.4.1.post1, numpy<2,
pandas, pyarrow, shap, joblib, matplotlib and pymatgen<2024.7.
"""

# %% imports
import gzip
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shap
from joblib import Parallel, delayed
from pymatgen.core import Composition, Element
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
fig_dir = Path("reports/crabnet_hyperparameter_immi/figures")
n_samples = 5_000

# %% Space surrogate, x1..x20 in PARAM_BOUNDS order
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
objectives = ["mae", "runtime", "model_size"]
fixed = {  # fidelity1 = 1, c1_0 (no bias), c2_0 (RobustL1), c3_0 (mat2vec)
    "train_frac": 1.0,
    "bias": 0,
    "use_RobustL1": 1,
    "elem_prop_magpie": 0,
    "elem_prop_mat2vec": 1,
    "elem_prop_onehot": 0,
}
surrogate = joblib.load("models/crabnet_hyperparameter/surrogate_models_hgbr_opt.pkl")


def predict(x, u=0.5, cats=fixed):
    """Surrogate predictions at scaled points x (noise percentile u, as in the Space)."""
    frame = pd.DataFrame(lo + x * (hi - lo), columns=hp).assign(**cats)
    out = {}
    for o in objectives:
        if o != "model_size":
            frame[f"{o}_rank"] = 1 - u if o == "runtime" else u
        out[o] = surrogate[o].predict(frame[surrogate[o].feature_names_in_])
    return pd.DataFrame(out)


def space_constraints(x):
    """x6 + x15 <= 1 (reflect across the diagonal) and x19 <= x20 (sort)."""
    x = x.copy()
    flip = x[:, i_emb] + x[:, i_pos] > 1
    x[flip, i_emb], x[flip, i_pos] = 1 - x[flip, i_pos], 1 - x[flip, i_emb]
    x[:, [i_b1, i_b2]] = np.sort(x[:, [i_b1, i_b2]], axis=1)
    return x


# %% MP alloy compositions (3 or more elements, lowest-energy polymorph per formula)
elements = "Al Co Cr Cu Fe Hf Mg Mn Mo Nb Ni Re Si Sn Ta Ti V W Zn Zr".split()
mp = pq.read_table(
    "data/external/materials_project/summary_2026-04-13.parquet",
    columns=[
        "formula_pretty", "composition", "elements", "energy_above_hull",
        "formation_energy_per_atom", "deprecated",
    ],
).to_pandas()  # fmt: skip
mp = mp[~mp["deprecated"] & mp["formation_energy_per_atom"].notna()]
mp = mp[mp["elements"].apply(lambda s: len(s) >= 3 and set(s) <= set(elements))]
mp = mp.sort_values("energy_above_hull").drop_duplicates("formula_pretty")
w_mp = pd.DataFrame([dict(c) for c in mp["composition"]]).reindex(columns=elements)
w_mp = w_mp.fillna(0.0).to_numpy()
w_mp = w_mp / w_mp.sum(1, keepdims=True)
k_mp = (w_mp > 0).sum(1)

# random compositions with as many active elements as the MP alloys
w = np.zeros((n_samples, 20))
for row, k in zip(w, rng.choice(k_mp, n_samples)):
    row[rng.choice(20, k, replace=False)] = rng.dirichlet(np.ones(k))
w[:, [i_b1, i_b2]] = np.sort(w[:, [i_b1, i_b2]], axis=1)

# %% element descriptor tables for the mixing mappings
props = [
    "Z", "atomic_mass", "row", "group", "X", "atomic_radius", "van_der_waals_radius",
    "mendeleev_no", "electrical_resistivity", "molar_volume", "thermal_conductivity",
    "melting_point", "density_of_solid", "ionization_energy", "electron_affinity",
    "average_ionic_radius", "youngs_modulus", "coefficient_of_linear_thermal_expansion",
    "max_oxidation_state", "velocity_of_sound",
]  # fmt: skip
table = pd.DataFrame(
    {p: [float(getattr(Element(e), p)) for e in elements] for p in props}, index=elements
)
for p in ["electrical_resistivity", "thermal_conductivity"]:  # span decades
    table[p] = np.log10(table[p])
table = (table - table.min()) / (table.max() - table.min())


def identity(c):
    return space_constraints(c)


def inverted(c):
    x = c.copy()
    x[:, i_alpha] = 1 - c[:, i_alpha]
    x[:, i_emb] = 1 - c[:, i_emb] - c[:, i_pos]
    return space_constraints(x)


def midpoint(c):
    return space_constraints(0.5 + 0.5 * (c - 1 / 20))


mappings = {"identity": identity, "inverted": inverted, "midpoint": midpoint}
for s in range(3):
    e_rand = space_constraints(np.random.default_rng(s).random((20, 20)))
    e_elem = space_constraints(
        table.to_numpy()[:, np.random.default_rng(s).permutation(20)]
    )
    mappings[f"random mixing {s}"] = lambda c, e=e_rand: c @ e
    mappings[f"elemental mixing {s}"] = lambda c, e=e_elem: c @ e
families = ["identity", "inverted", "midpoint", "random mixing", "elemental mixing"]


def family(name):
    return name.rstrip(" 012")


# %% metrics (same pipeline as constraint_duck_test.py)
kf = KFold(5, shuffle=True, random_state=0)


def scorecard(X, y):
    hgb = HistGradientBoostingRegressor(max_iter=300, random_state=0)
    r2 = r2_score(y, cross_val_predict(hgb, X, y, cv=kf))
    r2_lin = r2_score(y, cross_val_predict(RidgeCV(np.logspace(-6, 3, 19)), X, y, cv=kf))
    hgb.fit(X, y)
    idx = rng.choice(len(X), min(1_500, len(X)), replace=False)
    sv = np.abs(shap.TreeExplainer(hgb).shap_values(X[idx])).mean(0)
    imp = sv / sv.sum()
    return {
        "r2_hgb": r2,
        "lin_share": max(r2_lin, 0) / r2,
        "n_eff": 1 / (imp**2).sum(),
        "top_share": imp.max(),
        "nn_autocorr": nn_autocorr(X, y),
    }


def nn_autocorr(X, y):
    idx = rng.choice(len(X), min(1_500, len(X)), replace=False)
    nbr = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X[idx])[1][:, 1]
    return np.corrcoef(y[idx], y[nbr])[0, 1]


sobol = pd.read_csv("data/external/crabnet_hyperparameter/sobol_regression.csv")
x_sobol = np.unique(((sobol[hp].to_numpy() - lo) / (hi - lo)).round(6), axis=0)
nn_sobol = NearestNeighbors(n_neighbors=1).fit(x_sobol)
x_box = space_constraints(rng.random((2_000, 20)))
d_box = np.median(nn_sobol.kneighbors(x_box)[0])

u_noise = np.linspace(0.05, 0.95, 19)
cat_grid = [
    {"bias": b, "use_RobustL1": r}
    | {f"elem_prop_{e}": int(e == ep) for e in ["magpie", "mat2vec", "onehot"]}
    | {"train_frac": 1.0}
    for b in (0, 1)
    for r in (0, 1)
    for ep in ["magpie", "mat2vec", "onehot"]
]


def within_share(reps, t):
    dev = reps[t] - reps.groupby("point")[t].transform("mean")
    return (dev**2).sum() / ((reps[t] - reps[t].mean()) ** 2).sum()


rows = []
for name, f in mappings.items():
    x = f(w)
    pred = predict(x)
    # same inputs as MP: MP alloy compositions (identity-type mappings get three random
    # element-to-input assignments, the mixing mappings use their own element rows)
    if family(name) in ["identity", "inverted", "midpoint"]:
        x_mp = [f(space_constraints(w_mp[:, np.argsort(p)])) for p in
                [np.random.default_rng(s).permutation(20) for s in range(3)]]  # fmt: skip
    else:
        x_mp = [f(w_mp)]
    reps = pd.concat([predict(x[:500], u).assign(point=range(500)) for u in u_noise])
    reps_cat = pd.concat(
        [predict(x[:500], cats=c).assign(point=range(500)) for c in cat_grid]
    )
    dist = nn_sobol.kneighbors(x[:2_000])[0][:, 0]
    for t in objectives:
        y = pred[t].to_numpy()
        rows.append(
            {"mapping": name, "family": family(name), "objective": t}
            | scorecard(w, y)
            | {
                "nn_autocorr_mp_alloys": np.mean(
                    [nn_autocorr(w_mp, predict(xm)[t].to_numpy()) for xm in x_mp]
                ),
                "within_share": within_share(reps, t) if t != "model_size" else np.nan,
                "categorical_share": within_share(reps_cat, t),
                "nn_dist_median": np.median(dist),
                "at_lowest_median": np.median((x < 0.05).sum(1)),
                "median": np.median(y),
                "best": y.min(),
                "spearman_mae": pd.Series(y).corr(pred["mae"], method="spearman"),
            }
        )
    print(name, f"{np.median(dist):.2f}", flush=True)
metrics = pd.DataFrame(rows)
metrics.to_csv(fig_dir / "composition_mappings_metrics.csv", index=False)

# %% references: MP (from the duck test) and matbench_steels
mp_ref = pd.read_csv(fig_dir / "constraint_duck_test_metrics.csv").query("family == 'MP'")
steels = json.load(gzip.open("data/external/matbench/matbench_steels.json.gz"))
steels = pd.DataFrame(steels["data"], columns=steels["columns"])
w_steel = pd.DataFrame(
    [Composition(c).fractional_composition.as_dict() for c in steels["composition"]]
).fillna(0.0)
steel = scorecard(w_steel.to_numpy(), steels["yield strength"].to_numpy(float))
steel |= {"n_inputs": w_steel.shape[1]}
print("steels", {k: round(v, 3) for k, v in steel.items()})
print("box: median distance to the nearest Sobol set", round(d_box, 3))

summary = metrics.groupby(["family", "objective"], sort=False).mean(numeric_only=True)
print(summary.round(3).to_string())

# %% figure: scorecard, one panel per metric, MP range shaded
blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
ink, muted, band = "#0b0b0b", "#898781", "#f0efec"
plt.rcParams.update(
    {
        "font.size": 9.5,
        "axes.edgecolor": muted,
        "axes.labelcolor": ink,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "xtick.color": muted,
        "ytick.color": muted,
        "xtick.labelcolor": ink,
        "ytick.labelcolor": ink,
    }
)
panels = [
    ("n_eff", "n_eff", "(a) Effective inputs of 20\n(from mean |SHAP|)", (0, 20)),
    ("r2_hgb", "r2_hgb", "(b) Predictable from\ncomposition (CV R²)", (0, 1.02)),
    ("lin_share", "lin_share", "(c) Share captured by\nlinear mixing", (0, 1.02)),
    ("nn_autocorr_mp_alloys", "nn_autocorr", "(d) Smoothness on MP\nalloy compositions", (0, 1.02)),
    ("within_share", "within_share", "(e) Noise share at a fixed\ninput (MP: polymorphs)", (0, 0.72)),
    ("nn_dist_median", None, "(f) Distance to the\nnearest training run", (0, 1.7)),
]  # fmt: skip
mp_rows = {"nn_autocorr": mp_ref.query("dataset == 'MP alloys'")}
styles = {"mae": (blue, "o", "MAE"), "runtime": (orange, "s", "runtime"),
          "model_size": (aqua, "^", "model size")}  # fmt: skip
fig, axes = plt.subplots(1, 6, figsize=(15, 3.9), sharey=True, layout="constrained")
for ax, (col, ref_col, title, xlim) in zip(axes, panels):
    if ref_col is not None:
        ref = mp_rows.get(ref_col, mp_ref)[ref_col]
        ax.axvspan(ref.min(), ref.max(), color=band, zorder=0)
        ax.axvline(ref.median(), color=muted, lw=1, ls=":", zorder=1)
        if ref_col in steel and col != "within_share":
            ax.plot(steel[ref_col], -0.75, "D", mfc="white", mec=ink, ms=6, zorder=3)
    else:
        ax.axvline(d_box, color=muted, lw=1, ls=":")
        ax.annotate("box", (d_box, -0.75), xytext=(3, 0), textcoords="offset points",
                    color=muted, va="center", fontsize=8.5)  # fmt: skip
    for y0, fam in enumerate(families):
        sub = metrics[metrics["family"] == fam]
        for j, t in enumerate(objectives):
            v = sub.loc[sub["objective"] == t, col].dropna()
            if v.empty:
                continue
            c, m, _ = styles[t]
            y = y0 + (j - 1) * 0.2
            if len(v) > 1:
                ax.plot([v.min(), v.max()], [y, y], color=c, lw=1.5, alpha=0.6)
            ax.plot(v.mean(), y, m, color=c, ms=7, mec="white", mew=1, zorder=4)
    ax.set_xlim(*xlim)
    ax.set_title(title, loc="left")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=band, lw=0.8)
    ax.set_axisbelow(True)
axes[0].set_yticks(range(len(families)), families)
axes[0].set_ylim(len(families) - 0.5, -1.1)
handles = [
    plt.Line2D([], [], color=c, marker=m, ls="", ms=7, label=label)
    for c, m, label in styles.values()
] + [
    plt.Rectangle((0, 0), 1, 1, color=band, label="MP range (12 property sets)"),
    plt.Line2D([], [], color=muted, ls=":", label="MP median"),
    plt.Line2D([], [], marker="D", mfc="white", mec=ink, ls="", ms=6,
               label="matbench_steels yield strength (14 elements)"),
]  # fmt: skip
fig.legend(handles=handles, loc="outside lower center", ncols=6, frameon=False)
fig.savefig(fig_dir / "composition_mappings.png", dpi=200, facecolor="white")

# %% walk test: the same two optimizers on the benchmark and on MP emulators
# Problems: CrabNet MAE under four mappings, and boosted-tree emulators of MP alloy
# formation energy and energy above hull (fit on the MP alloy compositions above).
# Optimizers: random search, and a random-forest search (lower confidence bound over
# 1,000 random candidates plus 500 moves of fraction between two elements of the
# incumbent). 10 random starts, 100 evaluations.


def draw(n, gen):
    """Random compositions with as many active elements as the MP alloys."""
    k = gen.choice(k_mp, n)
    active = np.argsort(np.argsort(gen.random((n, 20)), axis=1), axis=1) < k[:, None]
    g = gen.exponential(size=(n, 20)) * active
    return g / g.sum(1, keepdims=True)


def local_moves(best, n, gen):
    c = np.repeat(best[None], n, 0)
    i = gen.choice(np.flatnonzero(best > 0), n)
    d = gen.random(n) * best[i]
    c[np.arange(n), i] -= d
    c[np.arange(n), gen.integers(20, size=n)] += d
    return c[(c > 1e-9).sum(1) >= 3]


def mae_of(mapping):
    def f(c):
        x = mapping(c)
        frame = pd.DataFrame(lo + x * (hi - lo), columns=hp).assign(**fixed, mae_rank=0.5)
        return surrogate["mae"].predict(frame[surrogate["mae"].feature_names_in_])

    return f


emulator = {
    t: HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(w_mp, mp[col])
    for t, col in [("E_form", "formation_energy_per_atom"), ("E_hull", "energy_above_hull")]
}
problems = {
    "CrabNet MAE, identity": mae_of(identity),
    "CrabNet MAE, inverted": mae_of(inverted),
    "CrabNet MAE, random mixing": mae_of(mappings["random mixing 0"]),
    "CrabNet MAE, elemental mixing": mae_of(mappings["elemental mixing 0"]),
    "MP alloys, formation energy": emulator["E_form"].predict,
    "MP alloys, energy above hull": emulator["E_hull"].predict,
}
n_init, budget, n_seeds = 10, 100, 8


def forest_search(f, seed):
    gen = np.random.default_rng(seed)
    C = draw(n_init, gen)
    y = f(C)
    while len(y) < budget:
        rf = RandomForestRegressor(30, max_features=0.5, random_state=seed).fit(C, y)
        cand = np.vstack([draw(1_000, gen), local_moves(C[y.argmin()], 500, gen)])
        per_tree = np.stack([tree.predict(cand) for tree in rf.estimators_])
        c_new = cand[[np.argmin(per_tree.mean(0) - per_tree.std(0))]]
        C, y = np.vstack([C, c_new]), np.append(y, f(c_new))
    return np.minimum.accumulate(y)


walk_rows, curves = [], {}
for name, f in problems.items():
    pool = f(draw(100_000, np.random.default_rng(99)))
    forest = np.array(
        Parallel(n_jobs=4, prefer="threads")(
            delayed(forest_search)(f, s) for s in range(n_seeds)
        )
    )
    rand = np.minimum.accumulate(
        np.stack([f(draw(budget, np.random.default_rng(1_000 + s))) for s in range(50)]),
        axis=1,
    )
    f_best = min(pool.min(), forest.min())
    closed = {  # share of the gap from the pool median to the best known value
        m: (np.median(pool) - v) / (np.median(pool) - f_best)
        for m, v in [("random", rand), ("forest", forest)]
    }
    curves[name] = closed
    for m, v in closed.items():
        walk_rows += [
            {"problem": name, "optimizer": m, "evaluations": n, "gap_closed": np.median(v[:, n - 1])}
            for n in (10, 25, 50, 100)
        ]
    print(name, {m: np.median(v[:, -1]).round(3) for m, v in closed.items()}, flush=True)
walk = pd.DataFrame(walk_rows)
walk.to_csv(fig_dir / "composition_mappings_walk.csv", index=False)

fig, axes = plt.subplots(1, len(problems), figsize=(15, 3.2), sharey=True, layout="constrained")
n = np.arange(1, budget + 1)
for ax, (name, closed) in zip(axes, curves.items()):
    for m, c, label in [("random", muted, "random search"), ("forest", blue, "random-forest search")]:
        ax.fill_between(n, *np.percentile(closed[m], [25, 75], axis=0), color=c, alpha=0.18, lw=0)
        ax.plot(n, np.median(closed[m], axis=0), color=c, lw=2, label=label)
    ax.set_title(name.replace(", ", "\n"), loc="left")
    ax.set_xlabel("evaluations")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=band, lw=0.8)
    ax.spines["left"].set_visible(True)
axes[0].set_ylabel("share of the gap closed\n(pool median → best known)")
axes[0].legend(loc="lower right", frameon=False)
fig.savefig(fig_dir / "composition_mappings_walk.png", dpi=200, facecolor="white")
