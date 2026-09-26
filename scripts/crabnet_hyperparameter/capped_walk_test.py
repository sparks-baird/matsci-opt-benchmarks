"""Walk test for the sum-to-one CrabNet benchmark with a cap: minimize MAE subject to
runtime (or model size) at or below a cap.

Random search and a random-forest search run on two settings, 100 evaluations each:

- the benchmark as it would be used: compositions anywhere on the simplex (3 or 4
  active elements, as in composition_mappings.py), sent through the Space surrogate
- real Materials Project (MP) compositions: the optimizers choose among the 2,239 MP
  compositions of the 20 alloy elements with 3 or more elements and an energy above
  hull of at most 0.1 eV/atom. The CrabNet problems use the surrogate at those
  compositions, and the MP problem uses the real DFT formation energy, with a cap on
  the real DFT density. Both problems then have identical inputs.

Boosted-tree emulators of MP (as in composition_mappings.py) are not used as a
reference here. At random compositions they are far from the MP data they were fit
on and return values that cannot occur (negative energies above hull, formation
energies far below the lowest MP value).

Caps are set from the random pool so that 25% or 10% of random compositions meet them.
The random-forest search models the capped quantity with a second forest and only
picks candidates that at least half its trees place under the cap. The score is the
share of the gap closed between the median random composition that meets the cap and
the best known one that meets it.

Inputs as in composition_mappings.py (Space surrogate, MP parquet). Run with Python
3.12, scikit-learn 1.4.1.post1, numpy<2, pandas, pyarrow, joblib, matplotlib and
pymatgen<2024.7.
"""

# %% imports
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from pymatgen.core import Element
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
fig_dir = Path("reports/crabnet_hyperparameter_immi/figures")

# %% Space surrogate and mappings (as in composition_mappings.py)
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
fixed = {  # fidelity1 = 1, c1_0 (no bias), c2_0 (RobustL1), c3_0 (mat2vec)
    "train_frac": 1.0,
    "bias": 0,
    "use_RobustL1": 1,
    "elem_prop_magpie": 0,
    "elem_prop_mat2vec": 1,
    "elem_prop_onehot": 0,
}
surrogate = joblib.load("models/crabnet_hyperparameter/surrogate_models_hgbr_opt.pkl")


def predict(x):
    """MAE, runtime and model size at scaled points x (median noise percentile)."""
    frame = pd.DataFrame(lo + x * (hi - lo), columns=hp).assign(
        **fixed, mae_rank=0.5, runtime_rank=0.5
    )
    return pd.DataFrame(
        {
            o: surrogate[o].predict(frame[surrogate[o].feature_names_in_])
            for o in ["mae", "runtime", "model_size"]
        }
    )


def space_constraints(x):
    """x6 + x15 <= 1 (reflect across the diagonal) and x19 <= x20 (sort)."""
    x = x.copy()
    flip = x[:, i_emb] + x[:, i_pos] > 1
    x[flip, i_emb], x[flip, i_pos] = 1 - x[flip, i_pos], 1 - x[flip, i_emb]
    x[:, [i_b1, i_b2]] = np.sort(x[:, [i_b1, i_b2]], axis=1)
    return x


def inverted(c):
    x = c.copy()
    x[:, i_alpha] = 1 - c[:, i_alpha]
    x[:, i_emb] = 1 - c[:, i_emb] - c[:, i_pos]
    return space_constraints(x)


elements = "Al Co Cr Cu Fe Hf Mg Mn Mo Nb Ni Re Si Sn Ta Ti V W Zn Zr".split()
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
e_elem = space_constraints(table.to_numpy()[:, np.random.default_rng(0).permutation(20)])
mappings = {
    "identity": space_constraints,
    "inverted": inverted,
    "elemental mixing": lambda c: c @ e_elem,
}

# %% MP alloy compositions (3 or more elements, lowest-energy polymorph per formula)
mp = pq.read_table(
    "data/external/materials_project/summary_2026-04-13.parquet",
    columns=[
        "formula_pretty", "composition", "elements", "energy_above_hull",
        "formation_energy_per_atom", "density", "deprecated",
    ],
).to_pandas()  # fmt: skip
mp = mp[~mp["deprecated"] & mp["formation_energy_per_atom"].notna()]
mp = mp[mp["elements"].apply(lambda s: len(s) >= 3 and set(s) <= set(elements))]
mp = mp.sort_values("energy_above_hull").drop_duplicates("formula_pretty")
w_mp = pd.DataFrame([dict(c) for c in mp["composition"]]).reindex(columns=elements)
w_mp = w_mp.fillna(0.0).to_numpy()
w_mp = w_mp / w_mp.sum(1, keepdims=True)
k_mp = (w_mp > 0).sum(1)
plausible = mp["energy_above_hull"].to_numpy() <= 0.1
w_real = w_mp[plausible]


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


# %% problems: objective y, capped quantity z, cap; caps from the random pool
pool_c = draw(100_000, np.random.default_rng(99))
settings = [("none", None, 1.0)] + [
    (f"{con} cap, {q:.0%} meet it", con, q)
    for con in ["runtime", "model_size"]
    for q in (0.25, 0.10)
]
problems = {}
for name, f in mappings.items():
    cont, real = predict(f(pool_c)), predict(f(w_real))
    for label, con, q in settings:
        for where, pool, x_pool in [("simplex", cont, pool_c), ("MP compositions", real, w_real)]:
            z = pool[con].to_numpy() if con else np.ones(len(pool))
            problems[(where, f"CrabNet MAE, {name}", label)] = {
                "f": lambda c, f=f, con=con: predict(f(c))[["mae", con or "mae"]].to_numpy().T,
                "y": pool["mae"].to_numpy(),
                "z": z,
                "cap": np.quantile(z, q) if con else np.inf,
                "x": x_pool,
            }  # fmt: skip
y_mp, z_mp = mp["formation_energy_per_atom"].to_numpy(), mp["density"].to_numpy()
for label, q in [("none", 1.0), ("density cap, 25% meet it", 0.25), ("density cap, 10% meet it", 0.10)]:
    z = z_mp[plausible] if q < 1 else np.ones(plausible.sum())
    problems[("MP compositions", "MP formation energy", label)] = {
        "y": y_mp[plausible],
        "z": z,
        "cap": np.quantile(z, q) if q < 1 else np.inf,
        "x": w_real,
    }  # fmt: skip
n_init, budget, n_forest, n_random = 10, 100, 8, 50


def acquire(C, y, z, cap, cand, seed):
    """Lowest confidence bound among candidates that half the trees put under the cap."""
    rf = RandomForestRegressor(30, max_features=0.5, random_state=seed).fit(C, y)
    per_tree = np.stack([t.predict(cand) for t in rf.estimators_])
    score = per_tree.mean(0) - per_tree.std(0)
    if np.isfinite(cap):
        rf_z = RandomForestRegressor(30, max_features=0.5, random_state=seed).fit(C, np.log(z))
        p_ok = (np.stack([t.predict(cand) for t in rf_z.estimators_]) <= np.log(cap)).mean(0)
        score = np.where(p_ok >= 0.5, score, np.inf) if (p_ok >= 0.5).any() else -p_ok
    return np.argmin(score)


def best_so_far(y, z, cap):
    return np.minimum.accumulate(np.where(z <= cap, y, np.inf))


def forest_simplex(p, seed):
    gen = np.random.default_rng(seed)
    C = draw(n_init, gen)
    y, z = p["f"](C)
    while len(y) < budget:
        ok = z <= p["cap"]
        inc = C[ok][y[ok].argmin()] if ok.any() else C[z.argmin()]
        cand = np.vstack([draw(1_000, gen), local_moves(inc, 500, gen)])
        c_new = cand[[acquire(C, y, z, p["cap"], cand, seed)]]
        y_new, z_new = p["f"](c_new)
        C, y, z = np.vstack([C, c_new]), np.append(y, y_new), np.append(z, z_new)
    return best_so_far(y, z, p["cap"])


def forest_pool(p, seed):
    gen = np.random.default_rng(seed)
    seen = list(gen.choice(len(p["y"]), n_init, replace=False))
    while len(seen) < budget:
        rest = np.setdiff1d(np.arange(len(p["y"])), seen)
        i = acquire(p["x"][seen], p["y"][seen], p["z"][seen], p["cap"], p["x"][rest], seed)
        seen.append(rest[i])
    return best_so_far(p["y"][seen], p["z"][seen], p["cap"])


def random_simplex(p, seed):
    y, z = p["f"](draw(budget, np.random.default_rng(1_000 + seed)))
    return best_so_far(y, z, p["cap"])


def random_pool(p, seed):
    idx = np.random.default_rng(1_000 + seed).permutation(len(p["y"]))[:budget]
    return best_so_far(p["y"][idx], p["z"][idx], p["cap"])


# %% run
keys = list(problems)
jobs = [(k, "forest", s) for k in keys for s in range(n_forest)]
jobs += [(k, "random", s) for k in keys for s in range(n_random)]


def run(key, opt, seed):
    simplex = key[0] == "simplex"
    if opt == "random":
        return (random_simplex if simplex else random_pool)(problems[key], seed)
    return (forest_simplex if simplex else forest_pool)(problems[key], seed)


out = Parallel(n_jobs=4, prefer="threads")(delayed(run)(*j) for j in jobs)
traces = {}
for (key, opt, _), v in zip(jobs, out):
    traces.setdefault(key, {}).setdefault(opt, []).append(v)

rows, curves = [], {}
for key, p in problems.items():
    ok = p["z"] <= p["cap"]
    y_ok = p["y"][ok]
    runs = {m: np.array(v) for m, v in traces[key].items()}
    f_best = min(y_ok.min(), runs["forest"].min())
    med = np.median(y_ok)
    closed = {  # before the first point under the cap: the worst pool value under it
        m: (med - np.where(np.isinf(v), y_ok.max(), v)) / (med - f_best)
        for m, v in runs.items()
    }
    curves[key] = closed
    rho = spearmanr(p["y"], p["z"])[0] if np.isfinite(p["cap"]) else np.nan
    for m, v in closed.items():
        for n in (10, 25, 50, 100):
            rows.append(
                {
                    "setting": key[0],
                    "problem": key[1],
                    "cap": key[2],
                    "optimizer": m,
                    "evaluations": n,
                    "gap_closed": np.median(v[:, n - 1]),
                    "gap_closed_q25": np.percentile(v[:, n - 1], 25),
                    "gap_closed_q75": np.percentile(v[:, n - 1], 75),
                    "cap_value": p["cap"],
                    "share_meeting_cap": ok.mean(),
                    "spearman_objective_capped": rho,
                    "median_meeting_cap": med,
                    "best_known_meeting_cap": f_best,
                    "best_known_overall": p["y"].min(),
                }
            )
walk = pd.DataFrame(rows)
walk.to_csv(fig_dir / "capped_walk_test.csv", index=False)
at_100 = walk.query("evaluations == 100").pivot_table(
    index=["setting", "problem", "cap"], columns="optimizer", values="gap_closed", sort=False
)
print(at_100.round(2).to_string())

# %% figure: share of the gap closed after 100 evaluations, random vs forest
ink, muted, band, blue = "#0b0b0b", "#898781", "#f0efec", "#2a78d6"
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
    ("simplex", f"CrabNet MAE, {m}", f"({a}) CrabNet MAE, {m}") for a, m in zip("abc", mappings)
] + [
    ("MP compositions", f"CrabNet MAE, {m}", f"({a}) CrabNet MAE, {m}") for a, m in zip("def", mappings)
] + [("MP compositions", "MP formation energy", "(g) MP formation energy (DFT)")]  # fmt: skip
fig, axes = plt.subplots(2, 4, figsize=(14, 5.4), sharex=True, layout="constrained")
for ax, (where, prob, title) in zip([*axes[0, :3], *axes[1]], panels):
    sub = walk.query("setting == @where and problem == @prob and evaluations == 100")
    caps = list(dict.fromkeys(sub["cap"]))
    for m, c, dy, label in [("random", muted, 0.14, "random search"), ("forest", blue, -0.14, "random-forest search")]:  # fmt: skip
        s = sub[sub["optimizer"] == m].set_index("cap").loc[caps]
        yy = np.arange(len(caps)) + dy
        ax.hlines(yy, s["gap_closed_q25"], s["gap_closed_q75"], color=c, lw=2, alpha=0.45)
        ax.plot(s["gap_closed"], yy, "o", color=c, ms=7, mec="white", mew=1, label=label)
    labels = [c.replace("model_size", "model size").replace(" cap,", ":") for c in caps]
    ax.set_yticks(range(len(caps)), ["no cap" if c == "none" else c for c in labels])
    if ax not in (axes[0, 0], axes[1, 0], axes[1, 3]):
        ax.tick_params(axis="y", labelleft=False)
    ax.set_ylim(4.5, -0.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_title(title, loc="left")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=band, lw=0.8)
    ax.set_axisbelow(True)
axes[0, 3].axis("off")
axes[0, 3].legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left", frameon=False)
axes[0, 3].text(
    0.02, 0.62,
    "Top row: compositions anywhere on the\nsimplex (the benchmark as it would be used).\n\n"
    "Bottom row: the optimizers choose among\n2,239 real MP compositions, so CrabNet\n"
    "and MP get identical inputs.\n\n"
    "0 is the median random composition that\nmeets the cap, 1 the best known one.\n"
    "Dots: medians over 50 random and 8 forest\nruns. Bars: 25th to 75th percentiles.",
    va="top", fontsize=8.5, color=ink, transform=axes[0, 3].transAxes,
)  # fmt: skip
fig.supxlabel("share of the gap closed after 100 evaluations", fontsize=9.5, color=ink)
fig.savefig(fig_dir / "capped_walk_test.png", dpi=200, facecolor="white")
