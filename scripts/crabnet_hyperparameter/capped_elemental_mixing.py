"""Try the capped elemental-mixing variant of the sum-to-one CrabNet benchmark with Ax.

The problem is the runtime-capped setting of capped_walk_test.py with the clearest gap
between model-based and random search:

- Inputs: fractions w1 to w20 of Al Co Cr Cu Fe Hf Mg Mn Mo Nb Ni Re Si Sn Ta Ti V W
  Zn Zr, summing to one with at least 3 nonzero.
- Mapping: x = w E. Row i of E holds 20 min-max scaled properties of element i, in
  property assignment 0 of composition_mappings.py. E is written to
  capped_elemental_mixing_E.csv (rows: elements; columns: the Space inputs x1 to x20,
  labeled by hyperparameter and property), so the mapping can be used without pymatgen.
  Entries in columns x6, x15, x19 and x20 were reflected or swapped where needed so
  every row meets x6 + x15 <= 1 and x19 <= x20. Both hold for every composition.
- Objective: minimize the MAE of the Space surrogate at x (train_frac = 1, RobustL1,
  mat2vec, no bias, median noise percentile).
- Constraint: runtime <= 157 s, which 25% of random compositions meet.

Optimizers, 100 evaluations each:

- random search over compositions with as many active elements as the Materials
  Project (MP) alloy compositions (3 or 4), as in capped_walk_test.py
- random search in the box: w = u / sum(u) with u uniform on [0, 1]^20, which is the
  space Ax searches
- the random-forest search of capped_walk_test.py
- Ax 1.3 with its default generation strategy (center point, Sobol, then BoTorch) and
  the outcome constraint runtime <= 157, searching u in [0, 1]^20 with w = u / sum(u).
  A point with fewer than 3 nonzero u is marked failed and still uses up an evaluation.

Scores follow capped_walk_test.py: the share of the gap closed between the median random
composition that meets the cap and the best known one (0 and 1).

Inputs as in capped_walk_test.py (Space surrogate, MP parquet). Run with Python 3.12,
scikit-learn 1.4.1.post1, numpy<2, pandas, pyarrow, joblib, matplotlib,
pymatgen<2024.7, ax-platform 1.3.1 and torch (CPU).
"""

# %% imports
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from ax import Client, RangeParameterConfig
from ax.utils.common.logger import set_ax_logger_levels
from joblib import Parallel, delayed
from pymatgen.core import Element
from sklearn.ensemble import RandomForestRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
fig_dir = Path("reports/crabnet_hyperparameter_immi/figures")
cap, budget, n_init, n_forest, n_ax, n_random = 157.0, 100, 10, 8, 8, 50

# %% Space surrogate and E (as in capped_walk_test.py)
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
i_emb, i_pos, i_b1, i_b2 = (hp.index(p) for p in ["emb_scaler", "pos_scaler", "betas1", "betas2"])
fixed = {  # fidelity1 = 1, c1_0 (no bias), c2_0 (RobustL1), c3_0 (mat2vec)
    "train_frac": 1.0,
    "bias": 0,
    "use_RobustL1": 1,
    "elem_prop_magpie": 0,
    "elem_prop_mat2vec": 1,
    "elem_prop_onehot": 0,
}
surrogate = joblib.load("models/crabnet_hyperparameter/surrogate_models_hgbr_opt.pkl")

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
order = np.random.default_rng(0).permutation(20)
e_elem = table.to_numpy()[:, order]
flip = e_elem[:, i_emb] + e_elem[:, i_pos] > 1  # x6 + x15 <= 1 (reflect)
e_elem[flip, i_emb], e_elem[flip, i_pos] = 1 - e_elem[flip, i_pos], 1 - e_elem[flip, i_emb]
e_elem[:, [i_b1, i_b2]] = np.sort(e_elem[:, [i_b1, i_b2]], axis=1)  # x19 <= x20
pd.DataFrame(
    e_elem,
    index=pd.Index(elements, name="element"),
    columns=pd.MultiIndex.from_arrays(
        [hp, [props[j] for j in order]], names=["hyperparameter", "property"]
    ),
).round(6).to_csv(fig_dir / "capped_elemental_mixing_E.csv")


def evaluate(w):
    """MAE and runtime at compositions w (rows sum to one)."""
    frame = pd.DataFrame(lo + (w @ e_elem) * (hi - lo), columns=hp).assign(
        **fixed, mae_rank=0.5, runtime_rank=0.5
    )
    return tuple(surrogate[o].predict(frame[surrogate[o].feature_names_in_]) for o in ["mae", "runtime"])


# %% random compositions with as many active elements as the MP alloys
mp = pq.read_table(
    "data/external/materials_project/summary_2026-04-13.parquet",
    columns=["formula_pretty", "composition", "elements", "energy_above_hull",
             "formation_energy_per_atom", "deprecated"],
).to_pandas()  # fmt: skip
mp = mp[~mp["deprecated"] & mp["formation_energy_per_atom"].notna()]
mp = mp[mp["elements"].apply(lambda s: len(s) >= 3 and set(s) <= set(elements))]
mp = mp.sort_values("energy_above_hull").drop_duplicates("formula_pretty")
k_mp = mp["elements"].apply(len).to_numpy()


def draw(n, gen):
    k = gen.choice(k_mp, n)
    active = np.argsort(np.argsort(gen.random((n, 20)), axis=1), axis=1) < k[:, None]
    g = gen.exponential(size=(n, 20)) * active
    return g / g.sum(1, keepdims=True)


def draw_box(n, gen):
    u = gen.random((n, 20))
    return u / u.sum(1, keepdims=True)


def local_moves(best, n, gen):
    c = np.repeat(best[None], n, 0)
    i = gen.choice(np.flatnonzero(best > 0), n)
    d = gen.random(n) * best[i]
    c[np.arange(n), i] -= d
    c[np.arange(n), gen.integers(20, size=n)] += d
    return c[(c > 1e-9).sum(1) >= 3]


def best_so_far(y, z):
    """Best MAE meeting the cap after each evaluation, and the share meeting it."""
    return np.minimum.accumulate(np.where(z <= cap, y, np.inf)), np.mean(z <= cap)


# %% optimizers
def random_search(seed, sampler):
    return best_so_far(*evaluate(sampler(budget, np.random.default_rng(1_000 + seed))))


def forest_search(seed):
    """Lowest confidence bound among candidates that half the trees put under the cap."""
    gen = np.random.default_rng(seed)
    C = draw(n_init, gen)
    y, z = evaluate(C)
    while len(y) < budget:
        ok = z <= cap
        inc = C[ok][y[ok].argmin()] if ok.any() else C[z.argmin()]
        cand = np.vstack([draw(1_000, gen), local_moves(inc, 500, gen)])
        rf = RandomForestRegressor(30, max_features=0.5, random_state=seed).fit(C, y)
        per_tree = np.stack([t.predict(cand) for t in rf.estimators_])
        rf_z = RandomForestRegressor(30, max_features=0.5, random_state=seed).fit(C, np.log(z))
        p_ok = (np.stack([t.predict(cand) for t in rf_z.estimators_]) <= np.log(cap)).mean(0)
        score = per_tree.mean(0) - per_tree.std(0)
        score = np.where(p_ok >= 0.5, score, np.inf) if (p_ok >= 0.5).any() else -p_ok
        c_new = cand[[np.argmin(score)]]
        y_new, z_new = evaluate(c_new)
        C, y, z = np.vstack([C, c_new]), np.append(y, y_new), np.append(z, z_new)
    return best_so_far(y, z)


def ax_search(seed):
    set_ax_logger_levels(logging.WARNING)
    torch.set_num_threads(1)
    client = Client(random_seed=seed)
    client.configure_experiment(
        parameters=[
            RangeParameterConfig(name=f"u{i}", parameter_type="float", bounds=(0.0, 1.0))
            for i in range(1, 21)
        ]
    )
    client.configure_optimization(objective="-mae", outcome_constraints=[f"runtime <= {cap}"])
    y, z = np.full(budget, np.inf), np.full(budget, np.inf)
    for n in range(budget):
        ((trial, p),) = client.get_next_trials(max_trials=1).items()
        u = np.array([p[f"u{i}"] for i in range(1, 21)])
        if (u > 0).sum() < 3:
            client.mark_trial_failed(trial_index=trial)
            continue
        mae, runtime = evaluate(u[None] / u.sum())
        y[n], z[n] = mae[0], runtime[0]
        client.complete_trial(trial_index=trial, raw_data={"mae": float(y[n]), "runtime": float(z[n])})
    return best_so_far(y, z)


# %% run
optimizers = ["random search", "random search in the box", "random-forest search", "Ax (BoTorch)"]
runs = {
    "random search": [random_search(s, draw) for s in range(n_random)],
    "random search in the box": [random_search(s, draw_box) for s in range(n_random)],
    "random-forest search": Parallel(n_jobs=4)(delayed(forest_search)(s) for s in range(n_forest)),
    "Ax (BoTorch)": Parallel(n_jobs=4)(delayed(ax_search)(s) for s in range(n_ax)),
}
traces = {m: np.array([t for t, _ in v]) for m, v in runs.items()}
share_ok = {m: np.median([f for _, f in v]) for m, v in runs.items()}

pool_y, pool_z = evaluate(draw(100_000, np.random.default_rng(99)))
ok = pool_z <= cap
median_ok = np.median(pool_y[ok])
best_known = min(pool_y[ok].min(), min(v.min() for v in traces.values()))
rows = []
for m, v in traces.items():
    closed = (median_ok - np.where(np.isinf(v), pool_y[ok].max(), v)) / (median_ok - best_known)
    for n in range(1, budget + 1):
        rows.append(
            {
                "optimizer": m,
                "evaluations": n,
                "runs": len(v),
                "best_mae_median": np.median(v[:, n - 1]),
                "best_mae_q25": np.percentile(v[:, n - 1], 25),
                "best_mae_q75": np.percentile(v[:, n - 1], 75),
                "gap_closed": np.median(closed[:, n - 1]),
                "gap_closed_q25": np.percentile(closed[:, n - 1], 25),
                "gap_closed_q75": np.percentile(closed[:, n - 1], 75),
                "cap": cap,
                "share_of_evaluations_meeting_cap": share_ok[m],
                "share_of_random_meeting_cap": ok.mean(),
                "median_random_meeting_cap": median_ok,
                "best_known_meeting_cap": best_known,
            }
        )
summary = pd.DataFrame(rows)
summary.to_csv(fig_dir / "capped_elemental_mixing.csv", index=False)
print(f"share of random compositions meeting the cap: {ok.mean():.3f}")
print(f"median / best known MAE meeting the cap: {median_ok:.4f} / {best_known:.4f} eV")
print(summary.query("evaluations in [10, 25, 50, 100]").round(3).to_string(index=False))

# %% figure
ink, muted, band = "#0b0b0b", "#898781", "#f0efec"
colors = dict(zip(optimizers, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
plt.rcParams.update(
    {
        "font.size": 9.5,
        "axes.edgecolor": muted,
        "axes.labelcolor": ink,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": muted,
        "ytick.color": muted,
        "xtick.labelcolor": ink,
        "ytick.labelcolor": ink,
    }
)
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained", width_ratios=[1.4, 1])
for m in optimizers:
    s = summary[summary["optimizer"] == m].replace(np.inf, np.nan)
    a.fill_between(s["evaluations"], s["best_mae_q25"], s["best_mae_q75"], color=colors[m], alpha=0.15, lw=0)
    a.plot(s["evaluations"], s["best_mae_median"], color=colors[m], lw=2, label=m)
a.axhline(best_known, color=muted, lw=1, ls=":")
a.text(budget, best_known, "best known", color=ink, fontsize=8.5, ha="right", va="bottom")
a.set_ylim(best_known - 0.01, median_ok + 0.02)
a.set_xlabel("evaluations")
a.set_ylabel("best MAE meeting the cap [eV]")
a.set_title("(a) Best MAE with runtime <= 157 s (median, 25th to 75th percentile)", loc="left")
a.legend(frameon=False, loc="upper right")
a.grid(axis="y", color=band, lw=0.8)
s100 = summary[summary["evaluations"] == budget].set_index("optimizer").loc[optimizers]
yy = np.arange(len(optimizers))
b.hlines(yy, s100["gap_closed_q25"], s100["gap_closed_q75"], color=[colors[m] for m in optimizers], lw=2, alpha=0.5)
b.scatter(s100["gap_closed"], yy, c=[colors[m] for m in optimizers], s=60, edgecolors="white", linewidths=1, zorder=3)
for y_, (m, r) in zip(yy, s100.iterrows()):
    b.text(r["gap_closed_q75"] + 0.03, y_, f"{r['gap_closed']:.2f} ({int(r['runs'])} runs)", va="center", fontsize=8.5, color=ink)
b.set_yticks(yy, optimizers)
b.set_ylim(len(optimizers) - 0.5, -0.5)
b.set_xlim(-0.05, 1.3)
b.set_xticks([0, 0.25, 0.5, 0.75, 1])
b.tick_params(axis="y", length=0)
b.spines["left"].set_visible(False)
b.grid(axis="x", color=band, lw=0.8)
b.set_axisbelow(True)
b.set_xlabel("share of the gap closed after 100 evaluations")
b.set_title("(b) 0 = median random composition meeting the cap, 1 = best known", loc="left")
fig.savefig(fig_dir / "capped_elemental_mixing.png", dpi=200, facecolor="white")
