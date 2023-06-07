# %% [markdown]
# Run a simple test using the PseudoCrabMinimal benchmark. Note that the sum of the
# parameters needs to equal 1.0.

# %%
import numpy as np
from ax.service.ax_client import AxClient
from skopt import gp_minimize
from skopt.plots import plot_convergence

from matsci_opt_benchmarks.crabnet_hyperparameter.core import PseudoCrabBasic
from matsci_opt_benchmarks.crabnet_hyperparameter.core.utils.plotting import line

# %%
dummy = False
pc = PseudoCrabBasic(dummy=dummy, model_dir="models/crabnet_hyperparameter")

# %%
results = pc.evaluate({"x1": 0.5, "x2": 0.2, "x3": 0.3}, dummy=dummy)

# %%
results

# %% [markdown]
# Next, we'll use [Ax](https://ax.dev) to minimize the MAE. We'll reparameterize the
# following constraint:
# $x_1 + x_2 + x_3 = 1.0$
#
# into:
#
# $x_1 + x_2 \le 1.0$ and $x_3 = x_1 + x_2 - 1.0$
#
# and remove $x_3$ from the search space by calculating it from $x_1$ and $x_2$ in the
# `evaluate` function.

# %%

ax_dfs = []
for i in range(3):
    ax_client = AxClient()
    ax_client.create_experiment(
        name="pseudo_crab",
        parameters=[
            {"name": "x1", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "x2", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "x3", "type": "range", "bounds": [0.0, 1.0]},
        ],
        objective_name="mae",
        minimize=True,
    )

    for _ in range(15):
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data=pc.evaluate(parameters),
        )

    best_parameters, metrics = ax_client.get_best_parameters()

    # %%
    best_parameters, metrics

    # %%
    from ax.utils.notebook.plotting import init_notebook_plotting, render

    init_notebook_plotting()

    # %%
    render(ax_client.get_optimization_trace())

    # %%
    ax_df = ax_client.get_trials_data_frame()[["x1", "x2", "x3", "mae"]]
    ax_df["type"] = "ax"
    ax_dfs.append(ax_df)

# %%


def f(x):
    return pc.evaluate({"x1": x[0], "x2": x[1], "x3": x[2]}, dummy=dummy)["mae"]


sk_dfs = []
for i in range(3):
    res = gp_minimize(
        f,  # the function to minimize
        [(0.0, 1.0)] * 3,  # the bounds on each dimension of x
        acq_func="EI",  # the acquisition function
        n_calls=15,  # the number of evaluations of f
        n_initial_points=6,  # the number of random initialization points
        random_state=1234,
    )  # the random seed

    # %%
    res

    # %%

    plot_convergence(res)

    # %%
    import pandas as pd

    sk_df = pd.concat(
        [
            pd.DataFrame(res.x_iters, columns=["x1", "x2", "x3"]),
            pd.DataFrame(res.func_vals, columns=["mae"]),
        ],
        axis=1,
    )
    sk_df["type"] = "scikit-optimize"
    sk_dfs.append(sk_df)

# %%
# compute the cumulative minimum for each df
for df in ax_dfs:
    df["best_mae"] = df["mae"].cummin()

for df in sk_dfs:
    df["best_mae"] = df["mae"].cummin()

# compute average and min across ax_dfs

ax_final = pd.DataFrame(
    [
        list(np.mean([ax_df["best_mae"] for ax_df in ax_dfs], axis=0)),
        list(range(0, 15)),
        list(np.std([ax_df["best_mae"] for ax_df in ax_dfs], axis=0)),
    ],
).T
ax_final.columns = ["best_mae", "iteration", "std"]

sk_final = pd.DataFrame(
    [
        list(np.mean([sk_df["best_mae"] for sk_df in sk_dfs], axis=0)),
        list(range(0, 15)),
        list(np.std([sk_df["best_mae"] for sk_df in sk_dfs], axis=0)),
    ],
).T
sk_final.columns = ["best_mae", "iteration", "std"]

ax_final["best_mae"]
sk_final["best_mae"]
df = pd.concat([ax_final, sk_final], axis=0)
df = df.reset_index()
# df["best_mae"] = df.groupby("type")["mae"].cummin()
df

# %%

line(df=df, x="index", y="best_mae", color="type", title="Pseudo Crab Optimization")

1 + 1

# %%
