"""Custom plotting utils for Adaptive Experimentation platform.

Modified from source: https://github.com/sparks-baird/crabnet-hyperparameter
"""
import collections
import pprint
from os import path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from ax.modelbridge import ModelBridge
from ax.plot.base import AxPlotConfig
from ax.plot.helper import compose_annotation
from ax.plot.trace import optimization_trace_single_method_plotly
from ax.utils.common.logger import get_logger
from boppf.utils.data import prep_input_data
from plotly import offline
from scipy.interpolate import interp1d

logger = get_logger(__name__)


def matplotlibify(fig, size=24, width_inches=3.5, height_inches=3.5, dpi=142):
    # make it look more like matplotlib
    # modified from:
    # https://medium.com/swlh/formatting-a-plotly-figure-with-matplotlib-style-fa56ddd97539)# noqa: E501
    family = "Arial"
    color = "black"
    font_dict = dict(family=family, size=size, color=color)

    fig.update_layout(
        font=font_dict,
        plot_bgcolor="white",
        width=width_inches * dpi,
        height=height_inches * dpi,
        margin=dict(r=40, t=20, b=10),
    )

    fig.update_yaxes(
        showline=True,  # add line at x=0
        linecolor="black",  # line color
        linewidth=2.4,  # line size
        ticks="inside",  # ticks outside axis
        tickfont=font_dict,  # tick label font
        mirror="allticks",  # add ticks to top/right axes
        tickwidth=2.4,  # tick width
        tickcolor="black",  # tick color
    )

    fig.update_xaxes(
        showline=True,
        showticklabels=True,
        linecolor="black",
        linewidth=2.4,
        ticks="inside",
        tickfont=font_dict,
        mirror="allticks",
        tickwidth=2.4,
        tickcolor="black",
    )
    fig.update(layout_coloraxis_showscale=False)
    fig.update_coloraxes(colorbar_tickfont=font_dict)
    for d in fig.data:
        if isinstance(d, go.Contour):
            d.colorbar.tickfont.size = size
            d.colorbar.tickfont.family = family
            d.colorbar.tickfont.color = color

    for a in fig.layout.annotations:
        a.font = font_dict

    width_default_px = fig.layout.width
    targ_dpi = 300
    scale = width_inches / (width_default_px / dpi) * (targ_dpi / dpi)

    return fig, scale


def my_plot_feature_importance_by_feature_plotly(
    model: ModelBridge = None,
    feature_importances: dict = None,
    error_x: dict = None,
    metric_names: Iterable[str] = None,
    relative: bool = True,
    caption: str = "",
) -> go.Figure:
    """One plot per metric, showing importances by feature.

    Args:
        model: A model with a ``feature_importances`` method.
        relative: whether to normalize feature importances so that they add to 1.
        caption: an HTML-formatted string to place at the bottom of the plot.

    Returns a go.Figure of feature importances.

    Notes:
        Copyright (c) Facebook, Inc. and its affiliates.

        This source code is licensed under the MIT license. Modifed by @sgbaird.
    """
    traces = []
    dropdown = []
    if metric_names is None:
        assert model is not None, "specify model or metric_names"
        metric_names = model.metric_names
    assert metric_names is not None, "specify model or metric_names"
    for i, metric_name in enumerate(sorted(metric_names)):
        try:
            if feature_importances is not None:
                importances = feature_importances
            else:
                assert model is not None, "model is None"
                importances = model.feature_importances(metric_name)
        except NotImplementedError:
            logger.warning(
                f"Model for {metric_name} does not support feature importances."
            )
            continue
        factor_col = "Factor"
        importance_col = "Importance"
        std_col = "StdDev"
        low_col = "err_minus"
        assert error_x is not None, "specify error_x"
        df = pd.DataFrame(
            [
                {factor_col: factor, importance_col: importance}
                for factor, importance in importances.items()
            ]
        )
        err_df = pd.Series(error_x).to_frame(name=std_col)
        err_df.index.names = [factor_col]
        df = pd.concat((df.set_index(factor_col), err_df), axis=1).reset_index()

        if relative:
            totals = df[importance_col].sum()
            df[importance_col] = df[importance_col].div(totals)
            df[std_col] = df[std_col].div(totals)

        low_df = df[std_col]
        low_df[low_df > df[importance_col]] = df[importance_col]
        df[low_col] = low_df

        df = df.sort_values(importance_col)
        traces.append(
            go.Bar(
                name=importance_col,
                orientation="h",
                visible=i == 0,
                x=df[importance_col],
                y=df[factor_col],
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=df[std_col].to_list(),
                    arrayminus=df[low_col].to_list(),
                ),
            )
        )

        is_visible = [False] * len(sorted(metric_names))
        is_visible[i] = True
        dropdown.append(
            {"args": ["visible", is_visible], "label": metric_name, "method": "restyle"}
        )
    if not traces:
        raise NotImplementedError("No traces found for metric")

    updatemenus = [
        {
            "x": 0,
            "y": 1,
            "yanchor": "top",
            "xanchor": "left",
            "buttons": dropdown,
            "pad": {
                "t": -40
            },  # hack to put dropdown below title regardless of number of features
        }
    ]
    features = traces[0].y
    title = (
        "Relative Feature Importances" if relative else "Absolute Feature Importances"
    )
    layout = go.Layout(
        height=200 + len(features) * 20,
        hovermode="closest",
        margin=go.layout.Margin(
            l=8 * min(max(len(idx) for idx in features), 75)  # noqa E741
        ),
        showlegend=False,
        title=title,
        updatemenus=updatemenus,
        annotations=compose_annotation(caption=caption),
    )

    if relative:
        layout.update({"xaxis": {"tickformat": ".0%"}})

    fig = go.Figure(data=traces, layout=layout)

    return fig


def my_round(x):
    if isinstance(x, str):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if np.round(x, 4) == 0:
            return np.format_float_scientific(x, precision=2, unique=False, trim="k")
        else:
            return np.format_float_positional(
                x, precision=4, unique=False, fractional=False, trim="k"
            )
    if isinstance(x, list):
        if isinstance(x[0], int):
            return x
        else:
            if np.round(x[0], 4) == 0:
                return [
                    np.format_float_scientific(a, precision=2, unique=False, trim=".")
                    for a in x
                ]
            else:
                return [
                    np.format_float_positional(
                        a, precision=4, unique=False, fractional=False, trim="."
                    )
                    for a in x
                ]


def df_to_rounded_csv(df, save_dir="results", save_name="rounded.csv"):
    print_df = df.applymap(my_round)
    fpath = path.join(save_dir, save_name)
    print_df.to_csv(fpath)


def plot_and_save(fig_path, fig, mpl_kwargs={}, show=False, update_legend=False):
    if show:
        offline.plot(fig)
    fig.write_html(fig_path + ".html")
    fig.to_json(fig_path + ".json")
    if update_legend:
        fig.update_layout(
            legend=dict(
                font=dict(size=16),
                yanchor="bottom",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(0,0,0,0)",
                # orientation="h",
            )
        )
    fig, scale = matplotlibify(fig, **mpl_kwargs)
    fig.write_image(fig_path + ".png")


def my_optimization_trace_single_method_plotly(
    experiment,
    ylabel="target",
    optimization_direction="minimize",
    plot_trial_points=True,
):
    trials = experiment.trials.values()

    best_objectives = np.array([[trial.objective_mean for trial in trials]])

    parameter_strs = [
        pprint.pformat(trial.arm.parameters).replace("\n", "<br>") for trial in trials
    ]

    best_objective_plot = optimization_trace_single_method_plotly(
        y=best_objectives,
        optimization_direction=optimization_direction,
        ylabel=ylabel,
        hover_labels=parameter_strs,
        plot_trial_points=plot_trial_points,
    )
    return best_objective_plot


def my_std_optimization_trace_single_method_plotly(
    experiments,
    ylabel="target",
    optimization_direction="minimize",
    plot_trial_points=False,
):
    best_objectives = []
    for experiment in experiments:
        trials = experiment.trials.values()
        best_objective = [trial.objective_mean for trial in trials]
        best_objectives.append(best_objective)

    best_objectives = np.array(best_objectives)

    best_objective_plot = optimization_trace_single_method_plotly(
        y=best_objectives,
        optimization_direction=optimization_direction,
        ylabel=ylabel,
        plot_trial_points=plot_trial_points,
    )

    return best_objective_plot


def to_plotly(axplotconfig):
    assert isinstance(axplotconfig, AxPlotConfig), "not an AxPlotConfig"
    data = axplotconfig[0]["data"]
    layout = axplotconfig[0]["layout"]
    fig = go.Figure({"data": data, "layout": layout})
    return fig


def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def plot_distribution(means, stds, fractions, tol=1e-6, size=33):
    s_radii, _, m_fracs = prep_input_data(means, stds, fractions, tol, size)

    s_flat = flatten(s_radii)
    samples = np.linspace(min(s_flat), max(s_flat), 1000)

    fns = [
        interp1d(sr, mf, kind="cubic", bounds_error=False, fill_value=0.0)
        for sr, mf in zip(s_radii, m_fracs)
    ]

    probs = np.sum([f(samples) for f in fns], axis=0)

    dfs = []
    n_modes = len(s_radii)
    for i, (s_mode_radii, m_mode_fracs) in enumerate(zip(s_radii, m_fracs)):
        mode = i + 1
        mode_str = f"scale={means[i]:.2f},s={stds[i]:.2f},p={np.sum(m_mode_fracs):.2f}"
        if mode == n_modes:
            mode_str = "<br>".join([mode_str, *[""] * (3 - n_modes)])
        df = pd.DataFrame(
            {
                "mode": mode_str,
                "s_radii": s_mode_radii,
                "m_fracs": m_mode_fracs,
            }
        )
        dfs.append(df)
    main_df = pd.concat(dfs, axis=0, join="inner")

    fig = px.scatter(main_df, x="s_radii", y="m_fracs", color="mode")
    fig.update_layout(legend=dict(title_text=""))
    fig.add_scatter(
        x=samples, y=probs, name="interpolated and summed", line=dict(color="black")
    )

    return fig


# def my_std_optimization_trace_single_method_plotly(
#     experiments,
#     ylabel="target",
#     optimization_direction="minimize",
#     plot_trial_points=True,
# ):
#     best_objectives = []
#     for experiment in experiments:
#         trials = experiment.trials.values()
#         best_objective = np.array([[trial.objective_mean for trial in trials]])
#         best_objectives.append(best_objective)
#     best_objectives_mean = np.mean(best_objectives, axis=1)
#     best_objectives_std = np.std(best_objectives, axis=1)

#     n_trials = len(best_objectives_mean)
#     x = list(range(1, n_trials + 1))
#     y = best_objectives_mean
#     y_lower = y - best_objectives_std
#     y_upper = y + best_objectives_std

#     best_objective_plot = go.Figure(
#         [
#             go.Scatter(x=x, y=y, line=dict(color="rgb(0,100,80)"), mode="lines"),
#             go.Scatter(
#                 x=x + x[::-1],  # x, then x reversed
#                 y=y_upper + y_lower[::-1],  # upper, then lower reversed
#                 fill="toself",
#                 fillcolor="rgba(0,100,80,0.2)",
#                 line=dict(color="rgba(255,255,255,0)"),
#                 hoverinfo="skip",
#                 showlegend=False,
#             ),
#         ]
#     )
#     return best_objectives_mean, best_objectives_std, best_objective_plot


# def to_plotly(axplotconfig):
#     if isinstance(axplotconfig, AxPlotConfig):
#         data = axplotconfig[0]["data"]
#         layout = axplotconfig[0]["layout"]
#         fig = go.Figure({"data": data, "layout": layout})
#     elif isinstance(axplotconfig, go.Figure):
#         warn("Figure is already a Plotly Figure")
#         # TODO: check to make sure it's a plotly figure
#         fig = axplotconfig
#     else:
#         raise ValueError("not an AxPlotConfig nor a Plotly Figure")
#     return fig

# weighted_radii = np.multiply(s_mode_radii, m_mode_fracs)

# probs = [
#     lognorm.pdf(
#         samples[np.all([samples >= min(sr), samples <= max(sr)])], s, scale=scale,
#     )
#     for s, scale, sr in zip(stds, means, s_radii)
# ]
# summed_probs = np.sum(probs, axis=0)

# probs = []
# fracs = fractions + [1 - sum(fractions)]
# for sample in samples:
#     prob = 0.0
#     for s, scale, sr, mf in zip(stds, means, s_radii, fracs):
#         if sample >= min(sr) and sample <= max(sr):
#             prob += mf * lognorm.pdf(sample, s, scale=scale)
#     probs.append(prob)
