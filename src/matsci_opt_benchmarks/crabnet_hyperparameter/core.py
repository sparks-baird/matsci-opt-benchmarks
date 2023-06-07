"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = crabnet_hyperparameter.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note:
    This file can be renamed depending on your needs or safely removed if not needed.

References:
    - https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""

import argparse
import logging
import sys
from os import path
from typing import List, Optional

import numpy as np

from matsci_opt_benchmarks.crabnet_hyperparameter import __version__
from matsci_opt_benchmarks.crabnet_hyperparameter.utils.parameters import (
    CrabNetSurrogateModel,
    matbench_metric_calculator,
    userparam_to_crabnetparam,
)
from matsci_opt_benchmarks.crabnet_hyperparameter.utils.validation import (
    check_float_ranges,
    sum_constraint_fn,
)

__author__ = "sgbaird"
__copyright__ = "sgbaird"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from matsci_opt_benchmarks.crabnet_hyperparameter.skeleton import fib`,
# when using this Python module as a library.


def fib(n):
    """Fibonacci example function

    Args:
      n (int): integer

    Returns:
      int: n-th Fibonacci number
    """
    assert n > 0
    a, b = 1, 1
    for _i in range(n - 1):
        a, b = b, a + b
    return a


SUPPORTED_OBJECTIVES = ["mae", "rmse", "runtime", "model_size"]
FLOAT_LIMIT = 20
CATEGORICAL_LIMIT = 3

# def evaluate(parameters):
#     results = matbench_metric_calculator(parameters)

#     outputs = {
#         "mae": results["average_mae"],
#         "rmse": results["average_rmse"],
#         "model_size": results["model_size"],
#         "runtime": results["runtime"],
#     }

#     return outputs


class PseudoCrab(object):
    def __init__(
        self,
        objectives: List[str] = ["mae"],
        iteration_budget: int = 100,
        n_float_params: int = 3,
        categorical_num_options: List[int] = [2, 2, 3],
        constraint_fn: Optional[callable] = sum_constraint_fn,
        surrogate=True,
        model_dir=None,
        dummy=False,
    ):
        self.dummy = dummy
        if model_dir is None:
            model_dir = path.join("..", "..", "models", "crabnet_hyperparameter")
            if self.dummy:
                model_dir = path.join(model_dir, "dummy")

        for obj in objectives:
            assert (
                obj in SUPPORTED_OBJECTIVES
            ), f"Unsupported objective: {obj}. Must be in {SUPPORTED_OBJECTIVES}"

        self.objectives = objectives
        self.iteration_budget = iteration_budget

        # TODO: consider distinguishing between float and int parameters
        if n_float_params > FLOAT_LIMIT:
            raise ValueError(
                f"{n_float_params} float parameters requested. No more than {FLOAT_LIMIT} allowed."  # noqa: E501
            )
        self.n_float_params = n_float_params
        self.n_categorical_params = len(categorical_num_options)

        if self.n_categorical_params > CATEGORICAL_LIMIT:
            raise ValueError(
                f"{self.n_categorical_params} categorical parameters requested. No more than {CATEGORICAL_LIMIT} allowed."  # noqa: E501
            )

        self.constraint_fn = (
            constraint_fn if constraint_fn is not None else lambda x: True
        )

        self.n_objectives = len(objectives)
        self.expected_float_keys = [f"x{i}" for i in range(1, self.n_float_params + 1)]
        self.expected_categorical_keys = [
            f"c{i}" for i in range(1, self.n_categorical_params + 1)
        ]
        self.expected_keys = self.expected_float_keys + self.expected_categorical_keys

        self.__num_evaluations = 0

        if surrogate:
            self.crabnet_surrogate = CrabNetSurrogateModel(model_dir=model_dir)
        else:
            self.crabnet_surrogate = None

    @property
    def num_evaluations(self):
        return self.__num_evaluations

    def evaluate(self, parameters):
        constraint_satisfied = self.constraint_fn(parameters)
        if not constraint_satisfied:
            # REVIEW: whether to raise a ValueError or return NaN outputs?
            raise ValueError(
                f"constraint_fn ({getattr(self.constraint_fn, '__name__', 'Unknown')}) not satisfied. Evaluation not counted towards budget."  # noqa: E501
            )

        check_float_ranges(parameters)

        self.__num_evaluations = self.num_evaluations + 1

        if self.num_evaluations > self.iteration_budget:
            raise ValueError("maximum number of evaluations has been reached")

        keys = list(parameters.keys())

        err_msg = ""
        missing_keys = np.setdiff1d(keys, self.expected_keys)
        if missing_keys:
            err_msg = err_msg + f"missing keys in parameters: {missing_keys}. "

        extra_keys = np.setdiff1d(self.expected_keys, keys)
        if extra_keys:
            err_msg = err_msg + f"extra keys in parameters: {extra_keys}. "

        if err_msg != "":
            raise KeyError(err_msg)

        crabnet_parameters = userparam_to_crabnetparam(parameters)

        results = matbench_metric_calculator(
            crabnet_parameters, surrogate=self.crabnet_surrogate, dummy=self.dummy
        )  # add try except block

        # # TODO: compute and return CrabNet objective(s) as dictionary
        # crabnet_mae = 0.123 # eV (dummy value)
        # crabnet_rmse = 0.234 # eV (dummy value)
        # runtime = 125 # seconds (dummy value)
        # model_size = 123456 # parameters (dummy value)

        # outputs = {"mae": results[0]["average_mae"], "rmse":
        # results[0]["average_rmse"]}

        return {k: results[k] for k in results.keys() if k in self.objectives}


# class PseudoCrab1(PseudoCrab):
#     def __init__(self):
#         PseudoCrab.__init__(
#             self,
#             objectives=["mae", "rmse"],
#             iteration_budget=100,
#             n_float_params=5,
#             categorical_num_options=[2, 2, 3],
#             constraint_fn=sum_constraint_fn,
#         )


# what about constraint function?
default_benchmarks = dict(
    dummy=dict(
        objective_names=["mae", "rmse"],
        iteration_budget=3,
        n_float_params=5,
        n_categorical_params=2,
    ),
    minimal=dict(
        objective_names=["mae", "rmse"],
        iteration_budget=100,
        n_float_params=5,
        n_categorical_params=0,
    ),  # single-vs-multi objective?
    benchmark2=dict(
        objective_names=["mae", "rmse"],
        iteration_budget=100,
        n_float_params=5,
        n_categorical_params=2,
    ),
    # benchmark3=dict(
    #     objective_names=["mae", "rmse"],
    #     iteration_budget=100,
    #     n_float_params=5,
    #     n_categorical_params=2,
    # ),
    # benchmark4=dict(
    #     objective_names=["mae", "rmse"],
    #     iteration_budget=100,
    #     n_float_params=5,
    #     n_categorical_params=2,
    # ),
    # base alloy optimization benchmark and constraint? base alloy at least X%,
    # remainder goes to other parameters
    performance=dict(
        objective_names=["mae", "rmse", "model_size", "runtime"],
        constraint_fn=sum_constraint_fn,
        iteration_budget=100,
        n_float_params=23,
        n_categorical_params=3,
    ),
)


class PseudoCrabBasic(PseudoCrab):
    def __init__(self, **kwargs):
        PseudoCrab.__init__(
            self,
            objectives=["mae"],
            iteration_budget=100,
            n_float_params=3,
            categorical_num_options=[],
            constraint_fn=None,
            **kwargs,
        )


class PseudoCrabModerate(PseudoCrab):
    def __init__(self, **kwargs):
        PseudoCrab.__init__(
            self,
            objectives=["mae"],
            iteration_budget=100,
            n_float_params=5,
            categorical_num_options=[
                2,
            ],
            constraint_fn=sum_constraint_fn,
            **kwargs,
        )


class PseudoCrabAdvanced(PseudoCrab):
    def __init__(self, **kwargs):
        PseudoCrab.__init__(
            self,
            objectives=["mae"],
            iteration_budget=100,
            n_float_params=10,
            categorical_num_options=[2],
            constraint_fn=sum_constraint_fn,
            **kwargs,
        )


class PseudoCrabPerformance(PseudoCrab):
    def __init__(self, **kwargs):
        PseudoCrab.__init__(
            self,
            objectives=["mae", "rmse", "model_size", "runtime"],
            iteration_budget=100,
            n_float_params=20,
            categorical_num_options=[2, 2, 3],
            constraint_fn=sum_constraint_fn,
            **kwargs,
        )


# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Just a Fibonacci demonstration")
    parser.add_argument(
        "--version",
        action="version",
        version="matsci-opt-benchmarks {ver}".format(ver=__version__),
    )
    parser.add_argument(dest="n", help="n-th Fibonacci number", type=int, metavar="INT")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")
    print("The {}-th Fibonacci number is {}".format(args.n, fib(args.n)))
    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m matsci_opt_benchmarks.crabnet_hyperparameter.skeleton 42
    #
    run()

# %% Code Graveyard

# crabnet_param_names = [
#     "N",
#     "alpha",
#     "d_model",
#     "dim_feedforward",
#     "dropout",
#     "emb_scaler",
#     "epochs_step",
#     "eps",
#     "fudge",
#     "heads",
#     "k",
#     "lr",
#     "pe_resolution",
#     "ple_resolution",
#     "pos_scaler",
#     "weight_decay",
#     "batch_size",
#     "out_hidden4",
#     "betas1",
#     "betas2",
#     "bias",
#     "criterion",
#     "elem_prop",
# ]
# new_parameters = {p: parameters[p] for p in parameters if p in crabnet_param_names}
