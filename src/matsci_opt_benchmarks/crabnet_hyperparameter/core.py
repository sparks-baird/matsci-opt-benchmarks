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

from matsci_opt_benchmarks.crabnet_hyperparameter import __version__

from crabnet.crabnet_ import CrabNet
from crabnet.utils.utils import count_parameters
from matbench.bench import MatbenchBenchmark

import numpy as np
import pandas as pd
import pprint
from time import time


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


#############

def get_parameters():
    
    parameters = [ 
        {"name": 'N', "type": "range", "bounds": [1,10]},
        {"name": 'alpha', "type": "range", "bounds": [0.0,1.0]},
        {"name": 'd_model', "type": "range", "bounds": [100,1024]},
        {"name": 'dim_feedforward', "type": "range", "bounds": [1024,4096]},
        {"name": 'dropout', "type": "range", "bounds": [0.0,1.0]},
        {"name": 'emb_scaler', "type": "range", "bounds": [0.0,1.0]},
        {"name": 'eps', "type": "range", "bounds": [1e-7, 1e-4]},
        {"name": 'epochs_step', "type": "range", "bounds": [5,20]},
        {"name": 'fudge', "type": "range", "bounds": [0.0,1.0]},
        {"name": 'heads', "type": "range", "bounds": [1,10]},
        {"name": 'k', "type": "range", "bounds": [2,10]},
        {"name": 'lr', "type": "range", "bounds": [1e-4, 6e-3]},
        {"name": 'pe_resolution', "type": "range", "bounds": [2500, 10000]},
        {"name": 'ple_resolution', "type": "range", "bounds": [2500, 10000]},
        {"name": 'pos_scaler', "type": "range", "bounds": [0.0,1.0]},
        {"name": 'weight_decay', "type": "range", "bounds": [0.0,1.0]},
        {"name": 'batch_size', "type": "range", "bounds": [32,256]},
        {"name": 'out_hidden4', "type": "range", "bounds": [32,512]},
        {"name": 'betas1', "type": "range", "bounds": [0.5,0.9999]},
        {"name": 'betas2', "type": "range", "bounds": [0.5,0.9999]},
        {"name": 'bias', "type": "choice", "values": [False, True]},
        {"name": 'criterion', "type": "choice", "values": ['RobustL1', 'RobustL2']},
        {"name": 'elem_prop', "type": "choice", "values": ['mat2vec', 'magpie']}
    ]

    parameter_constraints = ["betas1 <= betas2", "emb_scaler + pos_scaler <= 1.0"]

    # return parameters, parameter_constraints
    return parameters, parameter_constraints



def evaluate(parameters, dummy:bool):

    results = matbench_metric_calculator(parameters, dummy=dummy)

    outputs = {"mae": results[0]["average_mae"], "rmse": results[0]["average_rmse"], "model_size": results[0]["model_size"], "runtime": results[0]["runtime"]}

    return outputs



def correct_parameterization(parameterization: dict, verbose=False):
    # take dictionary of tunable hyperparameters and output hyperparameter combinations compatible with CrabNet
    if verbose:
        pprint.pprint(parameterization)

    parameterization["out_hidden"] = [
        parameterization.get("out_hidden4") * 8,
        parameterization.get("out_hidden4") * 4,
        parameterization.get("out_hidden4") * 2,
        parameterization.get("out_hidden4"),
    ]
    parameterization.pop("out_hidden4")

    parameterization["betas"] = (
        parameterization.get("betas1"),
        parameterization.get("betas2"),
    )
    parameterization.pop("betas1")
    parameterization.pop("betas2")

    d_model = parameterization["d_model"]

    # make heads even (unless it's 1) (because d_model must be even)
    heads = parameterization["heads"]
    if np.mod(heads, 2) != 0:
        heads = heads + 1
    parameterization["heads"] = heads

    # NOTE: d_model must be divisible by heads
    d_model = parameterization["heads"] * round(d_model / parameterization["heads"])

    parameterization["d_model"] = d_model

    parameterization["pos_scaler_log"] = (
        1 - parameterization["emb_scaler"] - parameterization["pos_scaler"]
    )

    parameterization["epochs"] = parameterization["epochs_step"] * 4

    return parameterization


def matbench_metric_calculator(crabnet_param, dummy:bool):

    t0 = time()

    print('user parameters are :', crabnet_param)
    ## default hyperparameters 
    parameterization = {
        'N':3,'alpha':0.5,'d_model':512, 'dim_feedforward':2048, 'dropout':0.1,   
        'emb_scaler':1.0,'epochs_step':10,'eps':0.000001,'fudge':0.02,'heads':4,
        'k':6,'lr':0.001,'pe_resolution':5000,'ple_resolution':5000,
        'pos_scaler':1.0,'weight_decay':0,'batch_size':32,'out_hidden4':128,'betas1':0.9,'betas2':0.999,
        'losscurve' : False, 'learningcurve' : False, 'bias':False,'criterion':'RobustL1' , 'elem_prop':'mat2vec'
        
        }
    
    ## update the values of the selected hyperparameters 
    parameterization.update(crabnet_param)
    
    print(parameterization)
    
    cb = CrabNet(**correct_parameterization(parameterization))
   
    mb = MatbenchBenchmark(autoload=False, subset = ["matbench_expt_gap"])

    for task in mb.tasks:
        task.load()
        for fold in task.folds:

            # Inputs are either chemical compositions as strings
            # or crystal structures as pymatgen.Structure objects.
            # Outputs are either floats (regression tasks) or bools (classification tasks)
            train_inputs, train_outputs = task.get_train_and_val_data(fold)

            # prep input for CrabNet
            train_df = pd.concat((train_inputs, train_outputs), axis=1, keys=["formula", "target"])

            if dummy:
                train_df = train_df.head(10)

            # train and validate your model
            cb.fit(train_df = train_df)

            # Get testing data
            test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
            test_df = pd.concat((test_inputs, test_outputs), axis=1, keys=["formula", "target"])

            # Predict on the testing data
            # Your output should be a pandas series, numpy array, or python iterable
            # where the array elements are floats or bools
            predictions = cb.predict(test_df = test_df)

            predictions = np.nan_to_num(predictions)

            # Record your data!
            task.record(fold, predictions)
            
    model_size = count_parameters(cb.model)

    return({'average_mae':task.scores['mae']['mean'], 'average_rmse':task.scores['rmse']['mean'], "model_size": model_size, "runtime": time() - t0}, crabnet_param)

 


#############


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
