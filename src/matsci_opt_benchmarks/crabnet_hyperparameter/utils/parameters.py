import pprint
from copy import copy
from time import time

import numpy as np
import pandas as pd
from crabnet.crabnet_ import CrabNet
from crabnet.utils.utils import count_parameters
from matbench.bench import MatbenchBenchmark
from numpy.random import default_rng
from xtal2png.utils.data import element_wise_scaler

from matsci_opt_benchmarks.crabnet_hyperparameter.utils.validation import (
    sum_constraint_fn,
)


def get_parameters():
    """Get parameter set and parameter constraints for CrabNet.

    Returns:
        (list(dict), list): CrabNet parameters, CrabNet parameter contraints for Ax
    """
    parameters = [
        {"name": "N", "type": "range", "bounds": [1, 10]},
        {"name": "alpha", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "d_model", "type": "range", "bounds": [100, 1024]},
        {"name": "dim_feedforward", "type": "range", "bounds": [1024, 4096]},
        {"name": "dropout", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "emb_scaler", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "eps", "type": "range", "bounds": [1e-7, 1e-4]},
        {"name": "epochs_step", "type": "range", "bounds": [5, 20]},
        {"name": "fudge", "type": "range", "bounds": [0.0, 0.1]},
        {"name": "heads", "type": "range", "bounds": [1, 10]},
        {"name": "k", "type": "range", "bounds": [2, 10]},
        {"name": "lr", "type": "range", "bounds": [1e-4, 6e-3]},
        {"name": "pe_resolution", "type": "range", "bounds": [2500, 10000]},
        {"name": "ple_resolution", "type": "range", "bounds": [2500, 10000]},
        {"name": "pos_scaler", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "weight_decay", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "batch_size", "type": "range", "bounds": [32, 256]},
        {"name": "out_hidden4", "type": "range", "bounds": [32, 512]},
        {"name": "betas1", "type": "range", "bounds": [0.5, 0.9999]},
        {"name": "betas2", "type": "range", "bounds": [0.5, 0.9999]},
        {"name": "bias", "type": "choice", "values": [False, True]},
        {"name": "criterion", "type": "choice", "values": ["RobustL1", "RobustL2"]},
        {
            "name": "elem_prop",
            "type": "choice",
            "values": ["mat2vec", "magpie", "onehot"],
        },
        {"name": "train_frac", "type": "range", "bounds": [0.01, 1.0]},
    ]

    parameter_constraints = ["betas1 <= betas2", "emb_scaler + pos_scaler <= 1.0"]

    return parameters, parameter_constraints


def correct_parameterization(parameters: dict, verbose=False):
    """Modify tunable hyperparameters for combatibility with CrabNet.

    Args:
        parameters (dict): Hyperparameter set used by Ax in optimization.
        verbose (bool, optional): Print function progress. Defaults to False.

    Returns:
        dict: Modified dictionary with the correct parameters for CrabNet
        compatibility.
    """
    # take dictionary of tunable hyperparameters and output hyperparameter
    # combinations compatible with CrabNet

    if verbose:
        pprint.pprint(parameters)

    parameters["out_hidden"] = [
        parameters.get("out_hidden4") * 8,
        parameters.get("out_hidden4") * 4,
        parameters.get("out_hidden4") * 2,
        parameters.get("out_hidden4"),
    ]
    parameters.pop("out_hidden4")

    parameters["betas"] = (
        parameters.get("betas1"),
        parameters.get("betas2"),
    )
    parameters.pop("betas1")
    parameters.pop("betas2")

    d_model = parameters["d_model"]

    # make heads even (unless it's 1) (because d_model must be even)
    heads = parameters["heads"]
    if np.mod(heads, 2) != 0:
        heads = heads + 1
    parameters["heads"] = heads

    # NOTE: d_model must be divisible by heads
    d_model = parameters["heads"] * round(d_model / parameters["heads"])

    parameters["d_model"] = d_model

    parameters["pos_scaler_log"] = (
        1 - parameters["emb_scaler"] - parameters["pos_scaler"]
    )

    parameters["epochs"] = parameters["epochs_step"] * 4

    return parameters


# Helper fuction to slice the dictionary based on starting with string s
def slicedict(d, s):
    return {k: v for k, v in d.items() if k.startswith(s)}


# Helper fuction to validate the number of categorical variables and their values
def validate_categorical_inputs(c_dict):
    assert (
        len(c_dict) <= 3
    ), "Number of categorical variable should be less than or equal to 3"

    for k, v in c_dict.items():
        assert v == 0 or v == 1 or v == 2, "{} should be either 0, 1, or 2".format(k)


# Helper Function to convert input parameters to CrabNet hyperparameters
def userparam_To_crabnetparam(user_param, seed=50):
    # set the seed
    np.random.seed(seed)

    # separating the integer/float from categorical variable
    x_dict = slicedict(user_param, "x")
    c_dict = slicedict(user_param, "c")

    # validate the number of categorical variables and their values
    validate_categorical_inputs(c_dict)

    # Defining Crabnet 3 categorical hyperparameters and their values
    crabnet_categorical = {
        "bias": ["False", "True"],
        "criterion": ["RobustL1", "RobustL2"],
        "elem_prop": ["mat2vec", "magpie"],
    }

    # Defining Crabnet all 20 hyperparameters (integer/float) and their ranges
    crabnet_hyperparam = {
        "N": [1, 10],
        "alpha": [0, 1],
        "d_model": [100, 1024],
        "dim_feedforward": [1024, 4096],
        "dropout": [0, 1],
        "emb_scaler": [0, 1.0],
        "epochs_step": [5, 20],
        "eps": [1e-7, 1e-4],
        "fudge": [0, 1.0],
        "heads": [1, 10],
        "k": [2, 10],
        "lr": [1e-4, 6e-3],
        "pe_resolution": [2500, 10000],
        "ple_resolution": [2500, 10000],
        "pos_scaler": [0, 1.0],
        "weight_decay": [0, 1.0],
        "batch_size": [32, 256],
        "out_hidden4": [32, 512],
        "betas1": [0.5, 0.9999],
        "betas2": [0.5, 0.9999],
    }

    # List of parameters having floating point values (total 10 nos)
    float_hyperparam = [
        "alpha",
        "dropout",
        "emb_scalar",
        "eps",
        "fudge",
        "lr",
        "pos_scaler",
        "weight_decay",
        "betas1",
        "betas2",
    ]

    # randomly selecting the parameters depending on x_dict size
    selected_param = np.random.choice(
        list(crabnet_hyperparam.keys()), size=len(x_dict), replace=False
    )

    # randomly selecting the categorical parameters depending on c_dict size
    selected_catogorical = np.random.choice(
        list(crabnet_categorical.keys()), size=len(c_dict), replace=False
    )

    # Intializing actual param dict
    actual_param = {}

    # converting user param to Crabnet hyperparam (int/float)
    for p, k in zip(selected_param, x_dict.keys()):
        # for handling hyperparameters having float values
        if p in float_hyperparam:
            actual_param[p] = np.float64(
                np.around(
                    element_wise_scaler(
                        x_dict[k],
                        feature_range=crabnet_hyperparam[p],
                        data_range=[0, 1],
                    ),
                    decimals=4,
                    out=None,
                )
            )

        # for hyperparameters having integer values
        else:
            actual_param[p] = np.int(
                np.round(
                    element_wise_scaler(
                        x_dict[k],
                        feature_range=crabnet_hyperparam[p],
                        data_range=[0, 1],
                    )
                )
            )

    # converting categorical user param to Crabnet categorical hyperparam
    for s, c in zip(selected_catogorical, c_dict.values()):
        actual_param[s] = crabnet_categorical[s][c]

    return actual_param


def matbench_metric_calculator(crabnet_param, dummy=False):
    print("user parameters are :", crabnet_param)
    # default hyperparameters
    parameterization = {
        "N": 3,
        "alpha": 0.5,
        "d_model": 512,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "emb_scaler": 1.0,
        "epochs_step": 10,
        "eps": 0.000001,
        "fudge": 0.02,
        "heads": 4,
        "k": 6,
        "lr": 0.001,
        "pe_resolution": 5000,
        "ple_resolution": 5000,
        "pos_scaler": 1.0,
        "weight_decay": 0,
        "batch_size": 32,
        "out_hidden4": 128,
        "betas1": 0.9,
        "betas2": 0.999,
        "losscurve": False,
        "learningcurve": False,
        "bias": False,
        "criterion": "RobustL1",
        "elem_prop": "mat2vec",
    }

    # *** replace rest of function with the evaluate_surrogate(params) function

    # update the values of the selected hyperparameters
    parameterization.update(crabnet_param)

    print(parameterization)

    cb = CrabNet(**correct_parameterization(parameterization))

    mb = MatbenchBenchmark(autoload=False, subset=["matbench_expt_gap"])

    for task in mb.tasks:
        task.load()
        for fold in task.folds:
            # Inputs are either chemical compositions as strings
            # or crystal structures as pymatgen.Structure objects.
            # Outputs are either floats (regression tasks) or bools (classification
            # tasks)
            train_inputs, train_outputs = task.get_train_and_val_data(fold)

            # prep input for CrabNet
            train_df = pd.concat(
                (train_inputs, train_outputs), axis=1, keys=["formula", "target"]
            )

            if dummy:
                train_df = train_df.head(10)

            # train and validate your model
            cb.fit(train_df=train_df)

            # Get testing data
            test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
            test_df = pd.concat(
                (test_inputs, test_outputs), axis=1, keys=["formula", "target"]
            )

            # Predict on the testing data
            # Your output should be a pandas series, numpy array, or python iterable
            # where the array elements are floats or bools
            predictions = cb.predict(test_df=test_df)

            predictions = np.nan_to_num(predictions)

            # Record your data!
            task.record(fold, predictions)

    model_size = count_parameters(cb.model)
    return (
        {
            "average_mae": task.scores["mae"]["mean"],
            "average_rmse": task.scores["rmse"]["mean"],
            "model_size": model_size,
        },
        crabnet_param,
    )
    # return({'average_mae':task.scores['mae']['mean'],
    # 'average_rmse':task.scores['rmse']['mean']}, crabnet_param)


def single_max_contraint_fn(parameters):
    """Return True if one of the parameters (x1, x2 ... xn) is >= 0.9, and x1 + x2 .. xn
    <= 1.0, else False."""

    xn_vals = np.array(
        [parameters.get(key) for key in parameters.keys() if key.startswith("x")]
    )

    if max(xn_vals) >= 0.9 and sum_constraint_fn(parameters):
        return True
    return False


def submitit_evaluate(parameters):
    """Trains CrabNet using the inputted parameter set and records the results.

    Args:
        parameters (list(dict)): Hyperparameter set for CrabNet.

    Returns:
        dict: Results after CrabNet training. MAE, RMSE, Model Size, Runtime. If
        there is an error, dict contains error at dict["error"]
    """
    t0 = time()

    print("user parameters are:", parameters)

    parameters = copy(parameters)
    train_frac = parameters.pop("train_frac")
    seed = parameters.pop("sample_seed")
    if "hardware" in parameters:
        parameters.pop("hardware")
    rng = default_rng(seed)

    # default hyperparameters
    parameterization = {
        "N": 3,
        "alpha": 0.5,
        "d_model": 512,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "emb_scaler": 1.0,
        "epochs_step": 10,
        "eps": 0.000001,
        "fudge": 0.02,
        "heads": 4,
        "k": 6,
        "lr": 0.001,
        "pe_resolution": 5000,
        "ple_resolution": 5000,
        "pos_scaler": 1.0,
        "weight_decay": 0,
        "batch_size": 32,
        "out_hidden4": 128,
        "betas1": 0.9,
        "betas2": 0.999,
        "losscurve": False,
        "learningcurve": False,
        "bias": False,
        "criterion": "RobustL1",
        "elem_prop": "mat2vec",
    }

    # update the values of the selected hyperparameters
    parameterization.update(parameters)

    print(parameterization)

    cb = CrabNet(**correct_parameterization(parameterization))

    mb = MatbenchBenchmark(autoload=False, subset=["matbench_expt_gap"])

    # TODO: try-except with NaN output if failure

    try:
        for task in mb.tasks:
            task.load()
            for fold in task.folds:
                # Inputs are either chemical compositions as strings or crystal
                # structures as pymatgen.Structure objects. Outputs are either
                # floats (regression tasks) or bools (classification tasks)
                train_inputs, train_outputs = task.get_train_and_val_data(fold)

                # prep input for CrabNet
                train_df = pd.concat(
                    (train_inputs, train_outputs), axis=1, keys=["formula", "target"]
                )

                train_df = train_df.sample(frac=train_frac, random_state=rng)

                # train and validate your model
                cb.fit(train_df=train_df)

                # Get testing data
                test_inputs, test_outputs = task.get_test_data(
                    fold, include_target=True
                )
                test_df = pd.concat(
                    (test_inputs, test_outputs), axis=1, keys=["formula", "target"]
                )

                # Predict on the testing data
                # Your output should be a pandas series, numpy array, or python iterable
                # where the array elements are floats or bools
                predictions = cb.predict(test_df=test_df)

                predictions = np.nan_to_num(predictions)

                # Record your data!
                task.record(fold, predictions)
            scores = task.scores
            # `fit` needs to be called prior to `count_parameters`
            # all 5 models should be same size, but we take the last for simplicity
            model_size = count_parameters(cb.model)

        # REVIEW: if using multiple tasks, return multiple `scores` dicts

        return {"scores": scores, "model_size": model_size, "runtime": time() - t0}
    except Exception as e:
        return {"error": str(e), "runtime": time() - t0}
