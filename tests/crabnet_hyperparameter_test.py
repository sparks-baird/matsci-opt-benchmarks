import numpy as np
import pytest

from matsci_opt_benchmarks.crabnet_hyperparameter.core import (
    PseudoCrab,
    PseudoCrabMinimal,
    PseudoCrabPerformance,
    fib,
    main,
)
from matsci_opt_benchmarks.crabnet_hyperparameter.utils.parameters import (
    matbench_metric_calculator,
    userparam_to_crabnetparam,
)

__author__ = "sgbaird"
__copyright__ = "sgbaird"
__license__ = "MIT"


def test_fib():
    """API Tests"""
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)


user_param = {"x1": 0.5, "x2": 0.2, "x3": 0.8, "x4": 0.9, "c1": 1, "c3": 1, "c2": 1}
user_param2 = {"x1": 0.5, "x2": 0.2, "x3": 0.3, "c1": 0, "c2": 1}


def test_userparam_to_crabnetparam():
    userparam_to_crabnetparam(user_param, seed=50)
    userparam_to_crabnetparam(user_param2, seed=50)


def test_matbench_metric_calculator():
    actual_param = userparam_to_crabnetparam(user_param, seed=50)
    matbench_metric_calculator(actual_param, dummy=True)
    actual_param2 = userparam_to_crabnetparam(user_param2, seed=50)
    matbench_metric_calculator(actual_param2, dummy=True)


def test_PseudoCrab():
    PseudoCrab()


def test_PseudoCrabMinimal():
    pc = PseudoCrabMinimal()
    results = pc.evaluate({"x1": 0.5, "x2": 0.2, "x3": 0.3}, dummy=True)
    mae = results["mae"]
    assert 0.8 < mae < 1.0


def test_PseudoCrabPerformance():
    pc = PseudoCrabPerformance()
    # create a vector of 23 random numbers
    X = np.random.rand(20)
    # perform l1 normalization
    X = X / np.sum(X)
    parameters = {f"x{i+1}": x for i, x in enumerate(X)}
    C = {"c1": 1, "c2": 1, "c3": 2}
    parameters.update(C)
    pc.evaluate(parameters, dummy=True)


def test_main(capsys):
    """CLI Tests"""
    # capsys is a pytest fixture that allows asserts against stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html
    main(["7"])
    captured = capsys.readouterr()
    assert "The 7-th Fibonacci number is 13" in captured.out


if __name__ == "__main__":
    # test_fib()
    # test_userparam_to_crabnetparam()
    # test_matbench_metric_calculator()
    # test_PseudoCrab()
    # test_PseudoCrabMinimal()
    test_PseudoCrabPerformance()
    # test_main()
