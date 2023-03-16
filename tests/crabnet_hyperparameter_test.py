import pytest

from matsci_opt_benchmarks.crabnet_hyperparameter.core import PseudoCrab1, fib, main
from matsci_opt_benchmarks.crabnet_hyperparameter.utils.parameters import (
    matbench_metric_calculator,
    userparam_To_crabnetparam,
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


def test_userparam_To_crabnetparam():
    user_param = {"x1": 0.5, "x2": 0.2, "x3": 0.8, "x4": 0.9, "c1": 1, "c3": 1, "c2": 1}
    userparam_To_crabnetparam(user_param, seed=50)
    user_param = {"x1": 0.5, "x2": 0.2, "x3": 0.3, "c1": 0, "c2": 1}
    return userparam_To_crabnetparam(user_param, seed=50)


def test_matbench_metric_calculator():
    user_param = {"x1": 0.5, "x2": 0.2, "x3": 0.8, "x4": 0.9, "c1": 1, "c3": 1, "c2": 1}
    actual_param = userparam_To_crabnetparam(user_param, seed=50)
    matbench_metric_calculator(actual_param, dummy=True)
    user_param = {"x1": 0.5, "x2": 0.2, "x3": 0.3, "c1": 0, "c2": 1}
    actual_param = userparam_To_crabnetparam(user_param, seed=50)
    return matbench_metric_calculator(actual_param, dummy=True)


def test_PseudoCrab1():
    return PseudoCrab1()


def test_main(capsys):
    """CLI Tests"""
    # capsys is a pytest fixture that allows asserts against stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html
    main(["7"])
    captured = capsys.readouterr()
    assert "The 7-th Fibonacci number is 13" in captured.out
