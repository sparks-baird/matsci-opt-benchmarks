import numpy as np


def check_float_ranges(parameters, tol=1e-6):
    xn_vals = np.array(
        [[key, parameters.get(key)] for key in parameters.keys() if key.startswith("x")]
    )
    # num_vals = xn_vals.shape[0]

    # checking float range (if val > 0 and val < 1 + 1e-6)   // ask if it should be val
    # < -1.e6 instead
    passed = True

    assert_message = ""

    for key_val in xn_vals:
        key = key_val[0]
        val = key_val[1]

        if float(val) < 0 or float(val) > (1 + 1e-6):
            passed = False

            if len(assert_message) == 0:
                assert_message += f"{key}={val}"
            else:
                assert_message += f", {key}={val}"

    assert passed, (
        "Parameters out of range. Failed due to the following parameter values: "
        + assert_message
    )


def sum_constraint_fn(parameters, tol=1e-6):
    """Return True if x1 + x2 + ... + xn <= 1.0, else False."""

    xn_vals = np.array(
        [parameters.get(key) for key in parameters.keys() if key.startswith("x")]
    )

    print(xn_vals.sum())

    if xn_vals.sum() - tol <= 1.0:
        return True
    return False
