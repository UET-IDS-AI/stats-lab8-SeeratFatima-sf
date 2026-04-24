import numpy as np

def joint_cdf_unit_square(x, y):
    """
    Return the joint CDF F_XY(x, y) for (X, Y) uniform on the unit square.
    """
    if x <= 0 or y <= 0:
        return 0.0
    elif x >= 1 and y >= 1:
        return 1.0
    elif 0 < x < 1 and y >= 1:
        return x
    elif x >= 1 and 0 < y < 1:
        return y
    else:
        return x * y


def rectangle_probability(x1, x2, y1, y2):
    """
    Compute P(x1 < X <= x2, y1 < Y <= y2)
    using the joint CDF rectangle formula:
    F(x2, y2) - F(x1, y2) - F(x2, y1) + F(x1, y1)
    """
    return (joint_cdf_unit_square(x2, y2)
            - joint_cdf_unit_square(x1, y2)
            - joint_cdf_unit_square(x2, y1)
            + joint_cdf_unit_square(x1, y1))


def marginal_fx_unit_square(x):
    """
    Return the marginal PDF f_X(x) for X when (X, Y) is uniform on the unit square.
    """
    if 0 < x < 1:
        return 1.0
    return 0.0


def marginal_fy_unit_square(y):
    """
    Return the marginal PDF f_Y(y) for Y when (X, Y) is uniform on the unit square.
    """
    if 0 < y < 1:
        return 1.0
    return 0.0

_joint_pmf = {
    (0, 0): 0.25,
    (0, 1): 0.25,
    (0, 2): 0.0,
    (1, 0): 0.0,
    (1, 1): 0.25,
    (1, 2): 0.25,
}


def joint_pmf_heads(x, y):
    """
    Return P_XY(x, y).
    """
    return _joint_pmf.get((x, y), 0.0)


def marginal_px_heads(x):
    """
    Return P_X(x) by summing the joint PMF over all y values.
    """
    return sum(joint_pmf_heads(x, y) for y in [0, 1, 2])


def marginal_py_heads(y):
    """
    Return P_Y(y) by summing the joint PMF over all x values.
    """
    return sum(joint_pmf_heads(x, y) for x in [0, 1])


def check_independence_heads():
    """
    Return True if X and Y are independent, else False.
    X and Y are independent iff P_XY(x, y) == P_X(x) * P_Y(y) for all (x, y).
    """
    for x in [0, 1]:
        for y in [0, 1, 2]:
            if not np.isclose(joint_pmf_heads(x, y),
                              marginal_px_heads(x) * marginal_py_heads(y)):
                return False
    return True
