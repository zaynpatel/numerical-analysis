"""Implements algorithms to solve nonlinear optimization problems

Algorithms include:
- Newton's method for systems and unconstrained optimization
- Gradient descent with line search
- BFGS
- Gauss-Newton method for nonlinear least-squares
"""
import numpy as np
import numpy.typing as npt

from scipy.differentiate import jacobian


def newtons_method_systems(x: list, p: list, tol=1e-10) -> npt.NDArray:
    """
    ...    

    :param f: Initial vector of functions
    :type: list
    :param eval_points: Initial vector of evaluation points
    :type: list
    :param p: Initial vector of direction information
    :type: list
    :param tol: Accepted error tolerance
    :type tol: int
    :return: Approximated solution vector
    :type: ndarray
    """
    def f(x):
        x1, x2 = x
        return [x1**2 - 2*x1 - x2 + 1, x1**2 + x2**2 - 1]
    x_k = x
    p_k = p
    x_k_plus_1 = None
    for _ in range(50):
        jac_at_x = jacobian(f, x_k)
        result = f(x_k)
        neg_result = [-1 * r for r in result]
        p_k = np.linalg.solve(jac_at_x.df, neg_result)
        x_k_plus_1 = x_k + p_k
    return x_k_plus_1




x = [2, 52000, 52000, 520000]
p = [0, 1, 0, 1]
print(newtons_method_systems(x, p))
