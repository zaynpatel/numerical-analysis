"""Implements algorithms to solve nonlinear optimization problems

Algorithms include:
- Newton's method for systems
- Newton's method for unconstrained minimization
- Weak line search using a simple backtracking algorithm
- BFGS method
- Gauss-Newton method for nonlinear least-squares
"""
import numpy as np
import numpy.typing as npt

from scipy.differentiate import jacobian, hessian

from src.direct_matrix_methods import confirm_column_vector
from src.iterative_matrix_methods import inner


def newtons_method_systems(f: function, x: list, p: list):
    """
    ...

    :param f: Function that evaluates points
    :type: function
    :param eval_points: Initial vector of evaluation points
    :type: list
    :param p: Initial vector of direction information
    :type: list
    :return: Approximated solution vector
    :rtype: list
    """
    x_k = x
    p_k = p
    for _ in range(50):
        jac_at_x = jacobian(f, x_k)
        result = f(x_k)
        neg_result = [-1 * r for r in result]
        p_k = np.linalg.solve(jac_at_x.df, neg_result)
        x_k = x_k + p_k
    return x_k

def newtons_method_minimization(phi: function, x_0: list, p: list):
    """
    -> You ought to mention that sometimes you will converge to a point but it won't be
    the minimization point. Instead it might be a saddle point.

    -> You can also mention a relative error termination criteria instead of needing to
    use the for loop #TODO: I could implement the termination criteria without needing a
    while loop condition, that might make the code cleaner too since an if check?
    """
    x_k = x_0
    p_k = p
    for _ in range(50):
        hes_at_x = hessian(phi, x_k)
        grad_at_x = jacobian(phi, x_k)  # Jacobian for a single function outputs gradient vector
        neg_result = [-1 * r for r in grad_at_x.df]
        p_k = np.linalg.solve(hes_at_x.ddf, neg_result)
        x_k = x_k + p_k
    return x_k

def weak_line_search(phi: function, x: npt.NDArray, p: npt.NDArray, cp=1e-4):
    """
    Implements a "simple" backtracking algorithm to find an optimal parameter alpha

    -> Explain more about where this comes from, weak strategy is to take decreasing powers of j
    -> Also, why is this the while condition?
    """
    alpha_k = 1  # start the alpha at some "max" value
    x_k = x
    while phi(x + (alpha_k * p)) > (phi(x_k) + cp*alpha_k * (inner(jacobian(phi, x_k).df, p))):
        alpha_k = (alpha_k) * (1/2)
    return alpha_k

def bfgs_method(phi: function, x_0: npt.NDArray, G_0: npt.NDArray, iter_max=20, grad_err_tol=1e-10):
    """
    - Begin with the importance of Quasi-Newton methods for speed and the ability to get to a minimization result
    but without needing explicit computation of the Hessian matrix (as the Newton methods need)

    - Mention that we update the inverse as opposed to computing a B_k and this is computationally simpler
    here because we use the Sherman-Morrison-Woodbury formula (this was in the SDSU ppt) which is a rank-two update

    - Explain the derivation and how we get terms like `y_k` and `p_k` *and* that the BFGS method makes the rank-two
    update using information it found from previous gradients and x values (this is the "incorporation of previous data")

    - Explain that we need a line search method to adjust the direction size we are taking towards the minimum
    This helps us not overshoot and also adjust the scalar here at each instance. You can also mention why the
    step size changes at each iteration (we are getting closer to min and overshooting again?)

    - Explain that for SPD B_k we guarantee descent direction because of the inverse energy property

    - Briefly discuss the termination criteria
    """
    confirm_column_vector(x_0)
    n, _ = G_0.shape
    G_k = G_0
    x_k = x_0
    x_k_plus_1 = x_0
    G_k_plus_1 = G_0
    for _ in range(iter_max):
        grad_at_x = jacobian(phi, x_k)
        if np.linalg.norm(grad_at_x.df) < grad_err_tol:
            return x_k_plus_1
        p_k = np.dot(-G_k, grad_at_x.df).reshape(-1, 1)
        line_search_alpha = weak_line_search(phi, x_k, p_k)
        x_k_plus_1 = x_k + line_search_alpha*p_k
        w_k = line_search_alpha*p_k
        y_k = (jacobian(phi, x_k_plus_1).df - grad_at_x.df).reshape(-1, 1)
        first_rank_one_update = np.identity(n) - (np.outer(w_k, y_k.T) / inner(y_k, w_k))
        second_rank_one_update = (np.identity(n) - (np.outer(y_k, w_k.T) / inner(y_k, w_k)))
        addition_at_end = (np.outer(w_k, w_k.T) / inner(y_k, w_k))
        G_k_plus_1 = (first_rank_one_update @ G_k @ second_rank_one_update) + addition_at_end
        x_k = x_k_plus_1
        G_k = G_k_plus_1
    return x_k_plus_1

def nonlinear_least_squares(phi: function, x_0: npt.NDArray, p_0: npt.NDArray, b: npt.NDArray, iter_max=20, step_size_norm=1e-7):
    """
    Docstring for nonlinear_least_squares

    -> Explain why we need to recalculate the Jacobian each time

    [1]: Reference on gradient of the min phi(x):
    https://math.stackexchange.com/questions/3508373/taking-the-gradient-of-f-mathbfx-frac12-mathbfa-mathbfx-ma
    """
    x_k = x_0
    p_k = p_0
    for _ in range(iter_max):
        jac_at_x = jacobian(phi, x_k)
        jacobian_T = jac_at_x.df.T
        J = jacobian_T @ jac_at_x.df
        residual = b - phi(x_k)
        j = jacobian_T @ residual
        p_k = np.linalg.solve(J, j)
        norm = np.linalg.norm(p_k)
        if norm < step_size_norm:
            return x_k
        x_k = x_k + p_k
    return x_k
