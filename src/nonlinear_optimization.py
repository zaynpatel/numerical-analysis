"""Implements algorithms to solve nonlinear optimization problems

Algorithms include:
- Newton's method for nonlinear systems
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


def newtons_method_systems(f: function, x: list, p: list, max_iter=20, tol=1e-10):
    """
    Computes a solution vector x using Newton's method for nonlinear systems

    Note: This is an iterative method. The need for iterative methods arises in
    nonlinear problems because there is no closed-form formula to find x. Instead
    we need to develop a sequence of iterates x_0, x_1, ..., x_n+1 and use a termination
    criteria to see if we have converged to a suitable (approximate) answer.

    In `root_finding.py` we saw Newton's method for a single-variable nonlinear equation.
    There, the key ideas was to linearize locally which was equivalent to a first order Taylor
    approximation in one variable. Once we found the first-order approximation we used it
    as the guess for the next iterate. The idea will be similar here, construct a first-order 
    Taylor approximation and use this to develop the next iterate.

    To do this we need to use the Taylor series for vector functions:
    Let x = (x_1, x_2, ..., x_n).T and f = (f_1, f_2, ..., f_m).T and assume that f(x) has bounded
    derivatives up to order at least two (i.e. the derivatives are contained in some interval). Then
    for a direction vector p = (p_1, p_2, ..., p_n).T the Taylor expansion for each function f_i
    in each coordinate x_j yields: f(x + p) = f(x) + J(x)p + O(||p||^2) where J(x) is the Jacobian matrix.
    The Jacobian appears in this problem because we have n functions with n unknowns and in each iteration
    we need to account for the change in each function and each variable.

    We write our system of nonlinear equations in similar form to the single, scalar nonlinear equation 
    in root finding. There, we sought a solution x s.t. f(x) = 0 and here we want the same equation but note
    that f and x are vectors. Thus we get f(x_k) + J(x_k)p_k = 0. We rearrange this equation to be:
    J(x_k)p_k = -f(x_k) and we can solve for our only unknown, p_k, using linear solve methods from
    `direct_matrix_methods.py`.

    Key idea: The convergence of Newton's method to a real solution depends on the initial guess. 
    Newton's method is good for local problems where the initial guess is close to the final solution.

    A good example of local convergence is shown in `test_nonlinear_optimization:test_newtons_method_systems_first_root` 
    and `test_nonlinear_optimization:test_newtons_method_systems_second_root`. In this example, two different starting
    points yield two different, correct roots.

    :param f: Function that evaluates points
    :type: function
    :param x: Initial vector of evaluation points
    :type: list
    :param p: Initial vector of direction information
    :type: list
    :param max_iter: Max number of iterations
    :type max_iter: int
    :param tol: Relative error tolerance
    :type tol: float
    :return: Approximated solution vector
    :rtype: list
    """
    x_k = x
    p_k = p
    for _ in range(max_iter):
        jac_at_x = jacobian(f, x_k)
        result = f(x_k)
        neg_result = [-1 * r for r in result]
        p_k = np.linalg.solve(jac_at_x.df, neg_result)
        if np.linalg.norm((x_k + p_k) - x_k) < tol * (1 + np.linalg.norm(x_k + p_k)):  # check the relative error
            return x_k + p_k
        x_k = x_k + p_k
    return x_k

def newtons_method_minimization(phi: function, x_0: list, p: list, max_iter=20, tol=1e-10):
    """
    Computes the minimum using Newton's method for unconstrained minimization

    Our goal is to minimize a function -- which we call phi -- of n variables. There is
    a special case of a minimization problem when phi is in "quadratic form." When we
    minimize this by taking the gradient [1] we get a linear system (Ax=b) and we can solve
    this using methods from `direct_matrix_methods.py`. But we often encounter cases other
    than quadratic forms and to solve these we need a Taylor series approximation for
    several variables:
    Let x = (x_1, x_2, ..., x_n).T and assume that phi(x) has bounded derivatives up to
    order at least 3. Then for a direction vector p = (p_1, p_2, ..., p_n).T, the Taylor
    expansion in each coordinate yields: phi(x + p) = phi(x) + grad(phi(x)).T@p +
    1/2*p.T@hess(phi(x))@p + O(||p||^3) where grad is the gradient and hess is the Hessian.

    We get the gradient here because we have *one function* of n-variables. This is equivalent
    to taking the Jacobian of one function. The Hessian appears because we assume bounded
    derivatives up to order at least 3.

    If we assume that x* is a minimum then the Taylor series approximation gives us
    phi(x* + p) = phi(x*) + grad(phi(x*)).T@p + 1/2*p.T@hess(phi(x*))@p + O(||p||^3) ≥ phi(x*).
    The only way that phi(x*) = phi(x*) is if grad(phi(x*)) is zero. Thus a necessary condition
    is for the gradient to be zero.

    At this point our condition for a minimum is f(x) = grad(phi(x)) = 0 (where x is a vector)
    and we want to find x. Solving for x involves solving a system of nonlinear equations
    because the gradient gives us several equations with unknown values for a single function.
    Thus, we can apply Newton's method for nonlinear systems to this problem. This yields
    grad(phi(x)) + hess(phi(x))@p = 0. We can rewrite this as hess(phi(x))@p = -grad(phi(x))
    and solve this linearly.

    Key idea: Newton's method is locally convergent which means that convergence is strongly
    associated with the initial guess. In `test_nonlinear_optimization:test_newtons_method_minimization`
    and `test_nonlinear_optimization:test_newtons_method_minimization_saddle_point`
    two different starting points lead to two different minimums -- one of these is a saddle
    point and the other is a true minimum. We will need to work with the Hessian
    to determine whether we are at a true minimum or not.

    [1]: Minimizing a convex quadratic form: 
    https://math.stackexchange.com/questions/2606391/minimization-of-a-convex-quadratic-form

    :param f: Function that evaluates points
    :type: function
    :param x_0: Initial vector of evaluation points
    :type: list
    :param p: Initial vector of direction information
    :type: list
    :param max_iter: Max number of iterations
    :type max_iter: int
    :param tol: Relative error tolerance
    :type tol: float
    :return: Approximated minimum point
    :rtype: list
    """
    x_k = x_0
    p_k = p
    for _ in range(max_iter):
        hes_at_x = hessian(phi, x_k)
        grad_at_x = jacobian(phi, x_k)  # Jacobian for a single function outputs gradient vector
        neg_result = [-1 * r for r in grad_at_x.df]
        p_k = np.linalg.solve(hes_at_x.ddf, neg_result)
        if np.linalg.norm((x_k + p_k) - x_k) < tol * (1 + np.linalg.norm(x_k + p_k)):
            return x_k + p_k
        x_k = x_k + p_k
    return x_k

def weak_line_search(phi: function, x: npt.NDArray, p: npt.NDArray, gc=1e-4):
    """
    Implements a "simple" backtracking algorithm to find an optimal parameter alpha

    :param phi: Function to evaluate points
    :type phi: function
    :param x: Vector of points to evaluate
    :type x: ndarray
    :param p: Direction vector
    :type p: ndarray
    :param gc: Guard constant (used to ensure useful descent steps)
    :type gc: float
    :return: Alpha value (step size)
    :rtype: float
    """
    alpha_k = 1  # start the alpha at some "max" value
    x_k = x
    while phi(x + (alpha_k * p)) > (phi(x_k) + gc*alpha_k * (inner(jacobian(phi, x_k).df, p))):
        alpha_k = (alpha_k) * (1/2)
    return alpha_k

def bfgs_method(phi: function, x_0: npt.NDArray, G_0: npt.NDArray, max_iter=20, grad_err_tol=1e-10):
    """
    Computes the minimum using the BFGS method

    The BFGS method is the most popular Quasi-Newton method. Quasi-Newton methods are a family of methods
    used to solve unconstrained minimization problems with an approximation of the Hessian. The Hessian
    is costly to compute and we would like to avoid computing it explicitly if we can.

    While the BFGS method is the most popular there are other variants which use similar math/algorithm
    to the method defined here *but* these methods find different ways to approximate the Hessian. The
    BFGS method approximates the *inverse* of the Hessian. It does this using the Sherman-Morrison-Woodbury
    formula to perform rank-two updates on the inverse. This avoids an explicit computation of the inverse
    which is extremely costly.

    The BFGS method is discussed symbolically in [1] but it is useful to explain some parts of it like the
    step size `p_k`. If we begin with the quadratic form m_k(p) = f_k + grad(f_k.T)p + 1/2*p.T*B_k*p then
    our goal is to find the value of p_k. We can do this by taking the gradient of m_k(p) w.r.t p.
    When we do this the first component becomes zero since there is no p, the second component becomes
    grad(f) since all the p terms are constant and become 1, and the third term is more complicated. To
    evaluate this we need the product rule since we are taking derivatives of two vectors with p terms.
    When we perform the product rule we end up with 1/2(B_k@p) + grad(f_k) = 0 and then we solve for p_k.

    Furthermore, [1] mentions that we calculate the new Hessian using a rank-two update from the information
    we find like `w_k` and `y_k`. This rank-two update allows us to update the matrix inverse using matrix-vector
    multiplications, incorporating updated information about gradients and search directions without a heavy cost.

    Separately, notice the use of a weak line search algorithm in the BFGS method. The purpose of using a line search
    algorithm in this method is to adjust magnitude of our step size at each iteration to avoid overshooting. For
    nonlinear problems especially we use adaptive line searches since we are dealing with more complicated geometry.

    Lastly, we use the magnitude of the gradient as the termination criteria because we know we are at a critical point
    (minimum in this BFGS problem) when grad(f(x)) = 0.

    [1]: BFGS method (Wikipedia): 
    https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm#Algorithm

    :param phi: Function to evaluate points
    :type phi: function
    :param x_0: Vector of points to evaluate
    :type x_0: ndarray
    :param G_0: Initial matrix used in step size calculation
    :type p_0: ndarray
    :param max_iter: Max number of iterations
    :type max_iter: int
    :param grad_err_tol: Acceptable tolerance on the gradient
    :type grad_err_tol: float
    :return: Approximated minimum
    :rtype: ndarray
    """
    confirm_column_vector(x_0)
    n, _ = G_0.shape
    G_k = G_0
    x_k = x_0
    x_k_plus_1 = x_0
    G_k_plus_1 = G_0
    for _ in range(max_iter):
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

def nonlinear_least_squares(g: function, x_0: npt.NDArray, p_0: npt.NDArray, b: npt.NDArray, max_iter=20, step_size_norm=1e-7):
    """
    Compute the solution of a nonlinear least squares problem using the Gauss-Newton method

    Our goal is to find the solution to a nonlinear least squares problem. Recall from
    `linear_least_squares:least_squares_calculation` that we can calculate this directly
    using the normal equations. There, our characterization of the problem was to minimize
    x in ||Ax - b||.

    In the literature we write g(x) = Ax and call g(x) the "model function" and it predicts
    data for each x. g depends nonlinearly on x and this nonlinearity means that we will need
    the Jacobian matrix to represent how each variable is affected by each equation. [1]

    We can rewrite our problem using this notation as ||g(x) - b||. Further, 
    it is recommended to rewrite ||g(x) - b|| as min of 1/2||g(x) - b||^2. We call this phi(x). 
    For clarity, the equation we work with is phi(x) = 1/2||g(x) - b||^2. We work with this equation 
    because taking derivatives to find the minimum is easier.

    When we compute the minimum [2] we get A(x*).T@(g(x*) - b) = 0 where A(x*) is the Jacobian
    of x*. This generalizes the normal equations to the case of nonlinear equations. But, this
    equation we obtain requires us to solve a nonlinear system of n equations since we do not
    know what values of x* make this zero. We are back to needing Newton's method for systems.

    We return to our original characterization of the problem ||g(x) - b|| and our goal is to
    approximate g(x_k+1). We can do this using Newton's method: g(x_k+1) = g(x_k) + A(x_k)p_k.
    This is very similar to the equation from Newton's method for systems. Plugging this into
    the original characterization we get ||A(x_k)p - (b - g(x_k))|| where (b - g(x_k)) is the
    residual.

    The normal equations for the norm above are A(x_k).T@(A(x_k))p_k = A(x_k).T(b-g(x_k)). We
    turn this into a linear solve in the algorithm below.

    We can check that our solution is correct by using the residual.

    [1]: Recall that we are in a least squares problem where m ≥ n. In other words the number of equations is
    greater than the number of unknowns so we need the Jacobian to represent changes of each equation w.r.t
    each unknown.
    [2]: Reference on gradient of the min phi(x):
    https://math.stackexchange.com/questions/3508373/taking-the-gradient-of-f-mathbfx-frac12-mathbfa-mathbfx-ma

    :param g: Function to evaluate points
    :type g: function
    :param x_0: Vector of points to evaluate
    :type x_0: ndarray
    :param p_0: Initial direction vector
    :type p_0: ndarray
    :param b: Column vector
    :type b: ndarray
    :param max_iter: Max number of iterations
    :type max_iter: int
    :param tol: Relative error tolerance
    :type tol: int
    :return: Approximated solution
    :rtype: ndarray
    """
    x_k = x_0
    p_k = p_0
    for _ in range(max_iter):
        jac_at_x = jacobian(g, x_k)
        jacobian_T = jac_at_x.df.T
        J = jacobian_T @ jac_at_x.df
        residual = b - g(x_k)
        j = jacobian_T @ residual
        p_k = np.linalg.solve(J, j)
        norm = np.linalg.norm(p_k)
        if norm < step_size_norm:
            return x_k
        x_k = x_k + p_k
    return x_k
