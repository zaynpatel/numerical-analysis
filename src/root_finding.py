"""Implements algorithms to solve scalar, nonlinear equations

Algorithms include:
- Bisection method
- Fixed point iteration
- Newton's method
- Secant method
"""
import numpy as np
import numpy.typing as npt


def bisect(func: function, a: float, b: float, atol: float) -> float:
    """
    Computes a root in an interval using the bisection method

    The key idea of the bisection method is that we have some interval
    [a, b] where the value of f changes sign (f(a) * f(p) < 0). When
    this happens we can invoke the Intermediate Value Theorem which says
    that if f(a) * f(p) < 0 in some interval [a, b] then there exists
    a point c where f(c) = 0. This point c is the root/solution of the
    function and that is what we want to find.

    Once the bisection method has an interval [a, b] which meets the
    requirements of the IVT then we can split the interval, at its midpoint,
    and we can apply IVT to the subintervals. If we call the midpoint p then
    we can compute f(a) * f(p) and check if this is less than zero. If it is
    then by the IVT we have a point c s.t. f(c) = 0 in that interval. If not,
    it has to be in the other interval. We continue this for n iterations.

    The consistent halving of the interval gives us a special property of the
    bisection method: we can compute the required number of iterations for
    convergence. The formula for this is in the algorithm:
    np.ceil(np.log2(b - a) - np.log2(2 * atol)).

    Note: If you want to use the bisect method to compute multiple roots for a
    given function then you need to pass in separate valid intervals.

    :param func: Function to find the root(s) of
    :type func: function
    :param a: Left side of the interval
    :type a: float
    :param b: Right side of the interval
    :type b: float
    :param atol: Absolute error tolerance between two guesses
    :type atol: float
    :return: Approximation of a root in an interval [a, b]
    :rtype: float
    """
    if a >= b or (func(a) * func(b) >= 0):
        raise Exception("Either the endpoints are not valid or the function does not meet the requirements.")

    n = np.ceil(np.log2(b - a) - np.log2(2 * atol))  # Compute the number of iterations needed, using log rules
    for _ in range(int(n)):
        p = (a + b) / 2  # compute the midpoint
        if func(a) * func(p) < 0:  # apply IVT
            b = p
        elif func(a) * func(p) > 0:
            a = p
        elif func(a) * func(p) == 0:
            return p
    return p

def fixed_point(g: function, x_0: float, max_iter=20, tol=1e-8):
    """
    Compute a root for a scalar continuous function in one variable using
    the fixed point method

    The key idea of the fixed point iteration method is to rewrite our problem
    f(x) = 0 as x = g(x) and iterate over the function g. In the formulation
    x = g(x), x is called a "fixed point" because the transformation g on x returns
    x itself, not some y which has been transformed by g. This algorithm specifically
    selects a function g(x) s.t. f(x) = 0 iff g(x) = x.

    To understand why this works we can suppose that we are given an interval [a, b]
    where a < b so that g(a) > a and g(b) < b. The case of g(a) = a or g(b) = b means
    that we have a fixed point. Define a continuous function phi(x) = g(x) - x and we
    can plug in a to find that phi(a) = g(a) - a => phi(a) > 0 (because of our inequalities
    above) and phi(b) = g(b) - b => phi(b) < 0. Thus we have two function values of opposite
    signs on the interval [a, b]. By Bolzano's Theorem this means that we have at least
    one root in the interval phi(x*) = 0 => g(x*) = x* => f(x*) = 0. So our function has a root.

    There is more theory to explain when this method converges and for which functions g. It is
    dependent on the derivative g being less than or equal to 1. More information about convergence
    is provided in [1].

    Note: If you want to use the fixed point method to compute multiple roots for a
    given function then you need to pass in separate valid initial guesses.

    [1]: Fixed point iteration convergence: https://math.stackexchange.com/a/3806248/1657049

    :param g: Function to find the fixed point of
    :type g: function
    :param x_0: Initial guess for x
    :type x_0: float
    :param max_iter: Max number of iterations
    :type max_iter: int
    :param tol: Error tolerance between successive guesses
    :type tol: float
    """
    x_k = x_0
    for _ in range(max_iter):
        x_k_minus_1 = x_k
        x_k = g(x_k)
        if np.linalg.norm(x_k - x_k_minus_1) < tol * (1 + np.linalg.norm(x_k)):
            return x_k
    raise RuntimeError(f"Failed to converge with initial fixed point: {x_0}")  #TODO: except the error and try Newton's method for g and re-run

def newtons_method(f: function, f_prime: function, x_0: float, max_iter=20, tol=1e-8):
    """
    Compute a root for a scalar differentiable function using Newton's method

    We derive Newton's method by making a first-order Taylor series approximation
    of the function f(x). This gives us:
    f(x) = f(x_k) + f'(x_k)(x* - x_k) + f''(a(x)(x - x_k)^2) / 2.
    We neglect the second order term when defining the next iterate. We can solve
    for x* = x_k - (f(x_k) / f'(x_k)) and this formula defines Newton's method.

    Note that this formula requires that the first and second derivatives exist
    and are continuous. This formula also assumes that f_prime can be found easily
    and used in evaluation. Finding f_prime can sometimes be a tricky task so we
    will still apply Newton's method but using a finite difference formula to find
    the derivative instead.

    Note: If you want to use Newton's method to compute multiple roots for a
    given function then you need to pass in separate valid initial guesses.

    :param f: Function to find the solution of
    :type f: function
    :param f_prime: Derivative of f
    :type f_prime: function
    :param x_0: Initial guess for x
    :type x_0: float
    :param max_iter: Max number of iterations
    :type max_iter: int
    :param tol: Error tolerance between successive guesses
    :type tol: float
    """
    x_k_plus_1 = None
    x_k = x_0
    for _ in range(max_iter):
        x_k_plus_1 = x_k - (f(x_k) / f_prime(x_k))
        if np.linalg.norm(x_k_plus_1 - x_k) < tol * (1 + np.linalg.norm(x_k_plus_1)):  # compute relative error
            return x_k_plus_1
        x_k = x_k_plus_1
    raise RuntimeError(f"Failed to converge to a soluton with initial guess: {x_0} after {max_iter} iterations")

def secant_method(f: function, x_0: float, x_1: float, max_iter=20, tol=1e-8):
    """
    Compute a root for a scalar differentiable function using the secant method

    The secant method is a variant of Newton's method that uses a finite difference
    to compute the derivative when it is too complicated to solve for. Finite
    difference approximations [1] use the definition of the derivative but with a
    small finite difference instead of a limit. In the secant method this difference 
    is the value of f(x_k) - f(x_k-1).

    We calculate the derivative using methods like [1] and plug that into the Newton's
    method formula mentioned in `root_finding:newtons_method` to solve for the root.

    Note: If you want to use the secant method to compute multiple roots for a
    given function then you need to pass in separate valid initial guesses for x_0 and
    x_1.

    [1]: Finite difference approximations: 
    https://en.wikipedia.org/wiki/Finite_difference
    
    :param f: Function to find the solution of
    :type f: function
    :param x_0: Initial guess for x
    :type x_0: float
    :param x_1: Second initial guess for x
    :type x_1: float
    :param max_iter: Max number of iterations
    :type max_iter: int
    :param tol: Error tolerance between successive guesses
    :type tol: float
    """
    x_k_plus_1 = None
    x_k = x_1
    x_k_minus_1 = x_0
    for _ in range(max_iter):
        x_k_plus_1 = x_k - ((f(x_k)*(x_k - x_k_minus_1)) / (f(x_k) - f(x_k_minus_1)))
        if np.linalg.norm(x_k_plus_1 - x_k) < tol * (1 + np.linalg.norm(x_k_plus_1)):
            return x_k_plus_1
        x_k = x_k_plus_1
    raise RuntimeError(f"Failed to converge to a soluton with initial guesses: {x_0, x_1} after {max_iter} iterations")
