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
    Computes the root in an interval using the bisection method

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
    Compute a fixed point for a scalar continuous function in one variable

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
