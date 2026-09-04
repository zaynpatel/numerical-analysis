import numpy as np
import pytest

from scipy.optimize import bisect, fixed_point, newton

from src import root_finding as rf


def test_bisesct():
    def f(x):
        return x**3 - 30*x**2 + 2552
    a = 0
    b = 20
    atol = 1e-8
    r = rf.bisect(f, a, b, atol)
    sp = bisect(f, a, b)
    assert np.allclose(r, sp)


def test_bisect_sin():
    # Test that the algorithm works against the "malicioius" case of sin(x)
    def f(x):
        return np.sin(x)
    a = -1
    b = 1
    atol = 1e-8
    r = rf.bisect(f, a, b, atol)
    sp = bisect(f, a, b)
    assert np.allclose(r, sp)


def test_fixed_point():
    def g(x):
        return 1 + (1/x)
    x_0 = 2
    r = rf.fixed_point(g, x_0)
    sp = fixed_point(g, x_0)
    assert np.allclose(r, sp)


def test_fixed_point_fail():
    def g(x):
        return 1 / (x - 1)
    x_0 = 1.6
    with pytest.raises(RuntimeError) as exc:
        rf.fixed_point(g, x_0)
    assert "Failed to converge" in str(exc.value)


def test_newtons_method():
    def f(x):
        return 2*np.cosh(x/4) - x

    def f_prime(x):
        return 0.5*np.sinh(x/4) - 1
    x_0 = 2
    r = rf.newtons_method(f, f_prime, x_0)
    sp = newton(func=f, fprime=f_prime, x0=x_0)
    assert np.allclose(r, sp)


def test_secant_method():
    def f(x):
        return 2*np.cosh(x/4) - x
    x_0 = 2
    x_1 = 4
    r = rf.secant_method(f, x_0, x_1, 20)
    sp = newton(f, x0=x_0)
    assert np.allclose(r, sp)
