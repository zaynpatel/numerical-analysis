import numpy as np
import pytest

from scipy.optimize import bisect, fixed_point

from src import root_finding as rf

def test_bisesct():
    def f(x):
        return x**3 - 30*x**2 + 2552
    a = 0
    b = 20
    atol=1e-8
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
