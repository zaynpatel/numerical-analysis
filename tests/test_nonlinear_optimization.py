import numpy as np
from scipy.optimize import minimize

from src import nonlinear_optimization as nlo


def test_newtons_method_systems_first_root():
    # Benchmark testing with a known example from the textbook (Example 9.2)
    def f(x):
        x1, x2 = x
        return [x1**2 - 2*x1 - x2 + 1, x1**2 + x2**2 - 1]
    x = [1, 1]  # The initial guess matters more than p
    p = [-2, 1]
    r = nlo.newtons_method_systems(f, x, p)
    assert np.allclose(r, [1, 0])


def test_newtons_method_systems_second_root():
    # Benchmark testing with a known example from the textbook (Example 9.2)
    def f(x):
        x1, x2 = x
        return [x1**2 - 2*x1 - x2 + 1, x1**2 + x2**2 - 1]
    x = [-1, 1]
    p = [1, 1]
    r = nlo.newtons_method_systems(f, x, p)
    assert np.allclose(r, [0, 1])


def test_newtons_method_minimization():
    # Benchmark testing with known function from textbook (Example 9.5)
    def phi(x):
        x1, x2 = x
        return 1/2*((1.5 - x1*(1-x2))**2 + (2.25 - x1*(1-x2**2))**2 + (2.625 - x1*(1-x2**3))**2)
    x_0 = [8, 0.2]
    p = [1, 1]
    r = nlo.newtons_method_minimization(phi, x_0, p)
    assert np.allclose(r, [3, .5], rtol=1e-3)


def test_newtons_method_minimization_saddle_point():
    # Benchmark testing with known function from textbook (Example 9.5)
    def phi(x):
        x1, x2 = x
        return 1/2*((1.5 - x1*(1-x2))**2 + (2.25 - x1*(1-x2**2))**2 + (2.625 - x1*(1-x2**3))**2)
    x_0 = [8, 0.8]
    p = [1, 1]
    r = nlo.newtons_method_minimization(phi, x_0, p)
    assert np.allclose(r, [0, 1], rtol=1e-3)


def test_bfgs_method():
    def phi(x):
        x1, x2 = x
        return 1/2*((1.5 - x1*(1-x2))**2 + (2.25 - x1*(1-x2**2))**2 + (2.625 - x1*(1-x2**3))**2)
    x_0 = np.array([1.3, 0.7], dtype=float)
    res = minimize(phi, x_0, method='BFGS', tol=1e-6)
    r = nlo.bfgs_method(phi, x_0, np.identity(len(x_0)))
    assert np.allclose(r, res.x)


def test_nonlinear_least_squares():
    def phi(x):
        return [(np.sin(x[0]*x[1]) + 3*x[2]), (np.cos(x[0]-x[1]) * 2*x[1]**2), (np.exp(x[0]+2*x[1]) + x[2])]
    x0 = np.array([3.5, 0.8, 7], dtype=float)
    p_0 = np.array([1, 1, 1], dtype=float)
    b = np.array([2, -2, 8], dtype=float)
    r = nlo.nonlinear_least_squares(phi, x0, p_0, b)
    norm_resi = np.linalg.norm(b - phi(r))  # check the residual
    assert np.allclose(norm_resi, np.float64(0.0))
