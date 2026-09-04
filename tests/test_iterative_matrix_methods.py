import numpy as np

from src import iterative_matrix_methods as imm


def test_check_strict_diagonal_dominance():
    A = np.array([[3, -2, 1],
                  [1, -3, 2],
                  [-1, 2, 4]], dtype=float)
    r = imm.check_strict_diagonal_dominance(A)
    assert r is False


def test_jacobi_method():
    A = np.array([[7, 3, 1],
                  [-3, 10, 2],
                  [1, 7, -15]], dtype=float)
    b = np.array([[3], [4], [2]], dtype=float)
    r = imm.jacobi_method(A, b)
    assert np.allclose(r, np.linalg.solve(A, b))


def test_gauss_seidel_method():
    A = np.array([[7, 3, 1],
                  [-3, 10, 2],
                  [1, 7, -15]], dtype=float)
    b = np.array([[3], [4], [2]], dtype=float)
    r = imm.gauss_seidel_method(A, b)
    assert np.allclose(r, np.linalg.solve(A, b))


def test_gauss_seidel_red_black():
    A = np.array([[7, 3, 1],
                  [-3, 10, 2],
                  [1, 7, -15]], dtype=float)
    b = np.array([[3], [4], [2]], dtype=float)
    r = imm.gauss_seidel_red_black(A, b)
    assert np.allclose(r, np.linalg.solve(A, b))


def test_gauss_seidel_sor():
    A = np.array([[7, 3, 1],
                  [-3, 10, 2],
                  [1, 7, -15]], dtype=float)
    b = np.array([[3], [4], [2]], dtype=float)
    omega = 2 / (1 + np.sin(np.pi * (1/16)))
    r = imm.gauss_seidel_sor(A, b, omega=omega)
    assert np.allclose(r, np.linalg.solve(A, b))


def test_conjugate_gradient():
    A = np.array([[7, 3, 1],
                  [3, 10, 2],
                  [1, 2, 15]], dtype=float)
    b = np.array([[28], [31], [22]], dtype=float)
    r = imm.conjugate_gradient(A, b)
    exact_answer = np.array([[3], [2], [1]], dtype=float)
    assert np.allclose(r, exact_answer)


def test_preconditioned_conjugate_gradient():
    A = np.array([[7, 3, 1],
                  [3, 10, 2],
                  [1, 2, 15]], dtype=float)
    b = np.array([[28], [31], [22]], dtype=float)
    r = imm.preconditioned_conjugate_gradient(A, b)
    exact_answer = np.array([[3], [2], [1]], dtype=float)
    assert np.allclose(r, exact_answer)
