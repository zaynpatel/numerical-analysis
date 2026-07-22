import numpy as np

from src import direct_matrix_methods as dmm
from src import linear_least_squares as lls


def test_classical_gram_schmidt_square():
    A = np.array([[3, 2, 1],
                  [9, 1, 4],
                  [5, 8, 1]], dtype=float)
    Q, R = lls.classical_gram_schmidt(A)
    assert np.allclose(A, Q @ R)


def test_modified_gram_schmidt_square():
    A = np.array([[3, 2, 1],
                  [9, 1, 4],
                  [5, 8, 1]], dtype=float)
    Q, R = lls.modified_gram_schmidt(A)
    assert np.allclose(A, Q @ R)


def test_householder_reflections():
    A = np.array([[3, 2, 1],
                  [9, 1, 4],
                  [5, 8, 1]], dtype=float)
    Q, R, _ = lls.householder_reflections(A)
    assert np.allclose(A, Q @ R)
    assert Q.shape == (A.shape)


def test_householder_reflections_m_over_n():
    A = np.array([[3, 2, 1],
                  [9, 1, 4],
                  [5, 8, 1],
                  [1, 1, 4]], dtype=float)
    Q, R, _ = lls.householder_reflections(A)
    assert np.allclose(A, Q @ R)
    m, _ = A.shape
    assert Q.shape == (m, m)


def test_givens_rotations_first():
    A = np.array([[1, -1],
                  [0, 2],
                  [1, 1]], dtype=float)
    Q, R = lls.givens_rotations(A)
    assert np.allclose(A, Q @ R)
    m, _ = A.shape
    assert Q.shape == (m, m)
    assert dmm.confirm_upper_triangular(R) == True


def test_givens_rotations_second():
    A = np.array([[0, -1, 1],
                  [4, 2, 0],
                  [3, 4, 0]], dtype=float)
    Q, R = lls.givens_rotations(A)
    assert np.allclose(A, Q @ R)
    m, _ = A.shape
    assert Q.shape == (m, m)
    assert dmm.confirm_upper_triangular(R) == True


def test_givens_rotations_third():
    A = np.array([[5, 6, 2, 9],
                  [9, 1, 4, 8],
                  [8, 3, 2, 6],
                  [1, 4, 2, 3]])
    Q, R = lls.givens_rotations(A)
    assert np.allclose(A, Q @ R)
    m, _ = A.shape
    assert Q.shape == (m, m)
    assert dmm.confirm_upper_triangular(R) == True


def test_linear_least_squares_calculation_explicit():
    A = np.array([[4, 4],
                  [1, 5],
                  [9, 2],
                  [4, 6],
                  [-1, 2]], dtype=float)
    _, n = A.shape
    b = np.array([[3], [4], [1], [1], [2]])
    x, Q, R = lls.least_squares_calculation(A, b, implicit=False)
    new_b = Q.T @ b
    npx = np.linalg.solve(R[:n, :n], new_b[:n])
    assert np.allclose(x, npx)


def test_linear_least_squares_calculation_implicit():
    A = np.array([[4, 4],
                  [1, 5],
                  [9, 2],
                  [4, 6],
                  [-1, 2]], dtype=float)
    b = np.array([[3], [4], [1], [1], [2]])
    x = lls.least_squares_calculation(A, b, implicit=True)
    npx = np.array([[-0.03282041], [0.41347472]])
    assert np.allclose(x, npx)
