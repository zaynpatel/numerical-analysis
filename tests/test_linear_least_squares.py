import numpy as np

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
    Q, R = lls.householder_reflections(A)
    assert np.allclose(A, Q @ R)
    assert Q.shape == (A.shape)


def test_householder_reflections_m_over_n():
    A = np.array([[3, 2, 1],
                  [9, 1, 4],
                  [5, 8, 1],
                  [1, 1, 4]], dtype=float)
    Q, R = lls.householder_reflections(A)
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


def test_givens_rotations_second():
    A = np.array([[0, -1, 1],
                  [4, 2, 0],
                  [3, 4, 0]], dtype=float)
    Q, R = lls.givens_rotations(A)
    assert np.allclose(A, Q @ R)
    m, _ = A.shape
    assert Q.shape == (m, m)
