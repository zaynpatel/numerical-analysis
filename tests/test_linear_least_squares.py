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

def test_householder_reflections():
    A = np.array([[3, 2, 1, 4],
                  [9, 1, 4, 2],
                  [5, 8, 1, 0]], dtype=float)
    Q, R = lls.householder_reflections(A)
    assert np.allclose(A, Q @ R)
