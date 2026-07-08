import numpy as np

from src import direct_matrix_methods as dmm


def test_confirm_column_vector():
    b = np.array([3, 0, 1])
    r = dmm.confirm_column_vector(b)
    assert np.allclose(r, np.array([[3], [0], [1]]))


def test_spd_true():
    A = np.array([[3, -1, 1],
              [-1, 3, 1],
              [1, 1, 2]], dtype=float)
    r = dmm.is_spd(A)
    assert r is True


def test_is_spd_false():
    A = np.array([[3, 0, 1],
                  [1, 5, 8],
                  [1, 2, 7]])
    r = dmm.is_spd(A)
    assert r is False


def test_back_substitution():
    A = np.array([[2, -4, 6],
              [0, 2, -2],
              [0, 0, 3]])
    b = np.array([[8], [1], [9]])
    r = dmm.back_substitution(A, b)
    assert np.allclose(r, [[2], [3.5], [3]])


def test_forward_substitution():
    A = np.array([[5, 0, 0],
              [1, 2, 0],
              [-1, 3, 2]])
    b = np.array([[15], [7], [5]])
    r = dmm.forward_substitution(A, b)
    assert np.array_equal(r, [[3], [2], [1]])


def test_cholesky_decomposition():
    A = np.array([[3, -1, 1],
              [-1, 3, 1],
              [1, 1, 2]], dtype=float)
    my_decomp = dmm.cholesky_decomposition(A)
    numpy_decomp = np.linalg.cholesky(A)
    print(my_decomp)
    print(numpy_decomp)
    assert np.allclose(my_decomp, numpy_decomp)
