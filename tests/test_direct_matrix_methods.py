import numpy as np

from src import direct_matrix_methods as dmm


def test_confirm_column_vector():
    b = np.array([3, 0, 1])
    r = dmm.confirm_column_vector(b)
    assert np.allclose(r, np.array([[3], [0], [1]]))


def test_confirm_upper_triangular_is_true():
    A = np.array([[2, 3, 2],
                  [0, 1, 8],
                  [0, 0, 5]])
    assert dmm.confirm_upper_triangular(A) is True


def test_confirm_upper_triangular_is_false():
    A = np.array([[2, 3, 0],
                  [1, 0, 0],
                  [2, 3, 0]])
    assert dmm.confirm_upper_triangular(A) is False


def test_export_lower_triangular():
    A = np.array([[3, 4, 2],
                  [1, 4, 2],
                  [9, 8, 2]], dtype=float)
    r = dmm.export_lower_triangular(A)
    assert np.allclose(r, np.tril(A))


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


def test_hessenberg_matrix_entries():
    A = np.array([[4, 2, 4, 3],
                  [1, -3, -6, -1],
                  [0, 3, 5, 8],
                  [0, 0, 1, 3]])
    n, _ = A.shape
    r = dmm._hessenberg_matrix_entries(n)
    assert r == 3.0


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
    assert np.allclose(my_decomp, numpy_decomp)


def test_efficient_cholesky():
    A = np.array([[3, -1, 1],
                  [-1, 3, 1],
                  [1, 1, 2]], dtype=float)
    my_decomp = dmm.efficient_cholesky(A)
    numpy_decomp = np.linalg.cholesky(A)
    assert np.allclose(my_decomp, numpy_decomp)


def test_gaussian_elimination():
    A = np.array([[1, -1, 3],
                  [1, 1, 0],
                  [3, -2, 1]], dtype=float)
    b = np.array([[2], [4], [1]], dtype=float)
    r_A, r_b = dmm.gaussian_elimination(A, b)
    comparison_A = np.array([[1, -1, 3],
                             [0, 2, -3],
                             [0, 0, -6.5]])
    comparison_b = np.array([[2], [2], [-6]])
    assert np.array_equal(r_A, comparison_A)
    assert np.array_equal(r_b, comparison_b)


def test_efficient_gaussian_elimination():
    A = np.array([[1, -1, 3],
                  [1, 1, 0],
                  [3, -2, 1]], dtype=float)
    b = np.array([[2], [4], [1]], dtype=float)
    r_A, r_b = dmm.efficient_gaussian_elimination(A, b)
    comparison_A = np.array([[1, -1, 3],
                             [0, 2, -3],
                             [0, 0, -6.5]])
    comparison_b = np.array([[2], [2], [-6]])
    assert np.array_equal(r_A, comparison_A)
    assert np.array_equal(r_b, comparison_b)


def test_gauss_backsub():
    A = np.array([[1, -1, 3],
                  [1, 1, 0],
                  [3, -2, 1]], dtype=float)
    b = np.array([[2], [4], [1]], dtype=float)
    r_A, r_b = dmm.gaussian_elimination(A, b)
    my_x = dmm.back_substitution(r_A, r_b)
    np_x = np.linalg.solve(r_A, r_b)
    assert np.allclose(my_x, np_x)


def test_lu_decomposition():
    A = np.array([[2, 4, 3, 5],
                  [-4, -7, -5, -8],
                  [6, 8, 2, 9],
                  [4, 9, -2, 14]], dtype=float)
    comparison_A = A.copy()  # Make a copy since A gets overwritten in function
    r_L, r_A = dmm.lu_decomposition(A)
    assert np.allclose(r_L @ r_A, comparison_A)


def test_lu_solve():
    A = np.array([[2, 4, 3, 5],
                  [-4, -7, -5, -8],
                  [6, 8, 2, 9],
                  [4, 9, -2, 14]], dtype=float)
    comparison_A = A.copy()
    b = np.array([[3], [1], [2], [5]])
    r = dmm.lu_solve(A, b)
    np_r = np.linalg.solve(comparison_A, b)
    assert np.allclose(r, np_r)
