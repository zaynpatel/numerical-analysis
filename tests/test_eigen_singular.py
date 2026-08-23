import numpy as np
from scipy.linalg import hessenberg

from src import eigen_singular as es

def test_power_method():
    n = 8
    runs = 20
    rng = np.random.default_rng()
    A = rng.integers(low=1, high=10, size=(n, n))
    v0 = np.array([[3], [4], [5], [6], [1], [2], [7], [8]])
    es_eval, es_evec = es.power_method(A, v0, runs)
    evals, evecs = np.linalg.eig(A)
    np_dominant_eval = np.float64(evals[0])
    np_dominant_evec = np.array(evecs.real[:, 0])
    assert np.allclose(es_eval, np_dominant_eval)
    # Work around the lack of uniqueness for eigenvectors/np scaling of some evecs by -1
    assert (
        np.allclose(es_evec, np_dominant_evec.reshape(8, 1))
        or
        np.allclose(es_evec, - np_dominant_evec.reshape(8, 1))
    )

def test_inverse_iteration():
    n = 4
    alpha = 1
    A = np.array([
    [4, 2, 0, 0],
    [2, 3, 1, 0],
    [0, 1, 2, 1],
    [0, 0, 1, 1]])
    v0 = np.array([[1], [0.5], [0.3], [0.2]])

    es_eval, es_evec = es.inverse_iteration(A, v0, alpha)
    evals, evecs = np.linalg.eig(A)

    target_idx = np.argmin(np.abs(evals - alpha))
    np_closest_eval = evals[target_idx].real
    np_closest_evec = evecs[:, target_idx].real
    assert np.allclose(es_eval, np_closest_eval, rtol=1e-4)
    # Work around the lack of uniqueness for eigenvectors/np scaling of some evecs by -1
    assert (
        np.allclose(es_evec, np_closest_evec.reshape(n, 1), atol=1e-2)
        or
        np.allclose(es_evec, - np_closest_evec.reshape(n, 1), atol=1e-2)
    )

def test_rayleigh_quotient_iteration():
    n = 4
    A = np.array([
    [4, 2, 0, 0],
    [2, 3, 1, 0],
    [0, 1, 2, 1],
    [0, 0, 1, 1]])
    v0 = np.array([[1], [0.5], [0.2], [0]])

    es_eval, es_evec = es.rayleigh_quotient_iteration(A, v0)
    evals, evecs = np.linalg.eig(A)

    target_idx = np.argmin(np.abs(evals - es_eval))
    np_closest_eval = evals[target_idx].real
    np_closest_evec = evecs[:, target_idx].real
    assert np.allclose(es_eval, np_closest_eval, rtol=1e-4)
    # Work around the lack of uniqueness for eigenvectors/np scaling of some evecs by -1
    assert (
        np.allclose(es_evec, np_closest_evec.reshape(n, 1), atol=1e-2)
        or
        np.allclose(es_evec, - np_closest_evec.reshape(n, 1), atol=1e-2)
    )

def test_svd_least_squares():
    A = np.array([[4, 4],
                [1, 1],
                [9, 9],
                [4, 4],
                [-1, -1]], dtype=float)
    b = np.array([[3], [4], [1], [1], [2]], dtype=float)
    es_x = es.svd_least_squares(A, b)
    np_x, _, _, _ = np.linalg.lstsq(A, b)
    assert (
        np.allclose(es_x, np_x)
        or
        np.allclose(es_x, -np_x)
    )

def test_subdiagonal_close_to_zero_false():
    A = np.array([[1, 2, 3, 4],
                  [2, 1, 3, 5],
                  [3, 3, 1, 0],
                  [4, 5, 0, 4]], dtype=float)
    n, _ = A.shape
    r = es._subdiagonal_close_to_zero(A, n)
    assert r is False

def test_subdiagonal_close_to_zero_true():
    A = np.array([[1, 2, 3, 4],
                  [0, 1, 3, 5],
                  [0, 0, 1, 0],
                  [0, 0, 0, 4]], dtype=float)
    n, _ = A.shape
    r = es._subdiagonal_close_to_zero(A, n)
    assert r is True

def test_converts_to_upper_hessenberg():
    A = np.array([[1, 4, 7, 2],
                  [9, 4, 3, 1],
                  [1, 8, 3, 5],
                  [1, 5, 2, 0]], dtype=float)
    A = es._convert_to_upper_hessenberg(A)
    scipy_hessenberg = hessenberg(A)
    assert np.allclose(A, scipy_hessenberg)

def test_qr_eigenvalue_method():
    A = np.array([[1, 5, 2],
                  [5, 1, 8],
                  [2, 8, 3]], dtype=float)
    evals, _ = np.linalg.eig(A)
    eigs = es.qr_eigenvalue_method(A)
    assert np.allclose(np.sort(eigs), np.sort(evals.real))

def test_efficient_qr_eigenvalue_method():
    A = np.array([[1, 5, 2],
                [5, 1, 8],
                [2, 8, 3]], dtype=float)
    evals, _ = np.linalg.eig(A)
    eigs = es.efficient_qr_eigenvalue_method(A)
    assert np.allclose(np.sort(eigs), np.sort(evals.real))
