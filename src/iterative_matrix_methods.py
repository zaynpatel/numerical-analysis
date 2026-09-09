"""Implements algorithms to solve Ax=b iteratively

Algorithms include:
- Gauss-Seidel and its variations (red-black, successive over-relaxation)
- Jacobi method
- Conjugate gradient and pre-conditioned conjugate gradient method
"""
import numpy as np
import numpy.typing as npt

from src import direct_matrix_methods as dmm


def check_zero_diagonal(A: npt.NDArray) -> bool:
    """
    Check if there is a zero on the diagonal of a matrix

    :param A: Input matrix A
    :type A: np.array
    :returns: True if there is a zero, False if no zero
    :rtype bool:
    """
    if 0 in np.diag(A):
        return True
    return False


def check_strict_diagonal_dominance(A: npt.NDArray) -> bool:
    """
    Check if the matrix A is strictly diagonally dominant

    Criteria: For each row, if the diagonal entry is *strictly* greater than the
    sum of the other elements in the row (not including the diagonal element)
    it is strictly diagonally dominant.

    :param A: Input matrix A
    :type A: np.array
    :return: True if the matrix is diagonally dominant, False if not
    :rtype: bool
    """
    for row_idx, row in enumerate(A):
        diag_element = np.abs(A[row_idx, row_idx])
        if diag_element <= np.sum(np.abs(row)) - diag_element:
            return False
    return True


def spectral_radius(A: npt.NDArray) -> int:
    """
    Compute the spectral radius of an input matrix A

    Definition: maximum of the absolute values of a matrix's eigenvalues
    Source: https://en.wikipedia.org/wiki/Spectral_radius
    """
    eig_vals, _ = np.linalg.eig(A)
    return np.sqrt(max(np.abs(eig_vals)))


def _check_A(A: npt.NDArray) -> None:
    """
    Checks that A:

    (a) is square
    (b) contains no zero diagonal entries
    (c) is strictly diagonally dominant

    :param A: Input matrix A
    :type A: np.array
    :raises Exception: detailed exception if there is a problem
    """
    m, n = A.shape
    if m != n:
        raise Exception("Jacobi method only works on square matrices")

    zero_in_diagonal = check_zero_diagonal(A)
    if zero_in_diagonal:
        raise Exception("A has at least one zero in its diagonal; this method will fail to run")

    strict_diagonal_dominance = check_strict_diagonal_dominance(A)
    if not strict_diagonal_dominance:
        raise Exception("Jacobi method on the input matrix A does not converge")


def inner(v1: npt.NDArray, v2: npt.NDArray) -> np.float64:
    """Compute the inner product of two column vectors and return a scalar"""
    m, n = v1.shape
    if n != 1:
        raise Exception("v1 needs to be a column vector")
    return (v1.T @ v2)[0][0]


def jacobi_method(A: npt.NDArray, b: npt.NDArray, tol=1e-15) -> npt.NDArray:
    """
    Compute a solution to Ax=b using the Jacobi method

    :param A: Input matrix A
    :type A: np.array
    :param b: Column vector b
    :type b: np.array
    :param tol: Permissible error tolerance, default is 1e-15
    :type tol: float
    :raises Exception: if A is not square, if A has a zero (or more) diagonal entries, if the matrix is not diagonally dominant
    :returns: Solution vector x
    :rtype: np.array
    """
    dmm.confirm_column_vector(b)
    _, n = A.shape
    _check_A(A)

    M = np.diag(np.diag(A))
    x = np.zeros((n, 1))
    T = np.identity(n) - (np.linalg.inv(M) @ A)  # flops?
    T_norm = spectral_radius(T)
    computed_difference = 100  # Choose an arbitrary starting difference
    while computed_difference >= tol:
        x_k = x.copy()
        for row_idx in range(n):
            mul_store = []
            for col_idx in range(n):
                if row_idx != col_idx:
                    mul_store.append(A[row_idx, col_idx] * x_k[col_idx])
            x[row_idx, 0] = (b[row_idx, 0] - (np.sum(mul_store))) / A[row_idx, row_idx]
        computed_difference = np.float64((T_norm / (1 - T_norm)) * np.linalg.norm(x - x_k))
    return x


def gauss_seidel_method(A: npt.NDArray, b: npt.NDArray, tol=1e-15) -> npt.NDArray:
    """
    Compute a solution to Ax=b using the Gauss-Seidel method

    :param A: Input matrix A
    :type A: np.array
    :param b: Column vector b
    :type b: np.array
    :param tol: Permissible error tolerance, default is 1e-15
    :type tol: float
    :raises Exception: If A is not square, if A has a zero (or more) diagonal entries, if the matrix is not diagonally dominant
    :return: Solution vector x
    :rtype: np.array
    """
    dmm.confirm_column_vector(b)
    m, n = A.shape
    _check_A(A)

    M = dmm.export_lower_triangular(A)
    x = np.zeros((n, 1))
    T = np.identity(n) - (np.linalg.inv(M) @ A)
    T_norm = spectral_radius(T)
    computed_difference = 100  # Choose an arbitrary starting difference
    while computed_difference >= tol:
        x_k = x.copy()
        for row_idx in range(n):
            j_lt_i = []
            j_gt_i = []
            for col_idx in range(n):
                if col_idx < row_idx:
                    j_lt_i.append(A[row_idx, col_idx] * x[col_idx])
                elif col_idx > row_idx:
                    j_gt_i.append(A[row_idx, col_idx] * x_k[col_idx])
            x[row_idx, 0] = (b[row_idx, 0] - (np.sum(j_lt_i) - np.sum(j_gt_i))) / A[row_idx, row_idx]
        computed_difference = np.float64((T_norm / (1 - T_norm)) * np.linalg.norm(x - x_k))
    return x


def gauss_seidel_red_black(A: npt.NDArray, b: npt.NDArray, tol=1e-15) -> npt.NDArray:
    """
    Compute a solution to Ax=b using the Gauss-Seidel method *with* red-black ordering

    This method colors each (i, j) point in a matrix red or black. A point is red if i + j is even
    and black if i + j is odd. Then, we iterate over the red points and compute the x_i^(k+1)
    values followed by iteration over the black points and compute the x_i^(k+1) values.

    :param A: Input matrix A
    :type A: np.array
    :param b: Column vector b
    :type b: np.array
    :param tol: Permissible error tolerance, default is 1e-15
    :type tol: float
    :raises Exception: If A is not square, if A has a zero (or more) diagonal entries, if the matrix is not diagonally dominant
    :return: Solution vector x
    :rtype: np.array
    """
    dmm.confirm_column_vector(b)
    m, n = A.shape
    _check_A(A)

    M = np.diag(np.diag(A))
    x = np.zeros((n, 1))
    T = np.identity(n) - (np.linalg.inv(M) @ A)
    T_norm = spectral_radius(T)

    computed_difference = 100
    while computed_difference >= tol:
        x_k = x.copy()
        for row_idx in range(0, n, 2):
            mul_store = []  # TODO: Explore using list comprehension instead of these lists
            for col_idx in range(n):
                if row_idx != col_idx:
                    print(x[col_idx])
                    mul_store.append(A[row_idx, col_idx] * x_k[col_idx])
            x[row_idx, 0] = (b[row_idx, 0] - np.sum(mul_store)) / A[row_idx, row_idx]
        for row_idx in range(1, n, 2):
            mul_s = []
            for col_idx in range(n):
                if row_idx != col_idx:
                    mul_s.append(A[row_idx, col_idx] * x[col_idx])
            x[row_idx, 0] = (b[row_idx, 0] - np.sum(mul_s)) / A[row_idx, row_idx]
        computed_difference = np.float64((T_norm / (1 - T_norm)) * np.linalg.norm(x - x_k))
    return x


def gauss_seidel_sor(A: npt.NDArray, b: npt.NDArray, omega=1.5, tol=1e-14):
    """
    Compute a solution to Ax=b using the Gauss-Seidel method *with* successive over-relaxation (SOR)

    SOR is a technique that introduces a parameter, ω (omega), as a coefficient to a GS update.
    This parameter is used to speed up convergence by calculating x_i^(k+1) with a positively
    scaled GS update.

    Symbolically SOR looks like: x_(k+1) <- (1 - ω)x_k + ω(x_k+1). A value of ω is not known before-hand,
    experimentation is needed to determine which value of ω leads to convergence. It has been shown that
    if ω is between 1 < ω < 2 then faster iteration can be obtained. Note that omega does not need to be
    a specific float value between 1 and 2. For example ω = (2 / 1 + sin(pi*h)) is a valid ω parameter
    which leads to "rapid" convergence for the A, b in the `test_gauss_seidel_method` from the
    tests/test_iterative_matrix_methods.py file. The h in that ω is a constant set to 1/16.

    :param A: Input matrix A
    :type A: np.array
    :param b: Column vector b
    :type b: np.array
    :param omega: Coefficient of GS update and current iteration, configurable parameter to be set and is used for faster iteration
    :type omega: float | function
    :param tol: Permissible error tolerance, default is 1e-14 (Note: Setting it to 1e-15 or greater results in an infinite loop)
    :type tol: float
    :raises Exception: If A is not square, if A has a zero (or more) diagonal entries, if the matrix is not diagonally dominant, if omega is less than zero
    :return: Solution vector x
    :rtype: np.array
    """
    if omega <= 0:
        raise Exception("Omega is not allowed to be less than zero.")
    m, n = A.shape
    _check_A(A)

    M = dmm.export_lower_triangular(A)
    x = np.zeros((n, 1))
    T = np.identity(n) - (omega * (np.linalg.inv(M) @ A))
    T_norm = spectral_radius(T)
    computed_difference = 100  # Choose an arbitrary starting difference
    while computed_difference >= tol:
        x_k = x.copy()
        for row_idx in range(n):
            j_lt_i = []
            j_gt_i = []
            for col_idx in range(n):
                if col_idx < row_idx:
                    j_lt_i.append(A[row_idx, col_idx] * x[col_idx])
                elif col_idx > row_idx:
                    j_gt_i.append(A[row_idx, col_idx] * x_k[col_idx])
            x[row_idx, 0] = ((1 - omega) * x_k[row_idx, 0]) + ((omega / A[row_idx, row_idx]) * (b[row_idx, 0] - np.sum(j_lt_i) - np.sum(j_gt_i)))
        computed_difference = np.float64((T_norm / (1 - T_norm)) * np.linalg.norm(x - x_k))
    return x


def conjugate_gradient(A: npt.NDArray, b: npt.NDArray, tol=1e-14) -> npt.NDArray:
    """
    Compute a solution to Ax=b using the conjugate gradient method

    :param A: Input matrix A
    :type A: np.array
    :param b: Column vector b
    :type b: np.array
    :param tol: Permissible error tolerance, default is 1e-14
    :type tol: float
    :raises Exception: If A is not symmetric positive definite
    :return: Solution vector x
    :rtype: np.array

    Source: https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
    """
    if not dmm.is_spd(A):
        raise Exception("Input matrix needs to be symmetric positive definite")
    _, n = A.shape

    x = np.zeros((n, 1))
    r = b - (A @ x)
    r_0 = b - (A @ x)
    p = r_0
    delta = inner(r, r)
    b_delta = inner(b, b)

    k = 0
    while delta > (tol ** 2) * b_delta:
        delta_k = delta.copy()
        p_k = p.copy()
        x_k = x.copy()
        r_k = r.copy()

        s = A @ p_k
        alpha = delta_k / inner(p_k, s)
        x = x_k + (alpha * p_k)
        r = r_k - (alpha * s)
        delta = inner(r, r)
        p = r + ((delta / delta_k) * p_k)
        k += 1
    return x


def preconditioned_conjugate_gradient(A: npt.NDArray, b: npt.NDArray, tol=1e-14) -> npt.NDArray:
    """
    Compute a solution to Ax=b using the preconditioned conjugate gradient method

    :param A: Input matrix A
    :type A: np.array
    :param b: Column vector b
    :type b: np.array
    :param tol: Permissible error tolerance, default is 1e-14
    :type tol: float
    :raises Exception: If A is not symmetric positive definite
    :return: Solution vector x
    :rtype: np.array
    """
    if not dmm.is_spd(A):
        raise Exception("Input matrix needs to be symmetric positive definite")
    _, n = A.shape

    x = np.zeros((n, 1))
    P = np.diag(np.diag(A))  # Diagonal matrix is the default choice of the preconditioner, for now
    r = b - (A @ x)
    r_0 = b - (A @ x)
    h = np.linalg.inv(P) @ r_0
    h_0 = np.linalg.inv(P) @ r_0
    p = h_0
    delta = inner(r_0, h)
    b_delta = inner(b, (np.linalg.inv(P) @ b))

    k = 0
    while delta > (tol ** 2) * b_delta:
        p_k = p.copy()
        delta_k = delta.copy()
        x_k = x.copy()
        r_k = r.copy()

        s = A @ p_k
        alpha = delta_k / (inner(p_k, s))
        x = x_k + (alpha * p_k)
        r = r_k - (alpha * s)
        h = (np.linalg.inv(P) @ r)
        delta = inner(r, h)
        p = h + ((delta / delta_k) * p_k)
        k += 1
    return x
