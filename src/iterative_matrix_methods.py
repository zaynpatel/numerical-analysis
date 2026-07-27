"""iterative_matrix_methods
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


def jacobi_method(A: npt.NDArray, b: npt.NDArray, tol=1e-15) -> npt.NDArray:
    """
    Compute a solution to Ax=b using the Jacobi method
    
    :param A: Input matrix A
    :type A: np.array
    :param b: Column vector b
    :type b: np.array
    :raises Exception: if A has a zero (or more) diagonal entries
    :returns: Solution vector x
    :rtype: np.array
    """
    dmm.confirm_column_vector(b)
    m, n = A.shape
    if m != n:
        raise Exception("Jacobi method only works on square matrices")

    zero_in_diagonal = check_zero_diagonal(A)
    if zero_in_diagonal:
        raise Exception("A has at least one zero in its diagonal; this method will fail to run")

    strict_diagonal_dominance = check_strict_diagonal_dominance(A)
    if not strict_diagonal_dominance:
        raise Exception("Jacobi method on the input matrix A does not converge")

    M = np.diag(np.diag(A))
    x = np.zeros((n, 1))
    x_prev = np.zeros((n, 1))

    T = np.identity(n) - (np.linalg.inv(M) @ A)  # flops?
    T_norm = spectral_radius(T)
    computed_difference = 100
    while computed_difference >= tol:
        print(computed_difference)
        for row_idx in range(n):
            mul_store = []
            for col_idx in range(n):
                if row_idx != col_idx:
                    mul_store.append(A[row_idx, col_idx] * x[row_idx])
            x_prev[row_idx, 0] = x[row_idx, 0]
            x[row_idx, 0] = (b[row_idx, 0] - (np.sum(mul_store))) / A[row_idx, row_idx]
        computed_difference = np.float64((T_norm / (1 - T_norm)) * np.linalg.norm(x - x_prev))
    return x
