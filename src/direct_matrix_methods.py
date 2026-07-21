"""direct_matrix_methods

This file contains functions to solve a matrix using direct methods (LU decomposition, Cholesky decomposition, perform back or forward substitution)
"""
import numpy as np
import numpy.typing as npt


def confirm_column_vector(b: npt.NDArray):
    if b.shape != (len(b), 1):
        if b.shape == (1, len(b)):
            b = b.transpose()
            return b
        elif b.shape == (len(b),):
            b = b[:, np.newaxis]
            return b
        else:
            raise Exception("b is not properly formatted as a column vector")
        

def confirm_upper_triangular(A: npt.NDArray):
    """Confirm a matrix is upper triangular

    :param A: Matrix to confirm upper-triangularness
    :type A: np.array
    :return: True if upper triangular, False if not upper triangular

    TODO: Consider how this can be implemented more efficiently
    """
    for row_number, column in enumerate(A):
        for column_number, _ in enumerate(column):
            if row_number > column_number and A[row_number, column_number] != 0:
                return False
    return True


def export_lower_triangular(A: npt.NDArray):
    """Returns the original matrix A with only the lower triangular portion and the other entries are zero"""
    for row_number, column in enumerate(A):
        for column_number, _ in enumerate(column):
            if row_number < column_number and A[row_number][column_number] != 0:
                A[row_number, column_number] = 0
    return A


def is_spd(A: npt.NDArray) -> bool:
    """Checks if a matrix is symmetric, positive definite"""
    if not np.array_equal(A, A.transpose()):
        return False
    eig_vals, _ = np.linalg.eig(A)
    zero_eig_val = np.any(np.isclose(eig_vals.all(), 0))
    if zero_eig_val:
        return False
    return True


def hessenberg_matrix_entries(n: int) -> int:
    """Computes the number of zero entries in an upper Hessenberg matrix
    
    Add explanation and reason for the closed-form formula b/c of sum of triangular numbers
    """
    total = n ** 2
    non_zero = (n * (n + 1) / 2) - (n - 1)
    zero = total - non_zero
    return zero


def back_substitution(A: npt.NDArray, b: npt.NDArray):
    """
    Perform back substitution given an upper triangular matrix A and a vector b

    :param A: Upper triangular matrix A
    :type A: np.array
    :param b: Column vector b
    :type b: np.array
    :return: Solution vector (referred to as x)
    """
    for row_number, column in enumerate(A):
        for column_number, _ in enumerate(column):
            if row_number > column_number and A[row_number][column_number] != 0:
                raise Exception("A is not upper triangular")
    
    confirm_column_vector(b)
    _, n = A.shape
    x = np.zeros((len(b), 1))
    vn, _ = x.shape
    x[vn - 1] = (b[vn -1] / A[vn -1][vn -1])
    for k in range(n - 2, -1, -1):
        x[k] = (b[k] - A[k][k+1:n] @ x[k+1:n]) / A[k][k]
    return x


def forward_substitution(A: npt.NDArray, b: npt.NDArray):
    """
    Perform forward substitution given a lower triangular matrix A and a vector b

    :param A: Lower triangular matrix A
    :type A: np.array
    :param b: Column vector b
    :type b: np.array
    :return: Solution vector (referred to as x)
    """
    for row_number, column in enumerate(A):
        for column_number, _ in enumerate(column):
            if row_number < column_number and A[row_number][column_number] != 0:
                raise Exception("A is not lower triangular")

    confirm_column_vector(b)
    _, n = A.shape
    x = np.zeros((len(b), 1))
    x[0] = b[0] / A[0][0]
    for k in range(1, n):
        x[k] = (b[k] - np.dot(A[k][0:k], x[0:k])) / A[k][k]
    return x


def cholesky_decomposition(A: npt.NDArray):
    """
    Compute the cholesky factor of a symmetric, positive definite matrix A

    :param A: SPD matrix A
    :type A: np.array
    :return: Lower triangular matrix A which is the Cholesky factor; transpose this to get the other factor
    """
    if not is_spd(A):
        raise Exception("Matrix is not symmetric, positive definite. Cholesky does not apply.")
    A = A.copy()
    _, n = A.shape
    for column_index in range(n - 1):
        A[column_index, column_index] = np.sqrt(A[column_index, column_index])
        for row_index in range(column_index + 1, n):
            A[row_index, column_index] = A[row_index, column_index] / A[column_index, column_index]
        for sub_column_index in range(column_index + 1, n):
            for sub_row_index in range(sub_column_index, n):
                A[sub_row_index, sub_column_index] = A[sub_row_index, sub_column_index] - (A[sub_row_index, column_index] * A[sub_column_index, column_index])
    A[n - 1, n - 1] = np.sqrt(A[n - 1, n - 1])
    A = export_lower_triangular(A)
    return A


def efficient_cholesky(A: npt.NDArray):
    """
    Compute the cholesky factor of a symmetric, positive definite matrix A
    using only one for loop and vectorizing other operations

    :param A: SPD matrix A
    :type A: np.array
    :return: Lower triangular matrix A which is the Cholesky factor; transpose this to get the other factor
    """
    _, n = A.shape
    A = A.copy()
    for k in range(n - 1):
        A[k, k] = np.sqrt(A[k, k])
        A[k + 1: n, k] = A[k + 1: n, k] / A[k, k]
        A[k + 1: n, k + 1: n] = A[k + 1: n, k + 1: n] - (A[k + 1: n, [k]] @ A[k + 1: n, [k]].transpose())
    A[n - 1, n - 1] = np.sqrt(A[n - 1, n - 1])
    A = export_lower_triangular(A)
    return A


def gaussian_elimination(A: npt.NDArray, b: npt.NDArray):
    """
    Perform Gaussian elimination on a matrix A and a vector b

    :param A: n x n matrix
    :type A: np.array
    :param b: Column vector b
    :type b: np.array
    :return: Upper triangular matrix and modified column vector
    """
    confirm_column_vector(b)
    m, n = A.shape
    if m != n:
        raise NotImplementedError("`gaussian_elimination` is not implemented for non-square matrices")
    eig_vals, _ = np.linalg.eig(A)
    zero_eig_val = np.any(np.isclose(eig_vals.all(), 0))
    if zero_eig_val:
        raise NotImplementedError(f"`gaussian_elimination` is not implemented for singular matrices")
    A = A.copy()
    for column_index in range(n - 1):
        for i in range(column_index + 1, n):
            multiplier = A[i, column_index] / A[column_index, column_index]
            for j in range(column_index, n):
                A[i, j] = A[i, j] - (multiplier * A[column_index, j])
            b[i] = b[i] - (multiplier * b[column_index])
    return A, b


def efficient_gaussian_elimination(A: npt.NDArray, b: npt.NDArray):
    """
    Perform Gaussian elimination on a matrix A and vector b
    using vectorized operations
    """
    # Need to perform checks

    # TODO: Confirm why you need to have a top loop but can vectorize the rest
    pass
