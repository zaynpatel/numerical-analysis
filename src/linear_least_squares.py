"""linear_least_squares
"""
import numpy as np
import numpy.typing as npt

from src import direct_matrix_methods as dmm

def classical_gram_schmidt(A: npt.NDArray):
    """
    Computes A = QR by performing classical Gram-Schmidt (CGS)

    :param A: Matrix A with independent columns
    :type A: np.array
    :raise NotImplementedError: if the matrix does not have full column rank
    :return: Orthogonal matrix Q with orthonormal columns and triangular matrix R with scalar projection lengths

    Temporary variable assignment: i is the column number, j is the step of the QR process
    Note: It is possible to perform this with `np.linalg.qr` and receive the same magnitudes but different signs;
    the QR decomposition is not unique. Specifically the sign of r_jj is not determined uniquely so it is possible
    to have a QR decomposition with the same magnitudes but different signs
    """
    A = A.copy()
    m, n = A.shape
    if np.linalg.matrix_rank(A) != n:
        raise NotImplementedError("classical_gram_schmidt does not have support for rank-deficient matrices")
    v = np.zeros((m, n))
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v[:, j] = A[:, j]
        for i in range(j):
            R[i, j] = Q[:, i].T @ A[:, j]
            v[:, j] = v[:, j] - (R[i, j] * Q[:, i])
        R[j, j] = np.linalg.norm(v[:, j])
        Q[:, j] = v[:, j] / R[j, j]
    return Q, R


def modified_gram_schmidt(A: npt.NDArray):
    """
    Computes A = QR by performing modified Gram-Schmidt (MGS)

    :param A: Matrix A with independent columns
    :type A: np.array
    :raise NotImplementedError: if the matrix does not have full column rank
    :return: Orthogonal matrix Q with orthonormal columns and triangular matrix R with scalar projection lengths

    MGS is more numerically stable than CGS because when we compute the projection (R[i, j]) we have subtracted
    orthogonal elements from v[:, j] already

    Note: The loop on L48, L49 is making a copy of A so we could instead write `V = A.copy()`
    """
    A = A.copy()
    m, n = A.shape
    if np.linalg.matrix_rank(A) != n:
        raise NotImplementedError("`modified_gram_schmidt does not have support for rank-deficient matrices`")
    v = np.zeros((m, n))
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for i in range(n):
        v[:, i] = A[:, i]
    for i in range(n):
        R[i, i] = np.linalg.norm(v[:, i])
        Q[:, i] = v[:, i] / R[i, i]
        for j in range(i + 1, n):
            R[i, j] = Q[:, i].T @ v[:, j]
            v[:, j] = v[:, j] - (R[i, j] * Q[:, i])
    return Q, R


def householder_reflections(A: npt.NDArray):
    """
    Computes A = QR by Householder reflections

    :param A: Matrix A
    :type A: np.array
    :return Q, R, v: Orthogonal matrix Q with orthonormal columns, upper triangular matrix R, and matrix of reflectors v

    Note: This returns a full QR decomposition (as opposed to reduced QR). The dimensions of the full
    decomposition are: A = m x n, Q = m x m, R = m x n.
    """
    A = A.copy()  # This gets overwritten to become R
    m, n = A.shape
    v = np.zeros((m, n))
    Q = np.identity(m)
    for k in range(n):
        x = A[k:m, [k]]
        x_shape = x.shape[0] - 1
        e_1 = np.array([1] + ([0] * x_shape)).reshape(x_shape + 1, 1)
        v[k:m, [k]] = ((np.sign(x[0]) * np.linalg.norm(x)) * e_1) + x  # compute direction
        v[k:m, [k]] = v[k:m, [k]] / np.linalg.norm(v[k:m, [k]])  # normalize
        A[k:m, k:n] = A[k:m, k:n] - (2 * v[k:m, [k]] @ (v[k:m, [k]].T @ A[k:m, k:n]))  # project and reflect
    A[np.abs(A) <= 1e-14] = 0.0

    p = min(m, n)
    for i in range(p - 1, -1, - 1):
        Q = Q - (2 * v[:, [i]] @ (v[:, [i]].T @ Q))
    return Q, A, v


def calculate_givens_matrix(column_index: int, row_index: int, matrix: npt.NDArray):
    """
    Helper function for `givens_rotations` to calculate G given the current state of a matrix

    :param column_index: Index of the current column
    :type column_index: int
    :param row_index: Index of the current row
    :type row_index: int
    :param matrix: "In process" matrix A with Givens transformations applied
    :type matrix: np.array
    """
    m, n = matrix.shape
    i = column_index
    k = row_index
    G = np.identity(m)

    c = matrix[column_index, i] / (np.sqrt((matrix[column_index, i]  ** 2) + (matrix[k, column_index] ** 2)))
    s = - (matrix[k, column_index] / (np.sqrt((matrix[column_index, i] ** 2) + (matrix[k, column_index] ** 2))))
    s_negative = - s
    G[i, i] = c
    G[i, k] = s_negative
    G[k, i] = s
    G[k, k] = c

    return G


def givens_rotations(A: npt.NDArray):
    """
    Computes A = QR by Givens rotations

    :param A: Matrix A
    :type A: np.array
    :return: Orthogonal matrix Q with orthonormal columns and an upper triangular matrix R

    Note: It is not encouraged to compare the results of GS, Householder, and Givens since the QR factorization
    is not unqiue and each decomposition technique uses a different process to obtain QR.
    """
    A = A.copy()  # This will turn into R
    m, n = A.shape
    Q = np.identity(m)
    for col_idx in range(n):
        for row_idx in range(m - 1, -1, -1):
            if row_idx > col_idx and A[row_idx, col_idx] != 0:
                givens_rotation = calculate_givens_matrix(column_index=col_idx, row_index=row_idx, matrix=A)
                A = givens_rotation @ A
                Q = Q @ givens_rotation.T
    A[np.abs(A) <= 1e-14] = 0.0
    return Q, A


def least_squares_calculation(A: npt.NDArray, b: npt.NDArray, implicit=True):
    """
    Computes the solution to a least squares problem

    :param A: Matrix A (overdetermined) to perform least squares on
    :type A: np.array
    :param b: Column vector b
    :type b: np.array
    :param implicit: Boolean flag that specifies whether to compute Q implicitly by (2) or construct Q via (1)
    :type implicit: bool
    :return: If implicit=True x, Q, R are returned (x is the solution vector); if implicit=False then only x is returned

    There are two ways the solution can be computed:
    (1) With an explicit Q
    (2) Using the reflectors from the Householder transformation to implicitly construct Q and apply it to b

    Q is a dense matrix since most/all of its entries are non-zero so it is not recommended to use (1). But for pedagogical purposes
    it can be useful to see Q or at least be aware that the normal equations simplify to Rx = Q.T @ b and confirm this answer with the
    implicit one in (2).
    """
    dmm.confirm_column_vector(b)
    Q, R, v = householder_reflections(A)
    q_m, q_n = Q.shape
    _, r_n = R.shape
    v_m, v_n = v.shape
    if not implicit:
        new_b = Q.T @ b
        square_R = R[:r_n, :r_n]
        x = dmm.back_substitution(square_R, new_b[:r_n])
        return x, Q, R

    for k in range(v_n):
        b[k:q_m] = b[k:q_m] - (2 * v[k:v_m, [k]] @ (v[k:v_m, [k]].T @ b[k:v_m]))
    square_R = R[:r_n, :r_n]
    x = dmm.back_substitution(square_R, b[:r_n])
    return x
