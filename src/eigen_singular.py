"""Implements algorithms to find eigenvectors and eigenvalues iteratively

Algorithms include:
- Power method (find dominant eigenpair)
- Shift-and-inverse iteration (find closest eigenpair to parameter alpha)
- Rayleigh quotient iteration (find closest eigenpair to previous eigenvalue)
- Least squares for rank-deficient matrices
- QR eigenvalue algorithm
"""
import numpy as np
import numpy.typing as npt

from src import linear_least_squares as lls


def _convert_to_upper_hessenberg(A: npt.NDArray) -> npt.NDArray:
    """
    Convert a matrix A into upper Hessenberg form (zeros below the first subdiagonal)

    Note: In this context we are converting a matrix A into upper Hessenberg form as the
    first step in the QR eigenvalue algorithm. We take a matrix A (which is possibly dense)
    and apply Householder transformations to every column except the last two so that we
    get a matrix with zeros below the first subdiagonal. When we convert a matrix A into
    upper Hessenberg form it reduces the computational cost of the QR eigenvalue algorithm.

    :param A: Input matrix A
    :type A: ndarray
    :return: upper Hessenberg matrix A
    :rtype: ndarray
    """
    n, _ = A.shape
    for k in range(n - 2):  # the "upper hessenberg part" is from applying these transformations to every column besides the last two
        z = A[k + 1: n, [k]]  # keep the k+1:n rows (don't touch the rows we don't want to transform to upper hessenberg) of the k'th column
        e_1 = [1] + ([0] * (n-k-2))
        e_1 = np.array(e_1).reshape((n-k-1), 1)
        u = z + np.sign(z[0]) * np.linalg.norm(z) * e_1  # compute direction
        u = u / np.linalg.norm(u)  # normalize
        A[k + 1:n, k:n] = A[k+1:n, k:n] - (2 * u @ u.T @ (A[k+1:n, k:n]))  # Let Q = I - 2*(u @ u.T) and see that this is AQ
        A[:n, k+1: n] = A[:n, k+1: n] - (2 * A[:n, k+1: n] @ (u @ u.T))  # Let Q = I - 2*(u @ u.T) and see that this is QA
    A[np.abs(A) <= 1e-15] = 0.0
    return A


def _subdiagonal_close_to_zero(A: npt.NDArray, n: int) -> bool:
    """
    Extract the subdiagonal of a matrix and check if it is close to zero
    If it is then return True, False if it is not.

    :param A: Input matrix A
    :type A: ndarray
    :param n: Number of columns in the matrix
    :type n: int
    :return: True if the matrix's subdiagonal is close to zero, false if not
    :rtype: bool
    """
    subdiagonal = []
    for i in range(n):
        for j in range(n):
            if i > j and (i - j) == 1:
                subdiagonal.append(A[i, j])
    np_subdiagonal = np.array(subdiagonal)
    if np.allclose(np_subdiagonal, np.zeros((len(subdiagonal), 1))):
        return True
    return False


def power_method(A: npt.NDArray, v0: npt.NDArray, runs: int) -> tuple[float, npt.NDArray]:
    """
    Compute the dominant eigenpair (eigenvalue, eigenvector) using the power method

    Note: The following is theoretical intuition for why this method works:
    (a) Assume that A has n linearly independent eigenvectors s.t we have an eigenbasis.
    (b) Since we have an eigenbasis we can represent some arbitrary vector x_0 as a linear
    combination of the eigenvectors: x_0 = c_1*v_1 + c_2*v_2 + ... + c_n*v_n.
    (c) Apply A to both sides: Ax_0 = c_1*Av_1 + c_2*Av_2 + ... + c_n*Av_n which we can rewrite
    as Ax_0 = c_1*λ_1*v_1 + c_2*λ_2*v_2 + ... + c_n*λ_n*v_n by the classic eigenvector equation
    Ax = λx.
    (d) We can rewrite this same equation by bringing c_1*v_1 in front so it looks like:
    A*x_0 = c_1*λ_1 [v_1 + (c_2*λ_2)/(c_1*λ_1) + ... + (c_n*λ_n)/(c_1*λ_1)]. Let x_1 equal
    [v_1 + (c_2*λ_2)/(c_1*λ_1) + ... + (c_n*λ_n)/(c_1*λ_1)]. So we get A*x_0 = c_1*λ_1*x_1.
    (e) Next we apply A to x_1 and get an answer of the form c_2*λ_2*x_2. We then apply A to x_2
    and do this until x_k-1. As we are doing the vector x begins to look like:
    [v_1 + (c_2*λ_2^k)/(c_1*λ_1^k) + ... + (c_n*λ_n^k)/(c_1*λ_1^k)] and the fractions
    (e.g. (c_2*λ_2^k)/(c_1*λ_1^k)) will decay to zero since λ_1^k >> λ_2^k and λ_1^k >> λ_3^k
    and so on.
    (e) Thus we end up with the equation Ax_k-1 = λ_1*x_1 where λ_1*x_1 is our dominant eigenpair.

    Steps (c) and (e) show that we begin with a matrix-vector multiplication A*x_0 and we get the result
    c_1*λ_1*x_1. We multiply A*x_1 and we get c_2*λ_2*x_2. Then we multiply A*x_2 to get c_3*λ_3*x_3.
    We can also write this is: A(A((A*x_0))) and writing it this way you can clearly see the powers of A.
    These powers are obviously not explicitly computed; we do matrix-vector operations instead where A
    was present.

    For this method there is no explicit termination criteria, instead the power method is called for `runs`
    iterations and then returns the approximated dominant eigenpair.

    Source: https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter15.02-The-Power-Method.html

    :param A: Input matrix A
    :type A: ndarray
    :param v0: Initial vector guess
    :type v0: ndarray
    :param runs: Number of times to run the power method
    :type alpha: int
    """
    m, n = A.shape
    if m != n:
        raise Exception("Input matrix A needs to be square to run the power method")
    v = v0
    eig_val = None
    for _ in range(runs):
        v_tilde = np.dot(A, v)
        v = v_tilde / np.linalg.norm(v_tilde)
        eig_val = np.dot(v.T, np.dot(A, v))
    return eig_val, v


def inverse_iteration(A: npt.NDArray, v0: npt.NDArray, alpha: float):
    """
    Compute an eigenpair, not necessarily dominant, for a matrix A using the shift-and-invert method

    The goal of the shift-and-invert method is to find *an* eigenpair of A. The idea is that we begin
    with a matrix A and shift it by a scalar multiple of the identity s.t. (A - αI) is the new matrix.
    Note that α is a real number. After shifting the matrix we invert it so our final shifted-and-inverted
    matrix is (A - αI)^-1.

    In practice we do not form an explicit inverse because of the computational cost (see [1]). Instead we
    will solve a linear system involving (A - αI). This has the same effect. For example, we want to solve
    for the eigenvector so we do ṽ = (A - αI)^-1*v_k-1. Without the matrix inverse this looks like
    (A - αI)ṽ = v_k-1. We still obtain ṽ but we do it more accurately and efficiently.
    The eigenvalue is obtained by the Rayleigh quotient, using the eigenvector obtained from the linear solve.

    This iteration gives an eigenpair closest to the alpha we choose. [2] explains this using the classic
    eigenvalue equation.

    Lastly, note that α is fixed so we could perform LU on the matrix (A - αI) and back/forward solve at each iteration.
    That would speed up the method slightly.

    [1]: Reference on inverse v. linear solve:
    https://math.stackexchange.com/questions/3528736/why-is-lu-preferred-over-a-1-to-solve-matrix-equations

    [2]: Explanation about the inverse iteration eigenpair:
    https://en.wikipedia.org/wiki/Inverse_iteration#Theory_and_convergence

    :param A: Input matrix A
    :type A: ndarray
    :param v0: Initial vector guess
    :type v0: ndarray
    :param alpha: Real number value of the shift
    :type alpha: float
    :return: Eigenpair with e-val closest to alpha
    :rtype: tuple[float, ndarray]
    """
    m, n = A.shape
    if m != n:
        raise Exception("Input matrix A needs to be square to run the inverse iteration method")
    if np.array_equal(v0, np.zeros((v0.shape[0], 1))):  # avoid norm error due to zero vector initialization
        rng = np.random.default_rng()
        v0 = rng.random((v0.shape[0], 1))
    v = v0
    eig_val = None
    for _ in range(60):
        v_tilde = np.linalg.solve((A - (alpha * np.identity(m))), v)
        v = v_tilde / np.linalg.norm(v_tilde)
        eig_val = np.dot(v.T, np.dot(A, v))
    eig_val = eig_val[0][0]
    return eig_val, v


def rayleigh_quotient_iteration(A: npt.NDArray, v0: npt.NDArray):
    """
    Compute an eigenpair, not necessarily dominant, for a matrix A using the Rayleigh quotient iteration

    Note: The explanation for *how* this works is similar to `inverse_iteration` but the Rayleigh iteration
    uses a "dynamic" parameter, the eigenvalue from the previous run, every time it solves the linear system.
    This singular change to use the eigenvalue from the previous iteration increases the flops of the method
    since it involves computing a linear solve each time *but* it achieves faster convergence than the inverse
    iteration.

    :param A: Input matrix A
    :type A: ndarray
    :param v0: Initial vector guess
    :type v0: ndarray
    :return: Eigenpair with e-val closest to the initial guess
    :rtype: tuple[float, ndarray]
    """
    m, n = A.shape
    if m != n:
        raise Exception("Input matrix A needs to be square to run rayleigh quotient iteration method")
    if np.array_equal(v0, np.zeros((v0.shape[0], 1))):  # avoid norm error due to zero vector initialization
        rng = np.random.default_rng()
        v0 = rng.random((v0.shape[0], 1))
    v = v0 / np.linalg.norm(v0)
    eig_val = np.dot(v0.T, np.dot(A, v0))
    for _ in range(50):
        v_tilde = np.linalg.solve((A - (eig_val * np.identity(n))), v)
        v = v_tilde / np.linalg.norm(v_tilde)
        eig_val = np.dot(v.T, np.dot(A, v))
    eig_val = eig_val[0][0]
    return eig_val, v


def svd_least_squares(A: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
    """
    Solve least-squares via the SVD

    Note: We use this method for rank-deficient matrices only. If we want to compute least-squares
    for a matrix with full column rank we use the QR decomposition and then solve the system. A QR
    method of solving this is implemented in `linear_least_squares.least_squares_calculation`. High
    flop count is the main reason why we don't use the SVD for matrices with full column rank. You can
    read [1] to learn more about the difference in flops between QR and SVD.

    [1]: Flop count of QR v. SVD: https://math.stackexchange.com/a/73813/1657049

    :param A: Input matrix A
    :type A: ndarray
    :param b: Output vector b
    :type b: ndarray
    :return: Solution vector x
    :rtype: ndarray
    """
    ill_conditioned = 10e6
    U, sig, V_t = np.linalg.svd(A)

    if (sig[0] / sig[-1]) >= ill_conditioned:  # if condition number is too high
        r = np.where(sig == np.min(sig[sig > 0]))[0][0]  # TODO: Decide on a better cutoff than the least singular value above zero
    else:
        r = np.where(sig == sig[-1])[0][0]
    z = np.dot(U.T, b)
    y = []
    for i in range(sig.shape[0]):
        if i < r:
            y.append(z[i] / sig[i])
        else:
            y.append(np.array([0]))
    np_y = np.array(y)
    x = np.dot(V_t.T, np_y)
    return x


def qr_eigenvalue_method(A: npt.NDArray):
    """
    Finds *all* eigenvalues of a matrix A by the QR eigenvalue method

    We perform the QR eigenvalue algorithm in two steps. The first step is to reduce the matrix
    to upper Hessenberg form. This is because the second step involves QR factorizations and it
    is computationally faster to do this with an upper Hessenberg matrix as opposed to a dense A
    matrix. The second step factors A into QR then constructs RQ and does this continuously until
    we have an upper triangular matrix. Upper triangular matrices are special because all the eigenvalues
    sit on the diagonal.

    The convergence of the QR eigenvalue method is "related to the rate of decay of the subdiagonal entries
    of upper Hessenberg." The faster the entries decay to zero, the faster we get our upper triangular form
    where we can export it and see the eigenvalues.

    :param A: Input matrix A
    :type A: ndarray
    :return: Numpy array containing all eigenvalues
    :rtype: ndarray
    """
    if not np.array_equal(A, A.T):
        raise NotImplementedError("`qr_eigenvalue_method` is not implemented for nonsymmetric matrices")
    m, n = A.shape
    if m != n:
        raise Exception(f"Input matrix needs to be square instead of ({m, n})")

    A = _convert_to_upper_hessenberg(A)
    while not _subdiagonal_close_to_zero(A, n):
        Q, R, _ = lls.householder_reflections(A)
        A = R @ Q
    eigs = np.diag(A)
    return eigs


def efficient_qr_eigenvalue_method(A: npt.NDArray, alpha=0.5):
    """
    Finds *all* eigenvalues of a matrix A by the QR eigenvalue method *with shifts*

    :param A: Input matrix A
    :type A: ndarray
    :param alpha: Real number shift value
    :type alpha: float
    :return: Numpy array containing all eigenvalues
    :rtype: ndarray
    """
    if not np.array_equal(A, A.T):
        raise NotImplementedError("`qr_eigenvalue_method` is not implemented for nonsymmetric matrices")
    m, n = A.shape
    if m != n:
        raise Exception(f"Input matrix needs to be square instead of ({m, n})")

    A = _convert_to_upper_hessenberg(A)
    while not _subdiagonal_close_to_zero(A, n):
        Q, R, _ = lls.householder_reflections(A - (alpha * np.identity(n)))  # TODO: Why does `lls.householder_reflections fall into infinite loop for alpha=1`
        A = (R @ Q) + (alpha * np.identity(n))
    eigs = np.diag(A)
    return eigs
