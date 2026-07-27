import numpy as np

from src import iterative_matrix_methods as imm

def test_check_strict_diagonal_dominance():
    A = np.array([[3, -2, 1],
                  [1, -3, 2],
                  [-1, 2, 4]], dtype=float)
    r = imm.check_strict_diagonal_dominance(A)
    assert r == False


def test_jacobi_method():
    A = np.array([[7, 3, 1],
                  [-3, 10, 2],
                  [1, 7, -15]], dtype=float)
    b = np.array([[3], [4], [2]], dtype=float)
    r = imm.jacobi_method(A, b)
    assert np.allclose(r, np.linalg.solve(A, b))
