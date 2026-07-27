import numpy as np

from src import iterative_matrix_methods as imm

def test_check_strict_diagonal_dominance():
    A = np.array([[3, -2, 1],
                  [1, -3, 2],
                  [-1, 2, 4]], dtype=float)
    r = imm.check_strict_diagonal_dominance(A)
    assert r == False
