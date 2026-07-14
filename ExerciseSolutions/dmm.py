import numpy as np
from scipy.linalg import lu
from scipy.linalg import solve_triangular

from src.direct_matrix_methods import forward_substitution, back_substitution

# A = np.array([[2, 4, 3, 5],
#               [-4, -7, -5, -8],
#               [6, 8, 2, 9],
#               [4, 9, -2, 14]])

# L = np.array([[1, 0, 0, 0],
#               [-2, 1, 0, 0],
#               [3, -4, 1, 0],
#               [2, 1, 3, 1]])

# U = np.array([[2, 4, 3, 5],
#               [0, 1, 1, 2],
#               [0, 0, -3, 2],
#               [0, 0, 0, -4]])

# M_1 = np.array([[1, 0, 0, 0],
#                 [2, 1, 0, 0],
#                 [-3, 0, 1, 0],
#                 [-2, 0, 0, 1]])

# M_2 = np.array([[1, 0, 0, 0],
#                 [0, 1, 0, 0],
#                 [0, 4, 1, 0],
#                 [0, -1, 0, 1]])

# M_3 = np.array([[1, 0, 0, 0],
#                 [0, 1, 0, 0],
#                 [0, 0, 1, 0],
#                 [0, 0, -3, 1]])

P = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 0, 1],
              [0, 0, 1, 0]])

A = np.array([[5, 6, 7, 8],
              [0, 4, 3, 2],
              [0, 0, 0, 1],
              [0, 0, -1, 2]])

L = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

U = np.array([[5, 6, 7, 8],
              [0, 4, 3, 2],
              [0, 0, -1, 2],
              [0, 0, 0, 1]])

b = np.array([[26], [9], [1], [-3]])
y = forward_substitution(L, b)
Ux = back_substitution(U, y)
print(Ux)
print(solve_triangular(U, b))  # scipy

print(np.linalg.solve(P @ A, b))  # This solves the system - I forgot the need for the permutation matrix originally but this works now

# print((P@A) @ Ux)
# print(b)
# P, L, U = lu(A)
# print(f"This is P: {P}")
# print(f"This is L: {L}")
# print(f"This is U: {U}")
