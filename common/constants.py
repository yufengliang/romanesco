import scipy as sp

# Pauli matrices 2 x 2
p0 = sp.matrix([[1, 0],[0, 1]])
p1 = sp.matrix([[0, 1], [1, 0]])
p2 = sp.matrix([[0, -1j], [1j, 0]])
p3 = sp.matrix([[1, 0], [0, -1]])
pauli_matrices = [p0, p1, p2, p3]

# export control
__all__ = [pauli_matrices]
