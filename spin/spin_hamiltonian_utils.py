import scipy as sp
import scipy.sparse
from typing import Dict, Tuple, Sequence
import unittest
import common.constants

def krons_by_search(matSeq: Sequence[sp.ndarray],
          nth: int,
          i: int,
          j: int,
          value: complex,
          elems: Sequence[Tuple]) -> bool:
    """Calculate the Kronecker product of a sequence of matrices given by matSeq.

    Instead of directly calculating the Kronecker product, the code searches for non-zero matrix
    elements by filtering out zero-out matrix blocks based on the input matrices. This will be
    highly efficient if the given matrices has a significant numbers of zeros.

    return a list of non-zero matrix elements to elems in this format:
    [(i0, j0, value0), (i1, j2, value1), ...]
    """
    if nth > len(matSeq):
        print('index (nth = {}) is larger than the length of matSeq ({}).'.format(nth, len(matSeq)))
        return False
    elif nth == len(matSeq):
        elems.append((i, j, value))
        return True
    else:
        try:
            m, n = sp.array(matSeq[nth]).shape
        except: # if matSeq[nth] is not a valid 2D array
            return False
        if m != n:
            return False
        for u in range(m):
            for v in range(m):
                if abs(matSeq[nth][u, v] * value) > 1e-16:
                    # Look for non-zero matrix elements recursively
                    if not krons_by_search(matSeq,
                                           nth + 1,
                                           i * m + u,
                                           j * m + v,
                                           matSeq[nth][u, v] * value,
                                           elems): return False
        return True

def to_sparse(elems, sparse_type):
    return sparse_type(([_[2] for _ in elems],
                       ([_[0] for _ in elems], [_[1] for _ in elems])))

# -------------------------------------------- Unit Tests ------------------------------------------

class test_krons_by_search(unittest.TestCase):

    def test_empty_sequence(self):
        elems = []
        value = 1.0
        res = krons_by_search(matSeq=[], nth=0, i=0, j=0, value=value, elems=elems)
        self.assertEqual(len(elems), 1)
        self.assertEqual(res, True)

    def test_single_matrix(self):
        elems = []
        value = 1.0
        res = krons_by_search(matSeq=[common.constants.pauli_matrices[0]],
                              nth=0, i=0, j=0, value=value, elems=elems)
        matrix = sp.array(to_sparse(elems, scipy.sparse.csr_matrix).todense())
        print(matrix)
        self.assertEqual(
            sp.alltrue(
                matrix ==
                sp.array([[1, 0],
                          [0, 1]])
            ),
            True
        )
        self.assertEqual(res, True)

    def test_two_matrices_xx(self):
        """ kron(sigma_x, sigma_x) """
        elems = []
        value = 1.0
        pms = common.constants.pauli_matrices
        res = krons_by_search(matSeq=[pms[1], pms[1]],
                              nth=0, i=0, j=0, value=value, elems=elems)
        matrix = sp.array(to_sparse(elems, scipy.sparse.csr_matrix).todense())
        print(matrix)
        self.assertEqual(
            sp.alltrue(
                matrix ==
                sp.array([[0, 0, 0, 1],
                          [0, 0, 1, 0],
                          [0, 1, 0, 0],
                          [1, 0, 0, 0]])
            ),
            True
        )
        self.assertEqual(res, True)

    def test_two_matrices_yy(self):
        """ kron(sigma_y, sigma_y) """
        elems = []
        value = 1.0
        pms = common.constants.pauli_matrices
        res = krons_by_search(matSeq=[pms[2], pms[2]],
                              nth=0, i=0, j=0, value=value, elems=elems)
        matrix = sp.array(to_sparse(elems, scipy.sparse.csr_matrix).todense())
        print(matrix)
        self.assertEqual(
            sp.alltrue(
                matrix ==
                sp.array([[ 0, 0, 0,-1],
                          [ 0, 0, 1, 0],
                          [ 0, 1, 0, 0],
                          [-1, 0, 0, 0]])
            ),
            True
        )
        self.assertEqual(res, True)

    def test_two_matrices_zz(self):
        """ kron(sigma_z, sigma_z) """
        elems = []
        value = 1.0
        pms = common.constants.pauli_matrices
        res = krons_by_search(matSeq=[pms[3], pms[3]],
                              nth=0, i=0, j=0, value=value, elems=elems)
        matrix = sp.array(to_sparse(elems, scipy.sparse.csr_matrix).todense())
        print(matrix)
        self.assertEqual(
            sp.alltrue(
                matrix ==
                sp.array([[ 1, 0, 0, 0],
                          [ 0,-1, 0, 0],
                          [ 0, 0,-1, 0],
                          [ 0, 0, 0, 1]])
            ),
            True
        )
        self.assertEqual(res, True)

    def test_two_matrices_zy(self):
        """ kron(sigma_z, sigma_y) """
        elems = []
        value = 1.0
        pms = common.constants.pauli_matrices
        res = krons_by_search(matSeq=[pms[3], pms[2]],
                              nth=0, i=0, j=0, value=value, elems=elems)
        matrix = sp.array(to_sparse(elems, scipy.sparse.csr_matrix).todense())
        print(matrix)
        self.assertEqual(
            sp.alltrue(
                matrix ==
                sp.array([[ 0, -1j,  0,  0],
                          [1j,   0,  0,  0],
                          [ 0,   0,  0, 1j],
                          [ 0,   0,-1j,  0]])
            ),
            True
        )
        self.assertEqual(res, True)

    def test_two_matrices_yz(self):
        """ kron(sigma_y, sigma_z) """
        elems = []
        value = 1.0
        pms = common.constants.pauli_matrices
        res = krons_by_search(matSeq=[pms[2], pms[3]],
                              nth=0, i=0, j=0, value=value, elems=elems)
        matrix = sp.array(to_sparse(elems, scipy.sparse.csr_matrix).todense())
        print(matrix)
        self.assertEqual(
            sp.alltrue(
                matrix ==
                sp.array([[ 0,   0,-1j,  0],
                          [ 0,   0,  0, 1j],
                          [1j,   0,  0,  0],
                          [ 0, -1j,  0,  0]])
            ),
            True
        )
        self.assertEqual(res, True)

if __name__ == '__main__':
    unittest.main()
