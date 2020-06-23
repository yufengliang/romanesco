from collections import defaultdict
import scipy as sp
import scipy.sparse
from typing import Dict, Tuple, Sequence
import unittest
import common.constants

def krons_by_search(matSeq: Sequence[sp.ndarray]):
    """Calculate the Kronecker product of a sequence of matrices given by matSeq.

    Instead of directly calculating the Kronecker product, the code searches for non-zero matrix
    elements by filtering out zero-out matrix blocks based on the input matrices. This will be
    highly efficient if the given matrices has a significant numbers of zeros.
    """
    # sanity check for matSeq
    dims = []
    for i, mat in enumerate(matSeq):
        m, n = mat.shape
        if m != n:
            raise ValueError("matSeq[{}] is not a square matrix.".format(i))
        dims.append(m)
    elems = []
    search_for_elems(matSeq=matSeq, nth=0, i=0, j=0, value=1.0, elems=elems)
    return elems

def confs_by_search(dims: Sequence[int]):
    """wrap up the search for configurations"""
    confs = []
    search_for_confs(dims=dims, nth=0, conf='', confs=confs)
    return confs

def search_for_confs(dims,
                     nth: int,
                     conf: str,
                     confs: Sequence[str]):
    """Search for all the configurations"""
    if nth < 0 or nth > len(dims):
        raise ValueError('index (nth = {}) is larger than the length of dims ({}).')
    elif nth == len(dims):
        if conf: confs.append(conf)
    else:
        for u in range(dims[nth]):
            search_for_confs(dims, nth + 1, conf + str(u), confs)

def search_for_elems(matSeq: Sequence[sp.ndarray],
          nth: int,
          i: int,
          j: int,
          value: complex,
          elems: Sequence[Tuple]) -> bool:
    """Implement the actual search for non-zero matrix elements.

    Append a list of non-zero matrix elements to elems in this format:

    elems = [(i0, j0, value0), (i1, j2, value1), ...]
    """
    if nth < 0 or nth > len(matSeq):
        raise ValueError('index (nth = {}) is larger than the length \
                          of matSeq ({}).'.format(nth, len(matSeq)))
    elif nth == len(matSeq):
        if nth > 0: elems.append((i, j, value))
    else:
        m = matSeq[nth].shape[0]
        # !!! This nested loop might not be efficient if each matrix in matSeq in sparse
        # consider wrap the matrix up in a class and a get_non_zero_mat_elems method
        for u in range(m):
            for v in range(m):
                if abs(matSeq[nth][u, v] * value) > 1e-16:
                    # Look for non-zero matrix elements recursively
                    search_for_elems(matSeq,
                                     nth + 1,
                                     i * m + u,
                                     j * m + v,
                                     matSeq[nth][u, v] * value,
                                     elems)

def to_sparse(elems, sparse_type):
    return sparse_type(([_[2] for _ in elems],
                       ([_[0] for _ in elems], [_[1] for _ in elems])))

class UnionFind:

    def __init__(self, n):
        self.n = n
        self.rank = [0] * n
        self.parent = [i for i in range(n)]

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    # unite by rank
    def unite(self, i, j):
    	rootI, rootJ = self.find(i), self.find(j)
    	if rootI == rootJ: return
    	if self.rank[rootI] < self.rank[rootJ]:
    		rootI, rootJ = rootJ, rootI
    	self.parent[rootJ] = rootI
    	if self.rank[rootI] == self.rank[rootJ]:
    		self.rank[rootI] += 1

    # results sorted by the group size in descending order
    # if two group sizes are the same, then sorted according to group indices.
    def get_groups(self):
        groups = defaultdict(list)
        for i in range(self.n):
            groups[self.find(i)].append(i)
        return sorted([sorted(group) for group in groups.values()], key = lambda g : [-len(g)] + g)

# -------------------------------------------- Unit Tests ------------------------------------------

class test_krons_by_search(unittest.TestCase):

    def test_confs_by_search(self):
        self.assertEqual(confs_by_search([]), [])
        self.assertEqual(confs_by_search([2]), ['0', '1'])
        self.assertEqual(confs_by_search([2, 2]), ['00', '01', '10', '11'])
        self.assertEqual(confs_by_search([2, 3]), ['00', '01', '02', '10', '11', '12'])

    def test_empty_sequence(self):
        elems = krons_by_search(matSeq=[])
        self.assertEqual(len(elems), 0)

    def test_single_matrix(self):
        elems = krons_by_search(matSeq=[common.constants.pauli_matrices[0]])
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

    def test_two_matrices_xx(self):
        """ kron(sigma_x, sigma_x) """
        sigma = common.constants.pauli_matrices
        elems = krons_by_search(matSeq=[sigma[1], sigma[1]])
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

    def test_two_matrices_yy(self):
        """ kron(sigma_y, sigma_y) """
        sigma = common.constants.pauli_matrices
        elems = krons_by_search(matSeq=[sigma[2], sigma[2]])
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

    def test_two_matrices_zz(self):
        """ kron(sigma_z, sigma_z) """
        sigma = common.constants.pauli_matrices
        elems = krons_by_search(matSeq=[sigma[3], sigma[3]])
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

    def test_two_matrices_zy(self):
        """ kron(sigma_z, sigma_y) """
        sigma = common.constants.pauli_matrices
        elems = krons_by_search(matSeq=[sigma[3], sigma[2]])
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

    def test_two_matrices_yz(self):
        """ kron(sigma_y, sigma_z) """
        sigma = common.constants.pauli_matrices
        elems = krons_by_search(matSeq=[sigma[2], sigma[3]])
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

    def test_confs_23(self):
        """ test kron(spin 1/2, spin 1)"""
        sigma = common.constants.pauli_matrices
        elems = krons_by_search(matSeq=[sigma[2], sp.eye(3)])

class test_UnionFind(unittest.TestCase):

    def test_case_1(self):
        uf = UnionFind(5)
        uf.unite(0, 1)
        uf.unite(1, 2)
        uf.unite(3, 4)
        print(uf.get_groups())
        self.assertEqual(uf.get_groups(), [[0, 1, 2], [3, 4]])

    def test_case_2(self):
        uf = UnionFind(6)
        uf.unite(0, 5)
        uf.unite(1, 2)
        uf.unite(3, 0)
        uf.unite(5, 1)
        print(uf.get_groups())
        self.assertEqual(uf.get_groups(), [[0, 1, 2, 3, 5], [4]])

    def test_case_3(self):
        uf = UnionFind(10)
        uf.unite(5, 3)
        uf.unite(2, 1)
        uf.unite(9, 7)
        uf.unite(0, 2)
        uf.unite(6, 4)
        uf.unite(1, 5)
        uf.unite(4, 1)
        print(uf.get_groups())
        self.assertEqual(uf.get_groups(), [[0, 1, 2, 3, 4, 5, 6], [7, 9], [8]])

    def test_case_4(self):
        uf = UnionFind(6)
        uf.unite(4, 1)
        uf.unite(3, 2)
        print(uf.get_groups())
        self.assertEqual(uf.get_groups(), [[1, 4], [2, 3], [0], [5]])

    def test_case_5(self):
        uf = UnionFind(6)
        uf.unite(0, 5)
        print(uf.get_groups())
        self.assertEqual(uf.get_groups(), [[0, 5], [1], [2], [3], [4]])

if __name__ == '__main__':
    unittest.main()
