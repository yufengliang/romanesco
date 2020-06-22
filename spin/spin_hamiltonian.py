import scipy as sp
from spin.spin_hamiltonian_utils import (krons_by_search, to_sparse)
import typing
import unittest
import common.constants

class SpinMatrix():
    """
    !!! no desc.
    """
    def __init__(self,
                 N: int,
                 spin_number,
                 sparse_type):
        """
        N: number of spins

        spin_number:
        a) If it is a single integer, it represents a system with only one kind of spins.
        b) If it is a list, it lists every spin number of each site in the system.

        spin_number = 2 means a spin-1/2 particle, 3 a spin-1 particle, and so forth.

        sparse_type must be one of the sparse matrix types defined in scipy.sparse.
        (https://docs.scipy.org/doc/scipy/reference/sparse.html)
        A few popular options are: csc_matrix, csr_matrix, and coo_matrix.

        """
        # number of spins
        self.N = N # !!! in future, should be made protected?

        # spins specify the angular momentum number of each spin
        if type(spin_number) == int:
            self._dim = spin_number ** self.N
            self._spin_number = [spin_number] * self.N
        elif type(spin_number) == list:
            if self.N != len(spin_number):
                raise ValueError('Number of spins ({}) and the length of spin numbers ({}) are not \
                                  equal'.format(self.N, len(spin_number)))
            self._dim = sp.prod(spin_number)
            self._spin_number = list(spin_number)
        else:
            raise ValueError('spin number must be either an integer or a list of integers, \
                              not {}.'.format(spin_number))

        self.matrix = sparse_type((self._dim, self._dim))
        self.sparse_type = sparse_type

    def get_spin(self,
                 i: int):
        return self._spin_number if type(self._spin_number) == int else self._spin_number[i]

    def add_kron_term(self,
                 inds: typing.Sequence[int],
                 mats: typing.Sequence[sp.matrix]):
        """ !!! no desc.
        Let's say:

        inds is [i, j, k, ...] (0-indexed)

        and

        mats is [mat1, mat2, mat3]

        Then the matrix product

        """
        if len(inds) > len(mats):
            raise ValueError('Length of inds({}) is larger than that of mats({}).'.
                             format(len(inds), len(mats)))
        n = len(inds)
        if len(set(inds)) < n:
            raise ValueError('Indexes in inds must be unique.')
        for i in range(n):
            nr, nc = sp.matrix(mats[i]).shape
            if nr != nc:
                raise ValueError('The mats[{}] is not square (nr = {}, nc = {}).'.format(i, nr, nc))
            spin = self.get_spin(inds[i])
            if nr != spin:
                raise ValueError('The size of mats[{}]({}) does not match the spin angular momentum number in record({}).'.format(i, nr, spin))
        order = sorted(range(n), key = lambda i : inds[i])
        i = 0
        matSeq = []
        for j in range(self.N):
            if i < len(order) and j == inds[order[i]]:
                new_mat = mats[order[i]]
                i += 1
            else:
                new_mat = sp.eye(self.get_spin(j))
            matSeq.append(new_mat)
        elems, confs = krons_by_search(matSeq)
        new_term = to_sparse(elems, self.sparse_type)
        self.matrix += new_term

# -------------------------------------------- Unit Tests ------------------------------------------

class test_SpinMatrix(unittest.TestCase):

    def test_heisenberg_couplings(self):
        sigma = common.constants.pauli_matrices
        sys = SpinMatrix(N=2, spin_number=2, sparse_type=sp.sparse.coo_matrix)
        sys.add_kron_term([0, 1], [sigma[1], sigma[1]]) # \krons(sigma_x, sigma_x)
        sys.add_kron_term([0, 1], [sigma[2], sigma[2]]) # \krons(sigma_y, sigma_y)
        sys.add_kron_term([0, 1], [sigma[3], sigma[3]]) # \krons(sigma_z, sigma_z)

        print('Heisenberg coupling: ')
        print(sys.matrix.todense())
        print()

        self.assertEqual(
            sp.alltrue(
                sys.matrix ==
                sp.array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
                          [ 0.+0.j, -1.+0.j,  2.+0.j,  0.+0.j],
                          [ 0.+0.j,  2.+0.j, -1.+0.j,  0.+0.j],
                          [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]])
            ),
            True
        )

    def test_heisenberg_couplings_spin_number_list(self):
        sigma = common.constants.pauli_matrices
        sys = SpinMatrix(N=2, spin_number=[2, 2], sparse_type=sp.sparse.coo_matrix)
        sys.add_kron_term([0, 1], [sigma[1], sigma[1]]) # \krons(sigma_x, sigma_x)
        sys.add_kron_term([0, 1], [sigma[2], sigma[2]]) # \krons(sigma_y, sigma_y)
        sys.add_kron_term([0, 1], [sigma[3], sigma[3]]) # \krons(sigma_z, sigma_z)

        print('Heisenberg coupling: ')
        print(sys.matrix.todense())
        print()

        self.assertEqual(
            sp.alltrue(
                sys.matrix ==
                sp.array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
                          [ 0.+0.j, -1.+0.j,  2.+0.j,  0.+0.j],
                          [ 0.+0.j,  2.+0.j, -1.+0.j,  0.+0.j],
                          [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]])
            ),
            True
        )

    def test_anisotropy_couplings(self):
        sigma = common.constants.pauli_matrices
        sys = SpinMatrix(N=2, spin_number=[2, 2], sparse_type=sp.sparse.csr_matrix)
        sys.add_kron_term([0, 1], [sigma[2], sigma[3]]) # \krons(sigma_y, sigma_z)

        print('Anistropy coupling: ')
        print(sys.matrix.todense())
        print()

        self.assertEqual(
            sp.alltrue(
                sys.matrix ==
                sp.array([[ 0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j],
                          [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j],
                          [ 0.+1.j,  0.+0.j,  0.+0.j,  0.+0.j],
                          [ 0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j]])
            ),
            True
        )

    def test_mixed_spins(self):
        sigma = common.constants.pauli_matrices
        sys = SpinMatrix(N=2, spin_number=[2, 3], sparse_type=sp.sparse.csr_matrix)
        rand_mat_3 = sp.matrix([[0,  1, 2],
                                [2, -1, 2],
                                [0,  0, 1]])
        sys.add_kron_term([0, 1], [sigma[2], rand_mat_3]) # \krons(sigma_y, rand_mat_3)
        sys.add_kron_term([0], [sigma[3]]) # add an on-site potential

        print('Mixed spins (1/2 and 1): ')
        print(sys.matrix.todense())
        print()

        self.assertEqual(
            sp.alltrue(
                sys.matrix ==
                sp.array([[ 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.-1.j, 0.-2.j],
                          [ 0.+0.j, 1.+0.j, 0.+0.j, 0.-2.j, 0.+1.j, 0.-2.j],
                          [ 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.-1.j],
                          [ 0.+0.j, 0.+1.j, 0.+2.j,-1.+0.j, 0.+0.j, 0.+0.j],
                          [ 0.+2.j, 0.-1.j, 0.+2.j, 0.+0.j,-1.+0.j, 0.+0.j],
                          [ 0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j, 0.+0.j,-1.+0.j]])
            ),
            True
        )

    def test_anisotropy_couplings_larger_matrix(self):
        sigma = common.constants.pauli_matrices
        sys = SpinMatrix(N=4, spin_number=2, sparse_type=sp.sparse.bsr_matrix)
        sys.add_kron_term([0, 2], [sigma[1], sigma[3]]) # \krons(sigma_x, sigma_z)
        sys.add_kron_term([0], [sigma[3]])

        print('Anistropy coupling with 4 1/2 spins: ')
        print(sys.matrix.todense())
        print()

        self.assertEqual(
            sp.alltrue(
                sys.matrix ==
                sp.array([[ 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                          [ 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                          [ 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,-1., 0., 0., 0., 0., 0.],
                          [ 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,-1., 0., 0., 0., 0.],
                          [ 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                          [ 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                          [ 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,-1., 0.],
                          [ 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,-1.],
                          [ 1., 0., 0., 0., 0., 0., 0., 0.,-1., 0., 0., 0., 0., 0., 0., 0.],
                          [ 0., 1., 0., 0., 0., 0., 0., 0., 0.,-1., 0., 0., 0., 0., 0., 0.],
                          [ 0., 0.,-1., 0., 0., 0., 0., 0., 0., 0.,-1., 0., 0., 0., 0., 0.],
                          [ 0., 0., 0.,-1., 0., 0., 0., 0., 0., 0., 0.,-1., 0., 0., 0., 0.],
                          [ 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,-1., 0., 0., 0.],
                          [ 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,-1., 0., 0.],
                          [ 0., 0., 0., 0., 0., 0.,-1., 0., 0., 0., 0., 0., 0., 0.,-1., 0.],
                          [ 0., 0., 0., 0., 0., 0., 0.,-1., 0., 0., 0., 0., 0., 0., 0.,-1.]])
            ),
            True
        )

if __name__ == '__main__':
    unittest.main()
