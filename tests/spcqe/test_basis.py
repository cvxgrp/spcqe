import unittest
import numpy as np
from spcqe.functions import make_basis_matrix
from spcqe.functions_old import basis_3, basis_2


class TestBasis(unittest.TestCase):
    def test_basis_3(self):
        basis_a = make_basis_matrix(3, 100, [11, 17, 23])
        basis_b = basis_3(3, 100, 11, 17, 23)
        np.testing.assert_array_equal(basis_a, basis_b)

    def test_basis_2(self):
        basis_a = make_basis_matrix(3, 100, [11, 17])
        basis_b = basis_2(3, 100, 11, 17)
        np.testing.assert_array_equal(basis_a, basis_b)
