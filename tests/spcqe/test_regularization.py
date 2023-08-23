import unittest
import numpy as np
from spcqe.functions import make_regularization_matrix, regularization_matrix_3, regularization_matrix_2


class TestBasis(unittest.TestCase):
    def test_reg_3(self):
        basis_a = basis(3, 100, [11, 17, 23])
        basis_b = basis_3(3, 100, 11, 17, 23)
        np.testing.assert_array_equal(basis_a, basis_b)

    def test_reg_2(self):
        basis_a = basis(3, 100, [11, 17])
        basis_b = basis_3(3, 100, 11, 17)
        np.testing.assert_array_equal(basis_a, basis_b)
