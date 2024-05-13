import unittest
import numpy as np
from spcqe.functions import make_basis_matrix
from spcqe.test_functions import basis_3, basis_2


class TestBasis(unittest.TestCase):
    def test_basis_3(self):
        basis_a = make_basis_matrix(num_harmonics=10, length=1000, periods=[11, 17, 23], trend=True)
        basis_b = basis_3(10, 1000, 23, 17, 11)
        np.testing.assert_array_equal(basis_a, basis_b)

    def test_basis_2(self):
        basis_a = make_basis_matrix(num_harmonics=10, length=1000, periods=[11, 17], trend=True)
        basis_b = basis_2(10, 1000, 17, 11)
        np.testing.assert_array_equal(basis_a, basis_b)

    def test_multiple_harmonics(self):
        basis = make_basis_matrix(num_harmonics=[3, 4], length=100, periods=[11, 17], trend=True)
        self.assertEqual(basis.shape[1], 64)

    def test_multiple_harmonics_and_max_cross_k(self):
        basis = make_basis_matrix(num_harmonics=[3, 6], length=100, periods=[17, 11], trend=True, max_cross_k=4)
        self.assertEqual(basis.shape[1], 68)


if __name__ == "__main__":
    unittest.main()
