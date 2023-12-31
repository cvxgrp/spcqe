import unittest
import numpy as np
from spcqe.functions import make_basis_matrix
from spcqe.functions_old import basis_3, basis_2


class TestBasis(unittest.TestCase):
    def test_basis_3(self):
        basis_a = make_basis_matrix(3, 100, [11, 17, 23])
        basis_b = basis_3(3, 100, 23, 17, 11)
        np.testing.assert_array_equal(basis_a, basis_b)

    def test_basis_2(self):
        basis_a = make_basis_matrix(3, 100, [11, 17])
        basis_b = basis_2(3, 100, 17, 11)
        np.testing.assert_array_equal(basis_a, basis_b)
        self.assertEqual(basis_a.shape[1], 50)

    def test_multiple_harmonics(self):
        basis = make_basis_matrix([3, 4], 100, [11, 17])
        self.assertEqual(basis.shape[1], 64)

    def test_multiple_harmonics_and_max_cross_k(self):
        basis = make_basis_matrix([3, 6], 100, [17, 11], max_cross_k=4)
        self.assertEqual(basis.shape[1], 68)


if __name__ == "__main__":
    unittest.main()
