import unittest
import numpy as np
import os

from spcqe.functions import make_basis_matrix

class TestBasis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'fixtures')

    def test_basis1(self):
        basis_a1 = np.load(os.path.join(self.directory, 'basis_a1.npy'))
        basis_b1 = make_basis_matrix(num_harmonics=10, length=500, periods=[11, 17, 23], trend=True)
        np.testing.assert_allclose(basis_a1, basis_b1, rtol=1e-5, atol=1e-8)

    def test_basis2(self):
        basis_a2 = np.load(os.path.join(self.directory, 'basis_a2.npy'))
        basis_b2 = make_basis_matrix(num_harmonics=10, length=400, periods=[11, 17], trend=False)
        np.testing.assert_allclose(basis_a2, basis_b2, rtol=1e-5, atol=1e-8)

    def test_basis3(self):
        basis_a3 = np.load(os.path.join(self.directory, 'basis_a3.npy'))
        basis_b3 = make_basis_matrix(num_harmonics=[3, 7], length=370, periods=[11, 3], standing_wave=[True, False], trend=False)
        np.testing.assert_allclose(basis_a3, basis_b3, rtol=1e-5, atol=1e-8)

    def test_basis4(self):
        basis_a4 = np.load(os.path.join(self.directory, 'basis_a4.npy'))
        basis_b4 = make_basis_matrix(num_harmonics=[4, 3], length=420, periods=[6, 8], standing_wave=True, trend=True, max_cross_k=4)
        np.testing.assert_allclose(basis_a4, basis_b4, rtol=1e-5, atol=1e-8)

if __name__ == "__main__":
    unittest.main()

