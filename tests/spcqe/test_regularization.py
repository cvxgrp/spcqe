import unittest
import scipy.sparse as sp
import numpy as np
import os

from spcqe.functions import make_regularization_matrix

class TestBasis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'fixtures')

    def test_basis1(self):
        reg_a1 = sp.load_npz(os.path.join(self.directory, 'reg_a1.npz'))
        reg_b1 = make_regularization_matrix(num_harmonics=10, weight=10, periods=[11, 17, 23], trend=True)
        np.testing.assert_allclose(reg_a1.toarray(), reg_b1.toarray(), rtol=1e-5, atol=1e-8)

    def test_reg2(self):
        reg_a2 = sp.load_npz(os.path.join(self.directory, 'reg_a2.npz'))
        reg_b2 = make_regularization_matrix(num_harmonics=10, weight=1, periods=[11, 17], trend=False)
        np.testing.assert_allclose(reg_a2.toarray(), reg_b2.toarray(), rtol=1e-5, atol=1e-8)

    def test_reg3(self):
        reg_a3 = sp.load_npz(os.path.join(self.directory, 'reg_a3.npz'))
        reg_b3 = make_regularization_matrix(num_harmonics=[3, 7], weight=0, periods=[11, 3], standing_wave=[True, False], trend=False)
        np.testing.assert_allclose(reg_a3.toarray(), reg_b3.toarray(), rtol=1e-5, atol=1e-8)

    def test_reg4(self):
        reg_a4 = sp.load_npz(os.path.join(self.directory, 'reg_a4.npz'))
        reg_b4 = make_regularization_matrix(num_harmonics=[4, 3], weight=3, periods=[6, 8], standing_wave=True, trend=True, max_cross_k=4)
        np.testing.assert_allclose(reg_a4.toarray(), reg_b4.toarray(), rtol=1e-5, atol=1e-8)

if __name__ == "__main__":
    unittest.main()
