import unittest
import numpy as np
from spcqe.functions import make_regularization_matrix
from spcqe.test_functions import regularization_matrix_3, regularization_matrix_2


class TestRegularization(unittest.TestCase):
    def test_reg_3(self):
        reg_a = make_regularization_matrix(num_harmonics=12, weight=1, periods=[23, 17, 11], trend=True)
        reg_b = regularization_matrix_3(12, 1, 23, 17, 11)
        np.testing.assert_array_equal(reg_a.data.ravel(), np.diag(reg_b))

    def test_reg_2(self):
        reg_a = make_regularization_matrix(num_harmonics=12, weight=1, periods=[11, 23], trend=True)
        reg_b = regularization_matrix_2(12, 1, 23, 11)
        np.testing.assert_array_equal(reg_a.data.ravel(), np.diag(reg_b))

    def test_multiple_harmonics(self):
        reg = make_regularization_matrix(num_harmonics=[3, 4], weight=100, periods=[17, 11], trend=False)
        self.assertEqual(reg.shape[0], 63)

    def test_multiple_harmonics_and_max_cross_k(self):
        reg = make_regularization_matrix(num_harmonics=[3, 6], weight=100, periods=[17, 11], trend=True, max_cross_k=4)
        self.assertEqual(reg.shape[0], 68)


if __name__ == "__main__":
    unittest.main()
