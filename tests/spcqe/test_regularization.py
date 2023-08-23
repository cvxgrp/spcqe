import unittest
import numpy as np
from spcqe.functions import make_regularization_matrix
from spcqe.functions_old import regularization_matrix_3, regularization_matrix_2


class TestBasis(unittest.TestCase):
    def test_reg_3(self):
        reg_a = make_regularization_matrix(3, 1, [11, 17, 23])
        reg_b = regularization_matrix_3(3, 1, 11, 17, 23)
        np.testing.assert_array_equal(reg_a, reg_b)

    def test_reg_2(self):
        reg_a = make_regularization_matrix(3, 1, [11, 23])
        reg_b = regularization_matrix_2(3, 1, 11, 23)
        np.testing.assert_array_equal(reg_a, reg_b)
