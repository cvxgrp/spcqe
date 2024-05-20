import unittest
import numpy as np
import os

from spcqe.quantiles import SmoothPeriodicQuantiles

nvals_dil = 101
my_quantiles = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]

class TestBasis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'fixtures')

    def test_transform1(self):
        sig = np.load(os.path.join(self.directory, 'pvsig.npy'))
        transformed_a1 = np.load(os.path.join(self.directory, 'transformed_data1.npy'))
        spq1 = SmoothPeriodicQuantiles(
            num_harmonics=[10,3],
            periods=[nvals_dil, 365.24225*nvals_dil],
            standing_wave=[True, False],
            trend=False,
            quantiles=my_quantiles,
            weight=10,
            problem='sequential',
            solver='qss',
            verbose=False,
        )
        spq1.fit(sig)
        transformed_b1 = spq1.transform(sig)
        np.testing.assert_allclose(transformed_a1, transformed_b1, rtol=1e-5, atol=1e-8)

    # def test_transform2(self):
    #     sig = np.load(os.path.join(self.directory, 'pvsig.npy'))
    #     transformed_a2 = np.load(os.path.join(self.directory, 'transformed_data2.npy'))
    #     spq2 = SmoothPeriodicQuantiles(
    #         num_harmonics=[10,3],
    #         periods=[nvals_dil, 365.24225*nvals_dil],
    #         standing_wave=[True, False],
    #         trend=False,
    #         quantiles=my_quantiles,
    #         weight=10,
    #         problem='sequential',
    #         solver='mosek',
    #         verbose=False,
    #     )
    #     spq2.fit(sig)
    #     transformed_b2 = spq2.transform(sig)
    #     np.testing.assert_allclose(transformed_a2, transformed_b2, rtol=1e-5, atol=1e-8)

    def test_transform3(self):
        sig = np.load(os.path.join(self.directory, 'pvsig.npy'))
        transformed_a3 = np.load(os.path.join(self.directory, 'transformed_data3.npy'))
        spq3 = SmoothPeriodicQuantiles(
            num_harmonics=[10,3],
            periods=[nvals_dil, 365.24225*nvals_dil],
            standing_wave=[True, False],
            trend=False,
            quantiles=my_quantiles,
            weight=10,
            problem='full',
            solver='mosek',
            verbose=False,
        )
        spq3.fit(sig)
        transformed_b3 = spq3.transform(sig)
        np.testing.assert_allclose(transformed_a3, transformed_b3, rtol=1e-5, atol=1e-8)

    def test_transform4(self):
        sig = np.load(os.path.join(self.directory, 'pvsig.npy'))
        transformed_a4 = np.load(os.path.join(self.directory, 'transformed_data4.npy'))
        spq4 = SmoothPeriodicQuantiles(
            num_harmonics=[10,3],
            periods=[nvals_dil, 365.24225*nvals_dil],
            standing_wave=[True, False],
            trend=False,
            quantiles=my_quantiles,
            weight=10,
            problem='full',
            solver='clarabel',
            verbose=False,
        )
        spq4.fit(sig)
        transformed_b4 = spq4.transform(sig)
        np.testing.assert_allclose(transformed_a4, transformed_b4, rtol=1e-5, atol=1e-8)

if __name__ == "__main__":
    unittest.main()
