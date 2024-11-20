import math
import unittest
from prod.coordinate_conversions import *

class MyTestCase(unittest.TestCase):
    def test_eta_phi_to_cartesian_basic_case(self):
        x, y, z = eta_phi_to_cartesian(eta=0, phi=0, R=1)
        self.assertAlmostEqual(x, 1)
        self.assertAlmostEqual(y, 0)
        self.assertAlmostEqual(z, 0)

    def test_eta_phi_to_cartesian_0_case(self):
        x, y, z = eta_phi_to_cartesian(eta=1, phi=1, R=0)
        self.assertAlmostEqual(x, 0)
        self.assertAlmostEqual(y, 0)
        self.assertAlmostEqual(z, 0)

    # TODO: add more tests

if __name__ == '__main__':
    unittest.main()
