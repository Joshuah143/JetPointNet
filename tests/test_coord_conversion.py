import math
import unittest
from prod.coordinate_conversions import *

class MyTestCase(unittest.TestCase):
    # Tests for eta_phi_to_cartesian(eta, phi, R=1)
    def test_eta_phi_to_cartesian_basic_case(self):
        x, y, z = eta_phi_to_cartesian(eta=0, phi=0, R=1)
        self.assertAlmostEqual(x, 1)
        self.assertAlmostEqual(y, 0)
        self.assertAlmostEqual(z, 0)

    def test_eta_phi_to_cartesian_theta_calculation_check(self):
        eta = 0.5
        phi = math.pi / 4
        R = 2
        theta = 2 * math.atan(math.exp(-eta))
        expected_x = R * math.sin(theta) * math.cos(phi)
        expected_y = R * math.sin(theta) * math.sin(phi)
        expected_z = R * math.cos(theta)
        x, y, z = eta_phi_to_cartesian(eta, phi, R)
        self.assertAlmostEqual(x, expected_x)
        self.assertAlmostEqual(y, expected_y)
        self.assertAlmostEqual(z, expected_z)

    def test_eta_phi_to_cartesian_negative_eta(self):
        x, y, z = eta_phi_to_cartesian(eta=-1, phi=math.pi/2, R=1)
        self.assertAlmostEqual(x, 0, places=7)
        self.assertLess(z, 0)

    def test_eta_phi_to_cartesian_phi_wraparound(self):
        x, y, z = eta_phi_to_cartesian(eta=0, phi=2*math.pi, R=1)
        self.assertAlmostEqual(x, 1)
        self.assertAlmostEqual(y, 0, places=7)
        self.assertAlmostEqual(z, 0)

    def test_eta_phi_to_cartesian_zero_radius(self):
        x, y, z = eta_phi_to_cartesian(eta=0, phi=0, R=0)
        self.assertAlmostEqual(x, 0)
        self.assertAlmostEqual(y, 0)
        self.assertAlmostEqual(z, 0)

    def test_eta_phi_to_cartesian_invalid_input_negative_radius(self):
        with self.assertRaises(ValueError):
            eta_phi_to_cartesian(eta=0, phi=0, R=-1)

    # Tests for intersection_fixed_z(eta, phi, fixed_z)
    def test_intersection_fixed_z_basic_intersection(self):
        result = intersection_fixed_z(eta=0, phi=0, fixed_z=0)
        self.assertIsNone(result)

    def test_intersection_fixed_z_positive_fixed_z(self):
        eta = 0.5
        fixed_z = 1
        rPerp = fixed_z / math.sinh(eta)
        x_expected = rPerp * math.cos(0)
        y_expected = rPerp * math.sin(0)
        result = intersection_fixed_z(eta, phi=0, fixed_z=fixed_z)
        self.assertIsNotNone(result)
        x, y, z = result
        self.assertAlmostEqual(x, x_expected)
        self.assertAlmostEqual(y, y_expected)
        self.assertAlmostEqual(z, fixed_z)

    def test_intersection_fixed_z_negative_fixed_z(self):
        eta = -0.5
        fixed_z = -1
        rPerp = fixed_z / math.sinh(eta)
        x_expected = rPerp * math.cos(math.pi/2)
        y_expected = rPerp * math.sin(math.pi/2)
        result = intersection_fixed_z(eta, phi=math.pi/2, fixed_z=fixed_z)
        self.assertIsNotNone(result)
        x, y, z = result
        self.assertAlmostEqual(x, x_expected)
        self.assertAlmostEqual(y, y_expected)
        self.assertAlmostEqual(z, fixed_z)

    def test_intersection_fixed_z_eta_zero(self):
        result = intersection_fixed_z(eta=0, phi=math.pi/4, fixed_z=1)
        self.assertIsNone(result)

    def test_intersection_fixed_z_invalid_input_zero_eta(self):
        result = intersection_fixed_z(eta=0, phi=0, fixed_z=1)
        self.assertIsNone(result)

    # Tests for calculate_delta_r(eta1, phi1, eta2, phi2)
    def test_calculate_delta_r_identical_points(self):
        delta_r = calculate_delta_r(eta1=0, phi1=0, eta2=0, phi2=0)
        self.assertAlmostEqual(delta_r, 0)

    def test_calculate_delta_r_phi_difference_less_than_pi(self):
        delta_r = calculate_delta_r(eta1=0, phi1=0, eta2=0, phi2=math.pi/2)
        self.assertAlmostEqual(delta_r, math.pi/2)

    def test_calculate_delta_r_phi_wraparound(self):
        delta_r = calculate_delta_r(eta1=0, phi1=math.pi, eta2=0, phi2=-math.pi)
        self.assertAlmostEqual(delta_r, 0)

    def test_calculate_delta_r_eta_difference(self):
        delta_r = calculate_delta_r(eta1=1, phi1=0, eta2=-1, phi2=0)
        self.assertAlmostEqual(delta_r, 2)

    def test_calculate_delta_r_both_differences(self):
        delta_r = calculate_delta_r(eta1=1, phi1=math.pi/4, eta2=0, phi2=-math.pi/4)
        expected_delta_r = math.hypot(1, math.pi/2)
        self.assertAlmostEqual(delta_r, expected_delta_r)

    def test_calculate_delta_r_invalid_input(self):
        with self.assertRaises(TypeError):
            calculate_delta_r(eta1=0, phi1='a', eta2=0, phi2=0)

if __name__ == '__main__':
    unittest.main()
