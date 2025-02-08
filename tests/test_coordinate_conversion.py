import unittest

import numpy as np
import pytest

from prod.utils.coordinate_conversions import (
    calculate_delta_r,
    eta_phi_to_cartesian,
    intersection_fixed_z,
)


def test_eta_phi_to_cartesian_basic_case():
    x, y, z = eta_phi_to_cartesian(eta=0, phi=0, radius=1)
    assert x == pytest.approx(1)
    assert y == pytest.approx(0)
    assert z == pytest.approx(0)


def test_eta_phi_to_cartesian_0_case():
    x, y, z = eta_phi_to_cartesian(eta=1, phi=1, radius=0)
    assert x == pytest.approx(0)
    assert y == pytest.approx(0)
    assert z == pytest.approx(0)


def test_eta_phi_to_cartesian_positive_eta_phi():
    x, y, z = eta_phi_to_cartesian(eta=1, phi=np.pi / 4, radius=2)
    assert x == pytest.approx(np.sqrt(2))  # cos(pi/4) * 2
    assert y == pytest.approx(np.sqrt(2))  # sin(pi/4) * 2
    assert z == pytest.approx(2 * np.sinh(1))


def test_eta_phi_to_cartesian_negative_eta_phi():
    x, y, z = eta_phi_to_cartesian(eta=-1, phi=-np.pi / 4, radius=1)
    assert x == pytest.approx(np.sqrt(2) / 2)  # cos(-pi/4)
    assert y == pytest.approx(-np.sqrt(2) / 2)  # sin(-pi/4)
    assert z == pytest.approx(np.sinh(-1))


def test_eta_phi_to_cartesian_large_R():
    x, y, z = eta_phi_to_cartesian(eta=0.5, phi=np.pi, radius=100)
    assert x == pytest.approx(-100)  # cos(pi) * 100
    assert y == pytest.approx(0)  # sin(pi) * 100
    assert z == pytest.approx(100 * np.sinh(0.5))


def test_eta_phi_to_cartesian_zero_eta():
    x, y, z = eta_phi_to_cartesian(eta=0, phi=np.pi / 2, radius=1)
    assert x == pytest.approx(0)  # cos(pi/2)
    assert y == pytest.approx(1)  # sin(pi/2)
    assert z == pytest.approx(0)  # sinh(0)


def test_eta_phi_to_cartesian_phi_wraparound():
    x1, y1, z1 = eta_phi_to_cartesian(eta=1, phi=0, radius=1)
    x2, y2, z2 = eta_phi_to_cartesian(eta=1, phi=2 * np.pi, radius=1)
    assert x1 == pytest.approx(x2)
    assert y1 == pytest.approx(y2)
    assert z1 == pytest.approx(z2)


# Tests for intersection_fixed_z
def test_intersection_fixed_z_basic_case():
    x, y, z = intersection_fixed_z(eta=1, phi=0, fixed_z=10)
    assert x == pytest.approx(10 / np.sinh(1))  # x
    assert y == pytest.approx(0)  # y
    assert z == pytest.approx(10)  # z


def test_intersection_fixed_z_negative_eta():
    x, y, z = intersection_fixed_z(eta=-1, phi=np.pi / 4, fixed_z=5)
    scale_factor = -5 / np.sinh(-1)
    assert x == pytest.approx(scale_factor * np.cos(np.pi / 4))  # x
    assert y == pytest.approx(scale_factor * np.sin(np.pi / 4))  # y
    assert z == pytest.approx(-5)  # z


def test_intersection_fixed_z_phi_wraparound():
    x1, y1, z1 = intersection_fixed_z(eta=1, phi=0, fixed_z=5)
    x2, y2, z2 = intersection_fixed_z(eta=1, phi=2 * np.pi, fixed_z=5)
    assert x1 == pytest.approx(x2)
    assert y1 == pytest.approx(y2)
    assert z1 == pytest.approx(z2)


def test_intersection_fixed_z_large_fixed_z():
    x, y, z = intersection_fixed_z(eta=0.5, phi=np.pi / 2, fixed_z=1e6)
    scale_factor = 1e6 / np.sinh(0.5)
    assert x == pytest.approx(0, abs=1e-6)
    assert y == pytest.approx(scale_factor)
    assert z == pytest.approx(1e6)


# Tests for calculate_delta_r
def test_calculate_delta_r_basic_case():
    result = calculate_delta_r(eta1=0, phi1=0, eta2=1, phi2=np.pi / 2)
    expected = np.sqrt(1 + (np.pi / 2) ** 2)
    assert result == pytest.approx(expected)


def test_calculate_delta_r_negative_eta_phi():
    result = calculate_delta_r(eta1=-1, phi1=-np.pi / 4, eta2=1, phi2=np.pi / 4)
    dphi = np.pi / 2
    deta = 2
    expected = np.sqrt(deta**2 + dphi**2)
    assert result == pytest.approx(expected)


def test_calculate_delta_r_phi_wraparound():
    result = calculate_delta_r(eta1=0, phi1=0, eta2=0, phi2=2 * np.pi)
    assert result == pytest.approx(0)  # Full circle


def test_calculate_delta_r_large_deta():
    result = calculate_delta_r(eta1=-100, phi1=0, eta2=100, phi2=np.pi)
    dphi = np.pi
    deta = 200
    expected = np.sqrt(deta**2 + dphi**2)
    assert result == pytest.approx(expected)


# TODO: add more tests

if __name__ == "__main__":
    unittest.main()
