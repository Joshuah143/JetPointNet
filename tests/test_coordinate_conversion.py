import unittest
import pytest
from prod.coordinate_conversions import eta_phi_to_cartesian


def test_eta_phi_to_cartesian_basic_case():
    x, y, z = eta_phi_to_cartesian(eta=0, phi=0, R=1)
    assert x == pytest.approx(1)
    assert y == pytest.approx(0)
    assert z == pytest.approx(0)


def test_eta_phi_to_cartesian_0_case():
    x, y, z = eta_phi_to_cartesian(eta=1, phi=1, R=0)
    assert x == pytest.approx(0)
    assert y == pytest.approx(0)
    assert z == pytest.approx(0)


# TODO: add more tests

if __name__ == "__main__":
    unittest.main()
