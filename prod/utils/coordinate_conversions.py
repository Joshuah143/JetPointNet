import numpy as np
from numba import njit


# TODO: only allow named arguments
def eta_phi_to_cartesian(
    eta: str | np.ndarray, phi: str | np.ndarray, radius: str | np.ndarray = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # theta = 2 * np.arctan(np.exp(-eta))

    eta = np.asarray(eta, dtype=np.float64)  # Force array
    phi = np.asarray(phi, dtype=np.float64)  # Force array
    radius = np.asarray(radius, dtype=np.float64)

    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    z = radius * np.sinh(eta)
    return x, y, z


# TODO: only allow named arguments
def intersection_fixed_z(
    eta: int | np.ndarray[np.float32],
    phi: int | np.ndarray[np.float32],
    fixed_z: int | np.ndarray[np.float32],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Warnings:
            Eta encodes the sign of z, so it should be positive for positive z and negative for negative z. The fixed_z should always be positive.

    Args:
        eta: eta of intersection, in radians, negative for negative z
        phi: the azimuthal angle of the intersection, in radians
        fixed_z: the absolute value of the z coordinate, always positive

    Returns:
        the x, y, z coordinates of the intersection point

    """
    x, y, z_unit = eta_phi_to_cartesian(eta, phi)
    scale_factor = np.sign(eta) * fixed_z / z_unit
    x *= scale_factor
    y *= scale_factor
    z = fixed_z * np.sign(eta)
    return x, y, z


# TODO: only allow named arguments
def calculate_delta_r(
    eta1: int | np.ndarray[np.float32],
    phi1: int | np.ndarray[np.float32],
    eta2: int | np.ndarray[np.float32],
    phi2: int | np.ndarray[np.float32],
) -> np.ndarray[np.float32]:
    dphi = np.mod(phi2 - phi1 + np.pi, 2 * np.pi) - np.pi
    # dphi = np.arctan2(np.sin(phi2 - phi1), np.cos(phi2 - phi1)) should be equivalent to the above
    deta = eta2 - eta1
    return np.sqrt(deta**2 + dphi**2)
