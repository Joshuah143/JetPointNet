import numpy as np


# TODO: only allow named arguments
def eta_phi_to_cartesian(eta, phi, R=1):
    # theta = 2 * np.arctan(np.exp(-eta))
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    z = R * np.sinh(eta)
    return x, y, z


# TODO: only allow named arguments
def intersection_fixed_z(eta, phi, fixed_z):
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
def calculate_delta_r(eta1, phi1, eta2, phi2):
    dphi = np.mod(phi2 - phi1 + np.pi, 2 * np.pi) - np.pi
    # dphi = np.arctan2(np.sin(phi2 - phi1), np.cos(phi2 - phi1)) should be equivalent to the above
    deta = eta2 - eta1
    return np.sqrt(deta**2 + dphi**2)
