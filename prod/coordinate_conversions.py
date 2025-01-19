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
