import numpy as np


sigma_r = 1.45e-21
sigma_n = 5.3e-27
c = 3e8  # m * s^{-1}
beta = 0.5  # TPA is small [beta] = cm * GW^{-1} (GigaWatt?)
alpha = 0.01  # 1e-17  # Unknown, free-carrier-absorbtion? [alpha] = cm^{-1}
# psi0 = 0  # Phase at z=0
n0 = 3.484  # Linear refractive index, unknown value
n2 = 1.8e-6  # Nonlinear Kerr paramerter, [n2] = m
radius = 5e-6  # Radius in m
# delta_psi  # Phase shift aquired during one round trip within the ring
l1 = 10e-6  # Lenght before coupler [l1] = m
l2 = 10e-6  # Lenght after coupler
r = 0.1  # Reflection factor
t = np.sqrt(1 - r ** 2)  # Transmission factor
tau = 1e-9  # Effective free carrier lifetime, [tau] = s
epsilon0 = 8.854187817e-12  # Intrinsic permittivity [epsilon0] = F/m
mu0 = 4 * np.pi * 1e-7  # Intrinsic permeability [mu0] = H/m
