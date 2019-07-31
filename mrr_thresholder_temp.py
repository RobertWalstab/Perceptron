import numpy as np
import matplotlib.pyplot as plt

A = 0.455                              # coupling ratio
r_1 = 0.987                            # coupling ratio1
r_2 = 0.979                            # coupling ratio2
psi_b = -0.0667                        # additional phase bias
phi_0 = -0.0894                        # initial phaseshift
lmda = 1500e-9                         # free space wavelength
R = 3.9e-6                             # Radius of the MRR
L = 2.5 * np.pi * R                    # effective length
n_2 = 4.2e-17                          # refractive index
n_1 = 2.45                             # refractive index SOI strip waveguide platform
a = np.sqrt(np.exp(-2.4 * L))          # round trip loss
phi_1 = [.0, .0, .0]                   # phase of Wave 1
phi_2 = [.0, .0, .0]                   # phase of Wave 2

A_eff = 1.4e-14                        # effective area

t_1 = .0                               # complex transmission of MRR 1
t_2 = .0                               # complex transmission of MRR

coeff_1 = [.0, .0, .0, .0]
coeff_2 = [.0, .0, .0, .0]

#           coeffs for transcendental equation
coeff_1[0] = a * r_1
coeff_1[1] = -(a * r_1 * phi_0)
coeff_1[2] = (1 - a * r_1) ** 2


#           coeffs for transcendental equation
coeff_2[0] = a * r_2
coeff_2[1] = -(a * r_2 * phi_0)
coeff_2[2] = (1 - a * r_2) ** 2


def mrr_threshold(p_in):
    coeff_1[3] = -(1 - a * r_1) ** 2 * phi_0 - (
            ((2 * np.pi * L * n_2) / (lmda * A)) * a ** 2 * (1 - r_2 ** 2) * (1 - A) * p_in / (A_eff * 1000))
    coeff_2[3] = (-(1 - a * r_2) ** 2 * phi_0 - (
            ((2 * np.pi * L * n_2) / (lmda * A)) * a ** 2 * (1 - r_2 ** 2) * A * p_in / (A_eff * 1000)))

    phi_1 = np.roots(coeff_1)
    phi_2 = np.roots(coeff_2)

    phi_1_r = .0
    phi_2_r = 0

    for i in range(0, 2, 1):
        if np.imag(phi_1[i]) == 0:
                phi_1_r = phi_1[i]
        if np.imag(phi_2[i]) == 0:
                phi_2_r = phi_2[i]

    exp = np.exp(1j * (np.pi + phi_1_r))
    t_1 = (exp * (a - (r_1 * np.exp(-1j * phi_1_r)) / (1 - a * r_1 * np.exp(1j * phi_1_r))))

    exp = np.exp(1j * (np.pi + phi_2_r))
    t_2 = (exp * (a - r_2 * np.exp(-1j * phi_2_r)) / (1 - a * r_2 * np.exp(1j * phi_2_r)))

    exp = np.exp(1j * psi_b)
    exp = t_1 - exp * t_2

    return A * (1 - A) * p_in * np.absolute(exp) ** 2

# ------ Testbench -------


print(mrr_threshold(float(input())))  # .789 Highvalue - .788 Lowvalue - end of graph 5.721

p_out = [.0]

for i in range(0, 500, 2):
    x = i / 100
    p_out.append(mrr_threshold(x))

print(p_out)

plt.plot(p_out)
plt.show()
