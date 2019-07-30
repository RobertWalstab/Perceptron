""" Written by Gabriel Teuchert"""

import numpy as np


class MrrThreshold:
    
P_in       # Input Power, discredited it certain steps

A           =   0.455                          # coupling ratio
r_1         =   0.987                          # coupling ratio1
r_2         =   0.979                          # coupling ratio2
psi_b       =  -0.0667                         # additional phase bias
phi_0       =  -0.0894                         # initial phaseshift
lmda        =   1500e-9                        # free space wavelength
R           =   3.9e-6                         # Radius of the MRR
L           =   2.5 * np.pi * R                # effektive länge
n_2         =   4.2e-17                        # refractive index
n_1         =   2.45                           # refractive index SOI strip waveguide platform
a           =   np.sqrt(np.exp(-2.4 * L))      # round trip loss
P_out       =   [.0]                           # Output Power in 3D for 3 possible stable solutions of phi
phi_1       =   [.0, .0, .0]                   # phase of Wave 1          #determined by future calculation
phi_2       =   [.0, .0, .0]                   # phase of Wave 2

A_eff       = 1.4e-14

# -------------------------- parameters depending on the MRR and coupling between MRR and MZI (WAVE 1) -----------------

t_1         =   .0                            # complex transmission of MRR 1
t_2         =   .0                            # complex transmission of MRR 2

# --------------------------- buffer memory for parameters to solve transcendental equation of every wave --------------

coeff_1     =   [.0, .0, .0, .0]
coeff_2     =   [.0, .0, .0, .0]
# -------------------------- parameters depending on the MRR and coupling between MRR and MZI (WAVE 1) -----------------

coeff_1[0]  = a * r_1
coeff_1[1]  = -(a * r_1 * phi_0)
coeff_1[2]  = (1 - a * r_1)**2

# -------------------------- parameters depending on the MRR and coupling between MRR and MZI (WAVE 2) -----------------

coeff_2[0]  = a * r_2
coeff_2[1]  = -(a * r_2 * phi_0)
coeff_2[2]  = (1 - a * r_2)**2

# -------------------------- just a bunch of variables I use as buffer memory -----------------------------------------

t           = [.0, .0, .0]
phi_1_r     = .0
phi_2_r     = .0
# i         = 0


coeff_1[3] = -(1 - a * r_1) ** 2 * phi_0 - (
    ((2 * np.pi * L * n_2) / (lmda * A)) * a ** 2 * (1 - r_2 ** 2) * (1- A) * P_in / (A_eff * 1000))

coeff_2[3] = (-(1 - a * r_2) ** 2 * phi_0 - 
    ((2 * np.pi * L * n_2) / (lmda * A)) * a ** 2 * (1 - r_1 ** 2) * A * P_in / (A_eff * 1000))

phi_1 = np.roots(coeff_1)
phi_2 = np.roots(coeff_2)

for i in range(0, 2, 1):
    if(np.imag(phi_1[i]) == 0):
        phi_1_r = phi_1[i]
    if(np.imag(phi_2[i]) == 0):
        phi_2_r = phi_2[i]

# to calculate t_1 from Equ. 6)
exp = np.exp(1j * (np.pi + phi_1_r))  # just to split up Equ. 5
t_1 = exp * (a - (r_1 * np.exp(-1j * phi_1_r)) / (1 - a * r_1 * np.exp(1j * phi_1_r))))  # Equation 5 relation of energy
# to calculate t_2 from Equ. 6)
exp = np.exp(1j * (np.pi + phi_2_r))
t_2 = exp * (a - r_2 * np.exp(-1j * phi_2_r)) / (1 - a * r_2 * np.exp(1j * phi_2_r)))

# all for sake to solve for P_out by Equ. 6)
exp = np.exp(1j * psi_b)  # no reason that this is called >exp< i just need another variable
exp = t_1 - exp * t_2
P_out = A * (1 - A) * P_in * np.absolute(exp)**2

return P_out
