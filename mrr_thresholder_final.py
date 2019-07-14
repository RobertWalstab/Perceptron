import numpy as np
import matplotlib.pyplot as plt

# Demanding Parameters given by the Paper
# (THE DRxEAM: an integrated Photonic Thresholder)

A = 0.440  # Coupling ratio
r_1         =   0.980 # ratio1
r_2         =   0.972         # coupling ratio2#
psi_b       =  -0.0138         # additional phase bias#
phi_0       =  -0.0894        # initial phaseshift#
lmda        =   1500e-9       # free space wavelength#
R           =   4e-6          # Radius of the MRR#
L           =   2 * np.pi * R # effective #
n_2         =   4.8e-17       # refractive index#
n_1         =   3.45          # refractive index SOI strip waveguide platform#
a           =   np.sqrt(np.exp(-2.4 * L))      # round trip loss#

P_in = np.arange(0, 7, 0.02)  # Input Power, discredited it certain steps#
P_out = [.0, .0, .0]  # Output Power in 3D for 3 possible stable solutions of phi
phi_1 = [.0, .0, .0]  # phase of Wave 1 determined by future calculation
phi_2 = [.0, .0, .0]  # phase of Wave 2

A_eff = 1.2e-14

# parameters depending on the MRR and coupling between MRR and MZI (WAVE 1)

t_1         =   [.0, .0, .0] # complex transmission of MRR 1#
t_2         =   [.0, .0, .0] # complex transmission of MRR

# Lists to safe solution of the transcendental equation

phaseshift_1=   [.0, .0, .0]  # phase shift solutions 1#
phaseshift_2=   [.0, .0, .0]  # phase shift solutions

# buffer memory for parameters to solve transcendental equation of every wave

coeff_1     =   [.0, .0, .0, .0]
coeff_2     =   [.0, .0, .0, .0]
# parameters depending on the MRR and coupling between MRR and MZI (WAVE 1)

coeff_1[0]  = a * r_1
coeff_1[1]  = -(a * r_1 * phi_0)
coeff_1[2]  = (1 - a * r_1)**2

# parameters depending on the MRR and coupling between MRR and MZI (WAVE 2)

coeff_2[0]  = a * r_2
coeff_2[1]  = -(a * r_2 * phi_0)
coeff_2[2]  = (1 - a * r_2)**2

# just a bunch of variables i use as buffer memory
# might be cumbersome but best way i could think of right now

t           = [.0, .0, .0]
i           = 0
x           = [.0]
phi_1_r     = 0
phi_2_r     = 0
index2=0

while (i < len(P_in)):
    coeff_1[3] = -(1 - a * r_1) ** 2 * phi_0 - (((2 * np.pi * L * n_2) / (lmda * A)) * a ** 2 * (1 - r_2 ** 2) * (1- A) * P_in[i] / (A_eff * 1000))
    coeff_2[3] = (-(1 - a * r_2) ** 2 * phi_0 - ((2 * np.pi * L * n_2) / (lmda * A)) * a ** 2 * (1 - r_1 ** 2) * A* P_in[i] / (A_eff * 1000))

    phi_1 = np.roots(coeff_1)
    phi_2 = np.roots(coeff_2)
    index1=0
    for ii in range(2, -1, -1):
        if(np.imag(phi_1[ii]) == 0):
            phi_1_r = phi_1[ii]
            index1+=1
        if(np.imag(phi_2[ii]) == 0):
            phi_2_r = phi_1[ii]
    if (index1>1):
        index2+=1
    index1=0

    phaseshift_1.append(phi_1[ii])
    phaseshift_2.append(phi_2[ii])

    # to calculate t_1 from Equ. 6)
    exp = np.exp(1j * (np.pi + phi_1_r))  # just to split up Equ. 5
    t_1.append(exp * (a - (r_1 * np.exp(-1j * phi_1_r)) / (1 - a * r_1 * np.exp(1j * phi_1_r))))  # Equation 5 relation of energy
    # t_1.append(t[i] * (phi_1_r * (A * P_in[i] / A_eff)))

    # to calculate t_2 from Equ. 6)
    exp = np.exp(1j * (np.pi + phi_2_r))  # same over here
    t_2.append(exp * (a - r_2 * np.exp(-1j * phi_2_r)) / (1 - a * r_2 * np.exp(1j * phi_2_r)))
    # t_2.append(t[i] * (phi_2_r * ((1 - A) * P_in[i] / A_eff)))

    # all for sake to solve for P_out by Equ. 6)
    exp = np.exp(1j * psi_b)  # no reason that this is called >exp< i just need another variable
    exp = t_1[i] - exp*t_2[i]
    P_out.append(A * (1 - A) * P_in[i] * np.absolute(exp)**2)

    i += 1

plt.plot(P_out)
plt.show()

print(index2)
