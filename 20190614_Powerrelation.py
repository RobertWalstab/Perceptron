import numpy as np
import matplotlib.pyplot as plt

# -------------------------- Demanding Parameters given by the Paper (THE DRxEAM: an integrated Photonic Thresholder ----

A           =   0.483;                          ''' coupling ratio'''
r_1         =   0.966;                          ''' coupling ratio1'''
r_2         =   0.979;                          ''' coupling ratio2'''
psi_b       =  -0.667;                          ''' additional phase bias'''
phi_0       =  -0.0481;                         ''' initial phaseshift'''
a           =   .0003;                          ''' round trip loss'''  #assumed
lmda        =   1550e-9;                        ''' free space wavelength'''
R           =   4e-4;                           ''' Radius of the MRR'''
L           =   2 * np.pi * R;                  ''' effective '''
n_2         =   7.5e-14;                        ''' refractive index'''
n_1         =   2.45;                           ''' refractive index SOI strip waveguide platform'''

P_in        =   np.arange(0, 7, 0.2);           ''' Input Power, discredited it certain steps'''
P_out       =   [.0, .0, .0];                   ''' Output Power in 3D for 3 possible stable solutions of phi'''
phi_1       =   [.0, .0, .0];                   ''' phase of Wave 1'''          #determined by future calculation
phi_2       =   [.0, .0, .0];                   ''' phase of Wave 2'''

# -------------------------- parameters depending on the MRR and coupling between MRR and MZI (WAVE 1) -----------------

t_1         =   [.0, .0, .0];                  ''' complex transmission of MRR 1'''
t_2         =   [.0, .0,.0];                   ''' complex transmission of MRR 2'''

# --------------------------- Lists to safe solution of the transcendental equation ------------------------------------

phaseshift_1=   [.0, .0, .0];                   ''' phase shift solutions 1'''
phaseshift_2=   [.0, .0, .0];                   ''' phase shift solutions 2'''

# --------------------------- buffer memory for parameters to solve transcendental equation of every wave --------------

coeff_1     =   [.0, .0, .0, .0];
coeff_2     =   [.0, .0, .0, .0];
#x           =   [.0, .0, .0, .0];

# -------------------------- parameters depending on the MRR and coupling between MRR and MZI (WAVE 1) -----------------

coeff_1[0]  = a * r_1;
coeff_1[1]  = -(a * r_1 * phi_0);
coeff_1[2]  = (1 - a * r_1)**2;
coeff_1[3]  = (-(1 - a * r_1)**2 * phi_0 - (((2 * np.pi * L * n_2) / (lmda * A)) * a**2 * (1 - r_1**2)));

# -------------------------- parameters depending on the MRR and coupling between MRR and MZI (WAVE 2) -----------------

coeff_2[0]  = a * r_2;
coeff_2[1]  = -(a * r_2 * phi_0);
coeff_2[2]  = (1 - a * r_2)**2;
coeff_2[3]  = (-(1 - a * r_2)**2 * phi_0 - (((2 * np.pi * L * n_2) / (lmda * A)) * a**2 * (1 - r_2**2)));

# -------------------------- just a bunch of variables i use as buffer memory ------------------------------------------# might be cumbersome but best way i could think of right now

x_1         = [.0, .0, .0, .0];
x_2         = [.0, .0, .0, .0];
x           = 0;
i           = 0;
ii          = 0;
t           = [.0, .0, .0];
ep         = 0;

# -------------------------- Solution I need for every P_in ------------------------------------------------------------

while(i < len(P_in)):

    # -------------------------- cubic transcendental equation and the parameters for its lorentzian approx ------------

    coeff_1[3]  = coeff_1[3] * P_in[i] / 1000;                                                                          # multiple with the increment P_in
    phi_1.append(np.roots(coeff_1))                                                                                     # solve the cubic equation for phase1 --> Equ. 4)
                                                                                                                        # write them in a list
    coeff_1[3]  = (-(1 - a * r_1)**2 * phi_0 - (((2 * np.pi * L * n_2) / (lmda * A)) * a**2 * (1 - r_1**2)));           # reset coeff_1[3] for the next value of P_in

    # -------------------------- cubic transcendental equation and the parameters for its lorentzian approx ------------

    coeff_2[3]  = coeff_2[3] * P_in[i] / 1000;                                                                          # Coeffizients for Equ. 4
    phi_2.append(np.roots(coeff_2));                                                                                    # same here just for the second MRR
    coeff_2[3]  = (-(1 - a * r_2)**2 * phi_0 - (((2 * np.pi * L * n_2) / (lmda * A)) * a**2 * (1 - r_2**2)));           # reset coeff_2[3]

    # -------------------------- Loop to calculate P_out ---------------------------------------------------------------# necessary because there are up to 3 possible solutions
    for ii in range(0, 3, 1):                                                                                           # cuz of 3 possible stable solution for EACH wave

        # to calculate t_1 from Equ. 6)
        exp             = np.exp(1j * (np.pi + phi_1[ii]));                                                             # just to seperate Equ. 5
        t[ii]           = exp * (a - (r_1 * np.exp(-1j * phi_1[ii])) / (1 - a * r_1 * np.exp(1j * phi_1[ii])));               # Equation 5 relation of energy
        t_1[ii]         = t[ii] * (phi_1[ii] * (A * P_in[i]));

        # to calculate t_2 from Equ. 6)
        exp             = np.exp(1j * (np.pi + phi_2[ii]));                                                             # same over here
        t[ii]           = exp * (a - r_2 * np.exp(-1j * phi_2[ii])) / (1 - a * r_2 * np.exp(1j * phi_2[ii]));
        t_2[ii]         = t[ii] * (phi_2[ii] * ((1 - A) * P_in[i]));

        # all for sake to solve for P_out by Equ. 6)
    exp             = np.exp(1j * psi_b);                                                                               # no reason that this is called >exp< i just need another variable
    exp             = t_1[ii] - exp;
    P_out.append(A * (1 - A) * P_in[i] * np.absolute(exp)**2);                                                          # calculated P_out - need to verify whether it is imaginary or not

    if(np.imag(P_out[ii]) != 0):
        P_out[i]   = 0;                                                                                                 # clear as it is not a stable solution
    # ------------------------------------------------------------------------------------------------------------------

    i += 1;                                                                                                             # go to the next value of P_in

plt.plot(P_out)
plt.show()

