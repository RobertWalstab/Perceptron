""" Written by Gabriel Teuchert, adapted towards PEP8 by Olaf Groescho. """

import numpy as np
# import matplotlib.pyplot as plt


class MrrThreshold:

    def run(self, p_in):

        # Demanding Parameters given by the Paper
        # (THE DRxEAM: an integrated Photonic Thresholder)

        A = 0.440  # Coupling ratio
        r_1 = 0.975  # ratio1
        r_2 = 0.979   # coupling ratio2#
        psi_b = -0.0667   # additional phase bias#
        phi_0 = -0.0894  # initial phaseshift#
        lmda = 1500e-9  # free space wavelength#
        R = 3.9e-6    # Radius of the MRR#
        L = 2.5 * np.pi * R  # effective #
        n_2 = 4.2e-17  # refractive index#
        # n_1 = 3.45    # refractive index SOI strip waveguide platform#
        a = np.sqrt(np.exp(-2.4 * L))  # round trip loss#

        # Input Power, discredited it certain steps#
        p_in = p_in + 2.5  # arange(0, 7, 0.02)
        print('p_in='+str(p_in))
        # Output Power in 3D for 3 possible stable solutions of phi
        p_out = .0
        # phase of Wave 1 determined by future calculation
        phi_1 = [.0, .0, .0]
        phi_2 = [.0, .0, .0]  # phase of Wave 2

        A_eff = 1e-14

        # Parameters depending on the MRR
        # and coupling between MRR and MZI (WAVE 1)

        t_1 = .0  # complex transmission of MRR 1#
        t_2 = .0  # complex transmission of MRR

        # Lists to safe solution of the transcendental equation

        # phaseshift_1 = [.0, .0, .0]  # phase shift solutions 1#
        # phaseshift_2 = [.0, .0, .0]  # phase shift solutions

        # buffer memory for parameters to solve transcendental
        # equation of every wave

        coeff_1 = [.0, .0, .0, .0]
        coeff_2 = [.0, .0, .0, .0]

        # parameters depending on the MRR
        # and coupling between MRR and MZI (WAVE 1)

        coeff_1[0] = a * r_1
        coeff_1[1] = -(a * r_1 * phi_0)
        coeff_1[2] = (1 - a * r_1)**2

        # parameters depending on the MRR
        # and coupling between MRR and MZI (WAVE 2)

        coeff_2[0] = a * r_2
        coeff_2[1] = -(a * r_2 * phi_0)
        coeff_2[2] = (1 - a * r_2)**2

        # just a bunch of variables i use as buffer memory
        # might be cumbersome but best way i could think of right now

        # t = [.0, .0, .0]
        # i = 0
        # x = [.0]
        phi_1_r = .0
        phi_2_r = .0
        # index2 = 0

        # while (i < len(p_in)):
        coeff_1[3] = -(1 - a * r_1)**2 * phi_0 - (((2 * np.pi * L * n_2)
                                                   / (lmda * A))
                                                  * a ** 2 * (1 - r_2**2)
                                                  * (1 - A) * p_in
                                                  / (A_eff * 1000))
        coeff_2[3] = (-(1 - a * r_2)**2 * phi_0 - ((2 * np.pi * L * n_2)
                                                   / (lmda * A))
                      * a**2 * (1 - r_1 ** 2) * A * p_in
                      / (A_eff * 1000))

        phi_1 = np.roots(coeff_1)
        phi_2 = np.roots(coeff_2)
        # index1 = 0
        for i in range(2, -1, -1):
            if(np.imag(phi_1[i]) == 0):
                phi_1_r = phi_1[i]
                # index1 += 1
            if(np.imag(phi_2[i]) == 0):
                phi_2_r = phi_1[i]
        # if (index1 > 1):
        #     index2 += 1
        # index1 = 0

        # phaseshift_1.append(phi_1[ii])
        # phaseshift_2.append(phi_2[ii])

        # to calculate t_1 from Equ. 6)
        exp = np.exp(1j * (np.pi + phi_1_r))  # just to split up Equ. 5
        t_1 = (exp * (a - (r_1 * np.exp(-1j * phi_1_r))
                      / (1 - a * r_1 * np.exp(1j * phi_1_r))))
        # Equation 5 relation of energy
        # t_1.append(t[i] * (phi_1_r * (A * p_in[i] / A_eff)))

        # to calculate t_2 from Equ. 6)
        exp = np.exp(1j * (np.pi + phi_2_r))  # same over here
        t_2 = (exp * (a - r_2 * np.exp(-1j * phi_2_r))
               / (1 - a * r_2 * np.exp(1j * phi_2_r)))
        # t_2.append(t[i] * (phi_2_r * ((1 - A) * p_in[i] / A_eff)))

        # all for sake to solve for p_out by Equ. 6)
        # no reason that this is called >exp< i just need another variable
        exp = np.exp(1j * psi_b)
        exp = t_1 - exp * t_2
        p_out = (A * (1 - A) * p_in * np.absolute(exp)**2)

        # i += 1
        # plt.plot(p_out)
        # plt.show()
        # print(index2)
        print('pout='+str(p_out))
        return p_out
