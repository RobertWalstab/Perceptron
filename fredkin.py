""" implements a quantum optical Fredkin gate according to
'Quantum Optical Fredkin Gate', G.J. Milburn (1989) 
"""

__author__ = 'Torge / Robert'


import numpy as np
# import matplotlib.pyplot as plt

# at ideal operating conditions chi = pi
chi = np.pi


class Fredkin:

    def calculate_fredkin(self, a_i, b_i, c_i):
        ''' implements an optical model of a Fredkin gate. '''

        def coupler(in_1, in_2, phi):
            ''' implements a static beamsplitter which mixes two input fields
            equations are given in
            "A coherent Perceptron for all-Optical Learning"
            equation (8)
            '''
            out_2 = in_1 * np.cos(phi)-in_2 * np.sin(phi)
            out_1 = in_1 * np.sin(phi) + in_2 * np.cos(phi)
            return [out_1, out_2]

        # a_1 = 1/(np.sqrt(2)) * (a_i + b_i)
        # gets the coupled input values of the Fredkin gate
        a = coupler(a_i, b_i, np.pi/4)
        # a_1 = 1/(np.sqrt(2)) * (a_i + b_i),
        # describes the field in arm 1 (see Fig1)
        a_1 = a[1]
        # a_2 = 1/(np.sqrt(2)) * (a_i-b_i), describes the field in arm 2
        # (see Fig1)
        a_2 = a[0]

        # equation 4
        u = 0.5 * (np.exp(1j * (0.5 * chi) * np.conjugate(a_2) * a_2)
                   + np.exp(chi * 1j
                            * (np.conjugate(c_i) * c_i
                               + 0.5 * np.conjugate(a_1) * a_1)))
        # equation 5
        v = 0.5 * (np.exp(1j * (0.5 * chi)
                          * np.conjugate(a_2) * a_2)
                   - np.exp(chi * 1j
                            * (np.conjugate(c_i) * c_i
                               + 0.5 * np.conjugate(a_1) * a_1)))

        b_o = u * a_i - v * b_i       # equation 1
        a_o = v * a_i + u * b_i       # equation 2
        c_o = np.exp((0.1 * 1j) * np.pi * np.conjugate(c_i) * c_i + 1j
                     * np.pi * np.conjugate(a_1) * a_1) * c_i   # equation 3
        # returns the absolute value of the output fields (rounded to diminish
        # python influences)

        # prints the Logic Table for a Fredkin gate (not needed for later
        # implementation of the whole perceptron)
        # print("a_i=0, b_i=0, c_i=0",calculate_fredkin(0,0,0))
        # print("a_i=1, b_i=0, c_i=0",calculate_fredkin(1,0,0))
        # print("a_i=0, b_i=1, c_i=0",calculate_fredkin(0,1,0))
        # print("a_i=1, b_i=1, c_i=0",calculate_fredkin(1,1,0))
        # print("a_i=0, b_i=0, c_i=1",calculate_fredkin(0,0,1))
        # print("a_i=1, b_i=0, c_i=1",calculate_fredkin(1,0,1))
        # print("a_i=0, b_i=1, c_i=1",calculate_fredkin(0,1,1))
        # print("a_i=1, b_i=1, c_i=1",calculate_fredkin(1,1,1))
        return round(abs(a_o)), round(abs(b_o)), round(abs(c_o))
