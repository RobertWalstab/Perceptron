""" Perceptron testing module

This module tests the behavior of the perceptron module.
"""


__author__ = 'Olaf Groescho'


import numpy as np
import matplotlib.pyplot as plt

import perceptron


def testing_perceptron():

    p1 = perceptron.Perceptron('per1', 1, phase_stored=0.15*np.pi)
    result1 = p1.predict([1, 0])
    print('result1 = '+str(result1))

    p2 = perceptron.Perceptron('per2', 1, phase_stored=0.2*np.pi)
    result2 = p2.predict([1, 0])
    print('result2 = '+str(result2))

    p3 = perceptron.Perceptron('per3', 1, phase_stored=0.25*np.pi)
    result3 = p3.predict([1, 0])
    print('result3 = '+str(result3))

    p4 = perceptron.Perceptron('per4', 1, phase_stored=0.3*np.pi)
    result4 = p4.predict([1, 0])
    print('result4 = '+str(result4))

    p5 = perceptron.Perceptron('per5', 1, phase_stored=0.35*np.pi)
    result5 = p5.predict([1, 0])
    print('result5 = '+str(result5))

    return None


def testing_prog_amplifier():
    ''' Plots the positve and negative biases. '''
    amplifier_name = 'test_prog_gain_amp'
    prog_gain_amp = perceptron.ProgrammableAmplifier(name=amplifier_name)
    opo_phase = np.arange(-np.pi, np.pi, 0.01)
  
    def extract_bias(phase):
        storage = prog_gain_amp.storage
        storage._set_phase(phase)
        coupler = prog_gain_amp.coupler_bias
        bias_pos, bias_neg = coupler.couple(prog_gain_amp.const_bias,
                                            prog_gain_amp.var_bias,
                                            phase=storage.out())
        return bias_pos, bias_neg
    biases = list(zip(*map(extract_bias, opo_phase)))
    pos_biases = np.array(list(biases[0]))
    neg_biases = np.array(list(biases[1]))

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(opo_phase, pos_biases, label='pos biases')
    ax.plot(opo_phase, neg_biases, label='neg biases')
    ax.legend()
    return ax
