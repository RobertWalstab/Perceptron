""" Perceptron testing module

This module tests the behavior of the perceptron module.
"""


__author__ = 'Olaf Groescho'


import numpy as np
import matplotlib.pyplot as plt

import perceptron

p_test = perceptron.Perceptron('per0', no_of_inputs=1,
                               phase_stored=0.0*np.pi)


def testing_perceptron_1():
    ''' Generates perceptrons and test if prediction outputs result. '''

    p0 = perceptron.Perceptron('per0', no_of_inputs=1,
                               phase_stored=-0.1*np.pi)
    result0 = p0.predict(inputs=[1])
    print('result0 = '+str(result0))

    p1 = perceptron.Perceptron('per1', no_of_inputs=1,
                               phase_stored=0.15*np.pi)
    result1 = p1.predict([1, 0])
    print('result1 = '+str(result1))

    p2 = perceptron.Perceptron('per2', no_of_inputs=1,
                               phase_stored=0.2*np.pi)
    result2 = p2.predict([1, 0])
    print('result2 = '+str(result2))

    p3 = perceptron.Perceptron('per3', no_of_inputs=1,
                               phase_stored=0.25*np.pi)
    result3 = p3.predict([1, 0])
    print('result3 = '+str(result3))

    p4 = perceptron.Perceptron('per4', no_of_inputs=1,
                               phase_stored=0.3*np.pi)
    result4 = p4.predict([1, 0])
    print('result4 = '+str(result4))

    p5 = perceptron.Perceptron('per5', no_of_inputs=1,
                               phase_stored=0.35*np.pi)
    result5 = p5.predict([1, 0])
    print('result5 = '+str(result5))

    p6 = perceptron.Perceptron('per6', no_of_inputs=1,
                               phase_stored=-0.5*np.pi)
    result6 = p6.predict([1, 0])
    print('result6 = '+str(result6))

    return None


def testing_perceptron_2():
    ''' Tests generation of perceptron with 4 synapses
    and its prediction and training. '''
    p = perceptron.Perceptron(name='neu', no_of_inputs=4)
    print(p.predict([0, 1, 1, 0]))
    p.train(training_inputs=[1, 0, 1, 0], label=0)
    p.train(training_inputs=[0, 1, 1, 0], label=1)
    print(p.synapses[0].amp.storage.phase)
    print(p.synapses[1].amp.storage.phase)
    print(p.synapses[2].amp.storage.phase)
    print(p.synapses[3].amp.storage.phase)
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


def test_coupler(in_1, in_2, ax=None, p=None, c_fkt=None):
    if p is None:
        p = np.arange(-np.pi, np.pi, 0.01)
    if c_fkt is None:
        c = perceptron.Coupler('Testing Coupler')
        c_fkt = c._couple
    couple = (lambda x: c_fkt(in_1, in_2, x))
    outputs = list(zip(*map(couple, p)))
    outs_1 = np.array(list(outputs[0]))
    outs_2 = np.array(list(outputs[1]))

    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.set_title('Input1 = '+str(in_1)+', Input2 = '+str(in_2))
    ax.plot(p, outs_1, label='Output1')
    ax.plot(p, outs_2, label='Output2')
    ax.axvline(x=0.25*np.pi, label='0.25*pi')
    ax.axvline(x=0.5*np.pi, label='0.5*pi')
    ax.axvline(x=-0.25*np.pi, label='-0.25*pi')
    ax.legend()
    return ax  # , outs_1, outs_2


def couple_old(in_1, in_2, theta):
    out_1 = in_1 * np.cos(theta) - in_2 * np.sin(theta)
    out_2 = in_1 * np.sin(theta) + in_2 * np.cos(theta)
    return out_1, out_2


def couple_new(in_1, in_2, theta):
    ''' Mixes two input signals into two output signals.
    Implements the coupler.

    Parameters:
    in_1, in_2 : Complex signals.
    theta : Angle in radiant.

    Output: out_1, out_2 : Complex output signals.
    '''

    out_1 = in_1 * np.cos(theta) + in_2 * 1j * np.sin(theta)
    out_2 = in_1 * 1j * np.sin(theta) + in_2 * np.cos(theta)
    return out_1, out_2


def xi_0():
    ''' Sets constant part of bias. '''
    matching = 1.993792
    return np.sqrt(0.9 * matching)


def xi_var(radiant):
    ''' Sets variable part of bias dependent on phase. '''
    matching = 1 / (10 * 1.795)
    return np.sqrt(0.1 * matching) * np.exp(1j * radiant)
