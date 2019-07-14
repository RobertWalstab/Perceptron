""" Perceptron module

This module models the behavior of a physical perceptron.
"""


__author__ = 'Olaf Groescho'


import numpy as np

import mrr_thresholder_final
import fredkin


class Perceptron:
    """ Whole perceptron, including a set of synapses and addition and
    threshold of synapse output. """

    def __init__(self, name, no_of_inputs, *args, **kwargs):
        self.no_of_inputs = no_of_inputs
        self.synapses = list([Synapse(name='synapse_'+str(n), *args, **kwargs)
                              for n in range(no_of_inputs)])
        self.thresholder = mrr_thresholder_final.MrrThreshold()
        self.name = name

    def predict(self, inputs):
        ''' Computes the predicted classification

        Paramerters:
        inputs : List of float input values. Length supposed to match
        perceptrons number of inputs.

        Output:
        l : Classification label, int, 0 or 1. '''
        if isinstance(inputs, list):
            if len(inputs) >= self.no_of_inputs:
                s = self._sum_outputs(inputs)
                return self._gauge(s)  # self.thresholder.run(s)
            else:
                print("Less inputs than synapses.")
                raise ValueError
        else:
            print('List as input type required.')
            raise TypeError
        return None

    def train(self, training_inputs, labels):
        ''' Trains stored weight by computing first predicted output and
        combining it with training data.

        Paramerters:
        training_inputs : List, length supposed to match perceptrons
        number of inputs.
        labels : List, length supposed to match perceptrons
        number of inputs.
        '''
        if isinstance(training_inputs, list) and isinstance(labels, list):
            minimum = self.no_of_inputs
            if len(training_inputs) >= minimum and len(labels) >= minimum:
                print("Adapt weights in synapses")
                honey = list(zip(training_inputs[:minimum], labels[:minimum]))
                s = self._sum_outputs(training_inputs)
                predicted = self._gauge(s)  # thresholder.run(s)
                _ = [self.synapses[n].adjust(predicted, *honey[n])
                     for n in range(minimum)]
            else:
                print("Less inputs than synapses.")
                raise ValueError
        else:
            print('List as input types required.')
            raise TypeError
        return None

    def _sum_outputs(self, inputs):
        ''' Implements the summation block. '''
        outputs = [self.synapses[n].out(inputs[n])
                   for n in range(self.no_of_inputs)]
        return np.sum(outputs)

    def _gauge(self, bushel):
        ''' Models the thresholder. '''
        if bushel > 0:
            activation = 1
        else:
            activation = 0
        return activation


class Synapse:
    """ Implements synapse circuit as drawn in
    'A coherent Perceptron for all-Optical Learning' section 2.3 (page 9). """

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.amp = ProgrammableAmplifier('prog_gain_amp', *args, **kwargs)
        self.coupler_in = Coupler(name='c_in')
        self.coupler_out = Coupler(name='c_out')
        self.gate = fredkin.Fredkin()

    def out(self, signal_in, expected=0, predicted=0):
        ''' Working mode, returns the weighted output.'''
        _, transformee, _, _, _ = self._receive(signal_in, expected, predicted,
                                                do_train=False)
        o = self.amp.amplify(transformee)
        return o

    def adjust(self, signal_in, expected, predicted):
        ''' Training mode, adapts the stored phase (weight).

        Paramerters:
        signal_in, expected, predicted : All floats. '''
        input_signals = self._receive(signal_in, expected, predicted,
                                      do_train=True)
        update_val = self._process_logic(input_signals)
        self.amp.adjust(update_val)
        return None

    def _receive(self, signal_in, expected, predicted, do_train):
        ''' Implements the signal input.

        Paramerters:
        signal_in: Amplitude, float.
        predicted: Amplitude, float.
        expected: Amplitude, float.
        do_train: Amplitude, float.

        Output:
        control: Amplitude, float.
        transformee: Amplitude, float.
        expected, predicted, do_train: Forwarded, typed as in input.
        '''
        control, transformee = self.coupler_in.couple(signal_in, 0)
        return control, transformee, expected, predicted, do_train

    def _process_logic(self, control, expected, predicted, do_train):
        ''' Implements the control of training/working mode.

        Paramerters:
        signal_in: Amplitude, float.
        predicted: Amplitude, float.
        expected: Amplitude, float.
        do_train: Amplitude, float.

        Output:
        u : Update value, float.
        '''

        switch = self.fredkin.calculate_fredkin
        refused, choosen = switch(control, 0, expected)
        print('choosen, refused '+str(choosen)+' '+str(refused))
        thwart = shift_sig(refused)
        dispose, attendance = switch(thwart, choosen, predicted)
        print('attendance, dispose '+str(attendance)+' '+str(dispose))
        _, update = switch(attendance, 0, do_train)
        return update


class Coupler:
    """ Implements a static beamsplitter which mixes two input fields.
    Equations are given in "A coherent Perceptron for all-Optical Learning"
    equation (8)"""

    def __init__(self, name, mixing_angle=0.5*np.pi):
        self.name = name
        self.theta = mixing_angle

    def couple(self, in_1, in_2, mixing_angle=None):
        ''' Mixes two input signals into two output signals.

        Paramerters :
        in_1, in_2 : Signal amplitudes, float values.
        phase : Phase difference between in_1 and in_2 in radiant.
        '''
        if mixing_angle is None:
            mixing_angle = self.theta
        out_1, out_2 = self._couple(in_1, in_2, mixing_angle)
        return out_1, out_2

    def delay(self, mixing_angle):
        ''' Returns the accumulated phase delay of both output signals.

        Paramerters:
        mixing_angle : initial phase shift of splitting angle control '''
        return self._phase_delay(mixing_angle), self._phase_delay(0)

    def _couple(self, in_1, in_2, theta):
        ''' Mixes two input signals into two output signals.
        Implements the coupler.

        Parameters:
        in_1, in_2 : Signal amplitudes, float values.
        theta : Angle, fraction or multiple of pi.

        Output: out_1, out_2 : Output signal amplitudes.
        '''

        out_1 = in_1 * np.cos(theta) - in_2 * np.sin(theta)
        out_2 = in_1 * np.sin(theta) + in_2 * np.cos(theta)
        return out_1, out_2

    def _phase_delay(initial=0):
        ''' Returns phase shift caused by coupler.
        Guessed operation principle. '''
        delay = 0.5 * np.pi + initial
        return delay


class BiasRouter:
    """ Models function of bias coupler. """

    def __init__(self, name, mixing_angle=0.5*np.pi):
        self.name = name
        self.theta = mixing_angle

    def couple(self, in_1, in_2, mixing_angle=None):
        ''' Mixes two input signals into two output signals.

        Paramerters :
        in_1, in_2 : Signal amplitudes, float values.
        phase : Phase difference between in_1 and in_2 in radiant.
        '''
        if mixing_angle is None:
            mixing_angle = self.theta
        out_1, out_2 = self._couple(in_1, in_2, mixing_angle)
        return out_1, out_2

    def delay(self, mixing_angle):
        ''' Not implementated.

        Paramerters:
        mixing_angle : initial phase shift of splitting angle control '''
        return None

    def _couple(self, in_1, in_2, theta):
        ''' Mixes two input signals into two output signals.
        Implements the coupler.

        Parameters:
        in_1, in_2 : Signal amplitudes, float values.
        theta : Angle, fraction or multiple of pi.

        Output: out_1, out_2 : Output signal amplitudes.
        '''

        out_1 = np.sqrt(in_1**2 + in_2**2 + 2 * in_1 * in_2 * np.cos(theta))
        out_2 = np.sqrt(in_1**2 + in_2**2 - 2 * in_1 * in_2 * np.cos(theta))
        return out_1, out_2


def switch(in_1, in_2, in_c):
    ''' Implements optical switch.
        Causes chrossed input-output transmission.

        Parameters:
        in_1 : Signal1 amplitude, float.
        in_2 : Signal2 amplitude, float.
        in_c : Control signal amplitude, float.

        Output:
        out_1 : Output1 signal amplitude, float.
        out_2 : Output1 signal amplitude, float.
        '''

    threshold_val = 0.1  # Example value
    out_1 = out_2 = None
    if in_c <= threshold_val:  # Example logic
        out_1 = in_1
        out_2 = in_2
    else:
        out_1 = in_2
        out_2 = in_1
    return out_1, out_2


def shift_sig(x):
    ''' Shifts signal 180 degree -> Negative amplitude'''
    return -x


class Amplifier:
    """ Implements amplifier with tuneable gain
        Two identical ring resonators placed in two arms of inferometer.
        Coupling coefficient k, Kerr coefficient chi """

    def __init__(self, name):
        self.name = name

    def amplify(self, signal_in, bias=1):
        ''' Amplifies signal.

        Parameters:
        signal_in : Signal amplitude

        Output:
        a : Amplified signal amplitude
        '''
        amplified = self._amplify(signal_in, bias)
        return amplified

    def _amplify(self, x, b):
        ''' Applies a gain function to input signal amplitude.
        The gain depends on a signal bias.

        Parameters:
        x : Signal amplitude
        b : Bias amplitude, relative to epsilon_max

        Internal Parameters:
        gain : Function object, (signal b) -> (float g)

        Output:
        a : Amplified signal amplitude
        '''
        def gain(bias):
            ''' Example behavior: triangle function'''
            g = None
            peak_val = 20
            peak_pos = 1
            up_time = 0.1
            down_time = 0.15
            if bias < (peak_pos - up_time):
                g = 0
            elif bias < peak_pos:
                g = (peak_val / up_time * (bias - (peak_pos - up_time)))
            elif bias < (peak_pos + down_time):
                g = (peak_val / down_time) * (-bias + peak_pos + down_time)
            else:
                g = 0
            # print('g = '+str(g))
            return g

        amplified = gain(b) * x
        return amplified

    def _phase_delay(self,):
        ''' Guessed behavior. '''
        initial_phase_splitter_shift = 1.75
        coupling_shift = 0.5
        resonator_shift = 2  # At resonance
        guide_shift = 0  # Approximated
        combiner_shift = 1-initial_phase_splitter_shift

        delay = np.pi * (initial_phase_splitter_shift
                         + coupling_shift
                         + resonator_shift
                         + guide_shift
                         + combiner_shift
                         + coupling_shift)
        return delay

    def alter_value(self, bias):
        print('Plain amplifier has no storage. ')
        print('Value vanishes. ')
        return None


class ProgrammableAmplifier:
    """ Implements the programmable gain amplifier described in
    'A coherent Perceptron for all-Optical Learning', section 2.2 (page 8).
    Contains two amplifiers, an OPO for storing the phase and constant bias
    signal for controlling amplifiers. """

    def __init__(self, name, const_bias=1.0,  # *np.sqrt(2),
                 variable_bias_amplitude=0.1,  # *np.sqrt(2),
                 *args, **kwargs):
        self.name = name
        self.const_bias = const_bias
        self.var_bias = variable_bias_amplitude
        self.coupler_sig = Coupler(name='coupler_sig',
                                   mixing_angle=0.5*np.pi)
        self.coupler_bias = BiasRouter(name='coupler_bias',
                                       mixing_angle=1.75*np.pi)
        self.amp1 = Amplifier('amp1')
        self.amp2 = Amplifier('amp2')
        self.storage = OpticalStorage('opo_storage', *args, **kwargs)

    def amplify(self, signal_in):
        ''' Equips signal with positive or negative weight.
        Signal is splitted, one part is shifted by ph for negated input.
        Bias coupler merges constant phase signal and shifted version of
        another signal. The phase shift is provided by phase storage.
        The constant phase signals amplitude is fixed as design parameter.
        The variable phase signals amplitude is fixed as well. '''
        phase = self.storage.out()
        bias_pos, bias_neg = self.coupler_bias.couple(self.var_bias,
                                                      self.const_bias,
                                                      mixing_angle=phase)
        pos_amplified = self.amp1.amplify(signal_in, bias_pos)
        signal_negated = shift_sig(signal_in)
        neg_amplified = self.amp2.amplify(signal_negated, bias_neg)
        amplified_total = pos_amplified + neg_amplified
        print('signal_in='+str(signal_in)+'\n'
              + ', mixing_angle='+str(phase/np.pi*180)+'\n'
              + ', const_bias='+str(self.const_bias)
              + ', var_bias='+str(self.var_bias)+'\n'
              + ', bias_pos='+str(bias_pos)
              + ', bias_neg='+str(bias_neg)+'\n'
              + ', amplified_total='+str(amplified_total))
        return amplified_total

    def alter_value(self, pulse):
        ''' Let the storage change its stored phase.'''
        self.storage.alter_value(pulse)
        return None


class OpticalStorage:
    """ Implements NOPO for storing phases.
    Further information in section 2.2 of
    'A coherent Perceptron for all-Optical Learning' """
    def __init__(self, name, phase_stored=0):
        self.name = name
        self.phase = phase_stored
        self.phase_min = 0
        self.phase_max = 2*np.pi
        self.step = 0.15
        self.threshold = 0.1

    def alter_value(self, pulse):
        ''' Alters the stored phase value in a restricted manner. '''
        if pulse < - self.threshold:
            if self.phase > self.phase_min:
                self.phase -= self.step
        if pulse > self.threshold:
            if self.phase < self.phase_max:
                self.phase += self.step
        return None

    def out(self):
        ''' Returns stored value. '''
        return self.phase

    def _set_phase(self, phase):
        ''' Sets the phase to a certain value for testing purposes.
        This method is not physically backed. '''
        self.phase = phase
        return None
