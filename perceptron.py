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

    def train(self, training_inputs, label):
        ''' Trains stored weight by computing first predicted output and
        combining it with training data.

        Paramerters:
        training_inputs : List, length supposed to match perceptrons
        number of inputs: Integer value.
        label : Value 0 or 1, expected output.
        '''
        if isinstance(training_inputs, list):  # and isinstance(labels, list):
            minimum = self.no_of_inputs
            if len(training_inputs) >= minimum:  # and len(labels) >= minimum:
                print("Adapt weights in synapses")
                input = training_inputs[:minimum]
                s = self._sum_outputs(training_inputs)
                predicted = self._gauge(s)  # thresholder.run(s)
                _ = [self.synapses[n].adjust(input[n],
                                             expected=label,
                                             predicted=predicted)
                     for n in range(minimum)]
            else:
                print("Less inputs than synapses.")
                raise ValueError
        else:
            print('List as input type required.')
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
        resulting = self._receive(signal_in, expected, predicted,
                                  do_train=False)
        signal = resulting['transformee']
        o = self.amp.amplify(signal)
        return o

    def adjust(self, signal_in, expected, predicted):
        ''' Training mode, adapts the stored phase (weight).

        Paramerters:
        signal_in, expected, predicted : All floats. '''
        input_signals = self._receive(signal_in, expected, predicted,
                                      do_train=True)
        _ = input_signals.pop('transformee', None)
        print(input_signals)
        update_val = self._process_logic(**input_signals)
        self.amp.alter_value(update_val)
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
        print('ccontrol='+str(control))
        signals = {'control': control,
                   'transformee': transformee,
                   'expected': expected,
                   'predicted': predicted,
                   'do_train': do_train}
        return signals  # control, transformee, expected, predicted, do_train

    def _process_logic(self, control, expected, predicted, do_train):
        ''' Implements the control of training/working mode.

        Paramerters:
        control: Amplitude, float.
        expected: Amplitude, float.
        predicted: Amplitude, float.
        do_train: Amplitude, float.

        Output:
        u : Update value, float.
        '''

        print('control '+str(control))
        switch = abs_deco(cross_deco(self.gate.calculate_fredkin))
        refused, choosen, _ = switch(control, 0, expected)
        print('choosen, refused '+str(choosen)+' '+str(refused))
        thwart = shift_sig(refused)
        dispose, attendance, _ = switch(thwart, choosen, predicted)
        print('attendance, dispose '+str(attendance)+' '+str(dispose))
        _, update, _ = switch(attendance, 0, do_train)
        return update


def cross_deco(milburn):
    ''' Repairs inverted control behavior of fredkin gate. '''
    def inverted(in1, in2, c):
        return milburn(in2, in1, c)
    return inverted


def abs_deco(complex):
    ''' Takes the absolute value and the sign of the real part.
    Removes output phase of fredkin gate.'''
    def absolute(*args, **kwargs):
        c = complex(*args, **kwargs)
        a = np.array(list(map(abs, c)))
        sign = (lambda s: np.sign(s.real))
        b = np.array(list(map(sign, c)))
        return tuple(b * a)
    return absolute


class Coupler:
    """ Implements a static beamsplitter which mixes two input fields.
    Equations are given in "A coherent Perceptron for all-Optical Learning"
    equation (8)"""

    def __init__(self, name, mixing_angle=0.25*np.pi):
        self.name = name
        self.theta = mixing_angle

    def couple(self, in_1, in_2, mixing_angle=None, amplitude=True):
        ''' Mixes two input signals into two output signals.

        Paramerters :
        in_1, in_2 : Signals, float values.
        mixing_angle : Angle setting the reflected, transmitted ratio.
        amplitude : If True, coupler treats input as amplitude and will give
        amplitude values as output. (Reals -> Real)
        If False, coupler treats input as field and will give field values
        as output. (Real -> Complex)
        '''
        if mixing_angle is None:
            mixing_angle = self.theta
        if amplitude:
            out_1, out_2 = self._couple_amplitude(in_1, in_2, mixing_angle)
        else:
            out_1, out_2 = self._couple_field(in_1, in_2, mixing_angle)
        return out_1, out_2

    def delay(self, mixing_angle):
        ''' Returns the accumulated phase delay of both output signals.

        Paramerters:
        mixing_angle : initial phase shift of splitting angle control
        '''
        return self._phase_delay(mixing_angle), self._phase_delay(0)

    def _couple_field(self, in_1, in_2, theta):
        ''' Mixes two input signals into two output signals.
        Implements the coupler.

        Parameters:
        in_1, in_2 : Complex signals. (Field)
        theta : Angle in radiant.

        Output:
        out_1, out_2 : Complex output signals.
        '''

        out_1 = in_1 * np.cos(theta) + in_2 * 1j * np.sin(theta)
        out_2 = in_1 * 1j * np.sin(theta) + in_2 * np.cos(theta)
        return out_1, out_2

    def _couple_amplitude(self, in_1, in_2, theta):
        ''' Mixes two input signals into two output signals.
        Implements the coupler.

        Parameters:
        in_1, in_2 : Signal amplitudes.
        theta : Angle in radiant.

        Output: out_1, out_2 : Output amplitudes.
        '''

        out_1 = in_1 * np.cos(theta) - in_2 * np.sin(theta)
        out_2 = in_1 * np.sin(theta) + in_2 * np.cos(theta)
        return out_1, out_2

    def _phase_delay(initial=0):
        ''' Returns phase shift caused by coupler.
        Guessed operation principle. '''
        delay = 0.5 * np.pi + initial
        return delay


# def switch(in_1, in_2, in_c):
#     ''' Implements optical switch.
#         Causes chrossed input-output transmission.

#         Parameters:
#         in_1 : Signal1 amplitude, float.
#         in_2 : Signal2 amplitude, float.
#         in_c : Control signal amplitude, float.

#         Output:
#         out_1 : Output1 signal amplitude, float.
#         out_2 : Output1 signal amplitude, float.
#         '''

#     threshold_val = 0.1  # Example value
#     out_1 = out_2 = None
#     if in_c <= threshold_val:  # Example logic
#         out_1 = in_1
#         out_2 = in_2
#     else:
#         out_1 = in_2
#         out_2 = in_1
#     return out_1, out_2


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
            up_time = 0.20
            down_time = 0.20
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

    def __init__(self, name, const_bias=None,
                 variable_bias_amplitude=None,
                 *args, **kwargs):
        self.name = name
        if const_bias is None:
            self.const_bias = self._xi_0()
        else:
            self.const_bias = const_bias
        self.coupler_sig = Coupler(name='coupler_sig',
                                   mixing_angle=0.25*np.pi)
        self.coupler_bias = Coupler(name='coupler_bias',
                                    mixing_angle=1.75*np.pi)
        self.amp1 = Amplifier('amp1')
        self.amp2 = Amplifier('amp2')
        self.storage = OpticalStorage('opo_storage',
                                      amplitude=variable_bias_amplitude,
                                      *args, **kwargs)

    def _xi_0(self):
        ''' Sets amplitude of constant part of bias. '''
        matching = 1.993792
        return np.sqrt(0.9 * matching)

    def amplify(self, signal_in):
        ''' Equips signal with positive or negative weight.
        Signal is splitted, one part is shifted by ph for negated input.
        Bias coupler merges constant phase signal and shifted version of
        another signal. The phase shift is provided by phase storage.
        The constant phase signals amplitude is fixed as design parameter.
        The variable phase signals amplitude is fixed as well. '''
        nopo_bias = self.storage.out()
        bias_pos, bias_neg = self.coupler_bias.couple(nopo_bias,
                                                      self.const_bias,
                                                      mixing_angle=0.25*np.pi,
                                                      amplitude=False)
        bias_pos = abs(bias_pos)**2
        bias_neg = abs(bias_neg)**2
        pos_amplified = self.amp1.amplify(signal_in, bias_pos)
        signal_negated = shift_sig(signal_in)
        neg_amplified = self.amp2.amplify(signal_negated, bias_neg)
        amplified_total = pos_amplified + neg_amplified
        print('signal_in='+str(signal_in)+', \n'
              + 'nopo_angle='+str(np.angle(nopo_bias))+', \n'
              + 'nopo_ampl='+str(abs(nopo_bias))+', \n'
              + 'const_bias='+str(self.const_bias)+', \n'
              + 'bias_pos='+str(bias_pos)+', \n'
              + 'bias_neg='+str(bias_neg)+', \n'
              + 'amplified_total='+str(amplified_total))
        return amplified_total

    def alter_value(self, pulse):
        ''' Let the storage change its stored phase.'''
        print('update_pulse='+str(pulse))
        self.storage.alter_value(pulse)
        return None


class OpticalStorage:
    """ Implements NOPO for storing phases.
    Further information in section 2.2 of
    'A coherent Perceptron for all-Optical Learning' """
    def __init__(self, name, phase_stored=0, amplitude=None):
        self.name = name
        self.phase = phase_stored
        self.phase_min = -np.pi
        self.phase_max = np.pi
        self.step = 0.15
        self.threshold = 0.1
        if amplitude is None:
            self.amplitude = self._xi_var()
        else:
            self.amplitude = amplitude

    def _xi_var(self):
        ''' Sets amplitude of variable part of bias.  '''
        matching = 1 / (10 * 1.795)
        return np.sqrt(0.1 * matching)

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
        ''' Returns signal with fixed amplitude and stored phase. '''
        return self.amplitude * np.exp(1j*self.phase)

    def _set_phase(self, phase):
        ''' Sets the phase to a certain value for testing purposes.
        This method is not physically backed. '''
        self.phase = phase
        return None
