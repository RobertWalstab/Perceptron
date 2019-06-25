import numpy as np


def amplify(x, b):
    ''' Implements amplifier with tuneable gain
    Prototype

    Two identical ring resonators placed in two arms of inferometer.
    Coupling coefficient k, Kerr coefficient chi

    Parameters:
    x : Signal amplitude
    b : Bias amplitude

    Internal Parameters:
    gain : Function object, (signal b) -> (float g)

    Output:
    a : Amplified signal amplitude
    '''

    def gain(bias):
        ''' Example behavior '''
        return np.exp(bias) - 2 * np.exp(bias - 1)

    amplified = gain(b) * x
    return amplified


def store_signal(x):
    ''' Implements programmable storage.
    Prototype.

    Non-degenerate parametric oscillator.
    Controls minus signal to the bias.

    Parameters:
    x : Signal amplitude

    Internal Parameters:
    k : Float, coupling factor

    Output:
    s : Stored signal
    '''
    k = 0.5  # Example coupling factor
    s = k * (1 - k) * x  # Example resonators behavior
    return s


def constant_signal():
    ''' Returns constant value, here 1 '''
    return 1


def couple(s1, s2, k):
    ''' Mixex two input signals into two output signals.

    Parameters:
    s1, s2 : Signal amplitudes, float values.

    Output: r1, r2 : Output signal amplitudes.
    '''
    
    r1 = s1 * (1 - k) + s2 * k
    r2 = s1 * k + s2 * (1 - k)
    return r1, r2


def compare(s1, s2, c):
    ''' Implements optical switch.

    Fredkin gate and thresholder

    Parameters:
    s1 : Signal1 amplitude, float.
    s2 : Signal2 amplitude, float.
    c : Control signal amplitude, float.

    Output:
    o : Output signal amplitude, float.
    '''

    th_val = 0.1  # Example value
    o = None
    if c >= th_val:  # Example logic
        o = s1
    else:
        o = s2
    return o


def shift_sig(x):
    ''' Shifts signal 90 degree -> Negative amplitude'''
    return -x


def input_logic(x, y, y_hat, t):
    ''' Implements the comtrol of training/working mode.

    Parameters:
    x : Float, signal in
    y : Float.
    y_hat : Float.
    t : Boolean flag, indicates the training mode

    Output:
    f : Feedback amplitude, float.
    '''
    
    mixing_guide = 0.1  # Placeholder for initial value
    feedback = 0.1  # Placeholder for feedback value
    mixing_guide = (compare(x, mixing_guide, y)
                    + compare(shift_sig(y), mixing_guide, y_hat))
    feedback = compare(mixing_guide, feedback, t)
    return feedback


#  Constants
signal_in = 1
decision = 1
expected = 0
train_mode = True

# Signal flow
control = input_logic(signal_in, decision, expected, train_mode)
bias = constant_signal() - store_signal(control)
bias1, bias2 = couple(constant_signal(), store_signal(control), k=0.2)
signal_out = amplify(signal_in, bias1) + amplify(shift_sig(signal_in), bias2)

print("Output signal = "+str(signal_out))
