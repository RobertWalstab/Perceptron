""" Module for modeling the phase shift of a Kerr amplifier. """

__author__ = 'Olaf Groescho'

import numpy as np

import genplotting


def phase_delay(r, a, detuning):
    ''' Computes effective phase shift according
    Bogaerts2012: "Silicon miroring resonators" Eq (4).

    Paramerters:
    r : Reflective coefficient, commonly  r in [0, 1) .
    a : Amplitude transmission, loss of traveling one round.
    detuning : Phase shift away from zero at coupling point.'''
    incident_phase = np.arctan((r * np.sin(detuning))
                               / (a - r * np.cos(detuning)))
    attenuated_phase = np.arctan((a * r * np.sin(detuning))
                                 / (1 - a * r * np.cos(detuning)))
    delay = np.pi + detuning + incident_phase + attenuated_phase
    return delay


def plot_example():
    ''' Plot phase delay for one operation point '''
    
    x = np.arange(-np.pi, np.pi, 0.01)
    x_info = 'Phase detuning'
    reflection = 0.8
    amp_transmission = 0.95
    delay = (lambda dt: phase_delay(reflection, amp_transmission, dt))
    y_info = 'Phase delay'
    explain = ('r='+str(reflection)+', a='+str(amp_transmission))
    ax = genplotting.plot_function(x, delay, label=explain)
    ax.set_xlabel(x_info)
    ax.set_ylabel(y_info)
    return ax
