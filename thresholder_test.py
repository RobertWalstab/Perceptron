""" Module tests and visualises function of thresholder. """


__author__ = 'Olaf Groescho'


import numpy as np

import genplotting
import mrr_thresholder_final


def plot_run():
    ''' Plots thresholder output power over input power. '''
    t = mrr_thresholder_final.MrrThreshold()
    a = np.arange(-100, 200, 1)
    a1 = genplotting.plot_function(a, t.run)
    return a1
