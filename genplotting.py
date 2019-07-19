import numpy as np
import matplotlib.pyplot as plt


def single_value_fkt(par, *args, **kwargs):
    ''' Example function. '''
    return 1


def append_plot(new_par, new_val, ax=None):
    ''' Append if available, otherwise make new plot. '''
    par_array = None
    val_array = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        par_array = np.array([])
        val_array = np.array([])
    else:
        line = ax.lines[0]
        par_array = line.get_xdata()
        val_array = line.get_ydata()
        ax.lines.remove(line)
    par_array = np.append(par_array, new_par)
    val_array = np.append(val_array, new_val)
    line = ax.plot(par_array, val_array, color='blue', marker='.', lw=2)
    plt.show()
    return ax


def val_splitter(val_fkt, *args, **kwargs):
    ''' Applies first argument to function and 
    returns argument and saturated function. '''
    
    def ret_fkt(val):
        return val, val_fkt(val, *args, **kwargs)
    return ret_fkt


def plot_function(x_vals, x_handle, ax=None, *args, **kwargs):
    ''' Plots function over input values.
    Paramerters:
    x_vals : list - like
    x_handle : Function object, only x parameter missing. 

    Output:
    ax : Matplotlib axis object. '''
    
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    ax.plot(x_vals, list(map(x_handle, x_vals)),
            *args, **kwargs)
    ax.legend()
    return ax
