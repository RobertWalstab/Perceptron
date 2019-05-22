from math import atan
from scipy import constants
import numpy as np
import ownconstants as ocst


def var_pi():
    return (ocst .mu0/ocst.epsilon0)**0.25 * (2*ocst.n0)**0.5


def ang_freq(f):
    ''' Angular frequency '''
    return 2 * np.pi * f


def ang_freq_r(f):
    ''' Angular frequency of r '''
    c = 300
    return ang_freq(c) / 1.55


def sigma(f):
    return ocst.sigma_r * (ang_freq_r(f) / ang_freq(f)) ** 2


def mu(f):
    return 2 * ang_freq(f) / ocst.c * ocst.sigma_n / ocst.sigma_r


def fce_r(f, tau):
    ''' Free carrier effect (Index r)
    Beta equals zero '''
    return (sigma(f) * ocst.tau * ocst.beta
            / (2 * constants.Planck * ang_freq(f)))


def fce_i(f, tau):
    ''' Free carrier effect (Index i)
    Beta equals zero '''
    return fce_r(f, tau) * (0.5 * mu(f))


def intensity(i0, z, f, tau):
    constant1 = 1 - np.exp(-2 * ocst.alpha * z)
    return ((i0 * np.exp(-ocst.alpha * z))
            / (np.sqrt(1 + i0**2 * (fce_r(f, tau) / ocst.alpha) * constant1)))


def leff(i0, z, f, tau):
    ''' Effective length '''
    a = i0 * np.sqrt(fce_r(f, tau) / ocst.alpha)
    print(a)
    b = intensity(i0, z, f, tau) * np.sqrt(fce_r(f, tau) / ocst.alpha)
    print(b)
    return (atan(a) - atan(b)) / (i0 * np.sqrt(fce_r(f, tau) * ocst.alpha))


def gamma(f):
    return (ang_freq(f) / ocst.c) * ocst.n2


def beta0(f):
    ''' propagation constant '''
    return ocst.n0 * k(f)


def delta_psi(psi0, f):
    ''' Phase shift aquired during one round trip within the ring. '''
    return 2 * np.pi * ocst.radius * beta0(f) - psi0


def psi(i0, z, f, tau, psi0):
    return (psi0
            + gamma(f) * i0 * leff(i0, z, f, tau)
            - mu(f) / 2  # Beta equals zero (fce_i(f, tau) / fce_r(f, tau))
            * (np.log(i0 / intensity(i0, z, f, tau) - ocst.alpha * z)))


def i4(i0, f, tau, psi0):
    z = 2 * np.pi * ocst.radius
    ci = intensity(i0, z, f, tau)
    return ((ocst.r ** 2 * i0 + ci - 2 * ocst.r * np.sqrt(i0 * ci)
            * np.cos(delta_psi(f, psi0)))
            / (1 - ocst.r ** 2))


def i1(i0, f, tau):
    z = 2 * np.pi * ocst.radius
    ci = intensity(i0, z, f, tau)
    return i0 - ci + i4(i0, z, f, tau)


def i_in(i0, f, tau):
    c2 = 1 - np.exp(2 * ocst.alpha * ocst.l1)
    return ((i1(i0, f, tau) * np.exp(ocst.alpha * ocst.l1))
            / (np.sqrt(1 + i1(i0, f, tau) ** 2
                       * fce_r(f, tau)
                       / ocst.alpha) * c2))


def i_tr(i0, f, tau, psi0):
    c2 = 1 - np.exp(- 2 * ocst.alpha * ocst.l2)
    divisor = (np.sqrt(1 + i4(i0, f, tau, psi0) ** 2
                       * fce_r(f, tau) / ocst.alpha) * c2)
    return ((i4(i0, f, tau, psi0) * np.exp(- ocst.alpha * ocst.l2))
            / divisor)


def e4(e1, e3):
    return ocst.r * e1 + 1j * ocst.t * e3


def e2(e1, e3):
    return 1j * ocst.t * e1 + ocst.r * e3


def k(f):
    return ang_freq(f) / ocst.c


# def coupler():
    
#     return None
