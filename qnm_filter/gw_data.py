__all__ = ['Data', 'Filter']

import qnm
import pandas as pd
import numpy as np
import lal
from collections import namedtuple
import scipy.signal as ss

T_MSUN = lal.MSUN_SI * lal.G_SI / lal.C_SI**3

ModeIndex = namedtuple('ModeIndex', ['l', 'm', 'n'])

def construct_mode_list(modes):
    if modes is None:
        modes = []
    elif isinstance(modes, str):
        from ast import literal_eval
        modes = literal_eval(modes)
    mode_list = []
    for (l, m, n) in modes:
        mode_list.append(ModeIndex(l, m, n))
    return mode_list

class Filter:
    def __init__(self, chi=None, mass=None, model_list=None):
        self.chi = chi
        self.mass = mass # in solar mass
        self.model_list = construct_mode_list(model_list)

    @property
    def get_spin(self) -> float:
        return self.chi   

    @property
    def get_mass(self) -> float:
        return self.mass   
    
    @property
    def get_model_list(self):
        return self.model_list  

    @staticmethod
    def mass_unit(mass):
        return mass * T_MSUN


    def single_filter(self, normalized_freq, l, m, n):
        """Compute the rational filter. normalized_freq is in the
        unit of the remnant mass
        """
        omega = qnm.modes_cache(s=-2, l=l, m=m, n=n)(a=self.chi)[0]
        return (normalized_freq-omega)/(normalized_freq-np.conj(omega))\
                *(normalized_freq+np.conj(omega))/(normalized_freq+omega)
    
    def total_filter(self, freq):
        """freq is in Hz.
        """
        final_rational_filter = 1
        if not bool(self.model_list):
            return final_rational_filter
        else:
            if (self.mass is None) or (self.chi is None):
                raise ValueError(f"Mass = {self.mass}"
                                 f" and Spin = {self.chi} are needed")
        normalized_freq = freq * self.mass * T_MSUN
        for mode in self.model_list:
            final_rational_filter *= self.single_filter(-normalized_freq,\
                                     mode.l, mode.m, mode.n)
        return final_rational_filter