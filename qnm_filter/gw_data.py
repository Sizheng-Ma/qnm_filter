"""Utilities to manipulate GW data and rational filters.
"""
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
    """Container for rational filters.

    Attributes
    ----------
    chi : float
        remnant dimensionless spin.
    mass : float
        remnant mass, in solar mass.
    model_list : tuple
        quasinormal modes to be filtered.
    """

    def __init__(self, chi=None, mass=None, model_list=None):
        self.chi = chi
        self.mass = mass # in solar mass
        self.model_list = construct_mode_list(model_list)

    @property
    def get_spin(self) -> float:
        """Return :attr:`Filter.chi`."""
        return self.chi   

    @property
    def get_mass(self) -> float:
        """Return :attr:`Filter.mass`."""
        return self.mass   
    
    @property
    def get_model_list(self):
        """Return :attr:`Filter.model_list`."""
        return self.model_list  

    @staticmethod
    def mass_unit(mass):
        """Convert mass unit from solar mass to second."""
        return mass * T_MSUN


    def single_filter(self, normalized_freq, l, m, n):
        """Compute rational filters. 

        Parameters
        ---------- 
        normalized_freq : array
            in remnant mass, frequencies that rational filters are evaluated at.
        """
        omega = qnm.modes_cache(s=-2, l=l, m=m, n=n)(a=self.chi)[0]
        return (normalized_freq-omega)/(normalized_freq-np.conj(omega))\
                *(normalized_freq+np.conj(omega))/(normalized_freq+omega)
    
    def total_filter(self, freq):
        """The total rational filter that removes the modes stored in :attr:`Filter.model_list`.

        Parameters
        ---------- 
        freq : array
            in Hz, frequencies that the total filter is evaluated at.
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


class Data(pd.Series):
    def __init__(self, *args, ifo=None, info=None,  **kwargs):
        if ifo is not None:
            ifo = ifo.upper()
        kwargs['name'] = kwargs.get('name', ifo)
        super(Data, self).__init__(*args, **kwargs)
        self.ifo = ifo
        self.info = info or {}

    @property
    def time(self):
        """Time stamps."""
        return self.index.values

    @property
    def delta_t(self) -> float:
        """Sampling time interval."""
        return self.index[1] - self.index[0]

    @property
    def fsamp(self) -> float:
        """Sampling frequency (`1/delta_t`)."""
        return 1/self.delta_t

    @property
    def fft_freq(self):
        return np.fft.rfftfreq(len(self), d=self.delta_t) * 2 * np.pi
    
    @property
    def fft_data(self):
        return np.fft.rfft(self.values, norm='ortho')

    def condition(self, t0=None, srate=None, flow=None, fhigh=None, trim=0.25,
                  remove_mean=True, **kwargs):
        srate = kwargs.pop('srate', srate)
        flow = kwargs.pop('flow', flow)
        raw_data = self.values
        raw_time = self.index.values

        ds = int(round(self.fsamp/srate))

        if t0 is not None:
            ds = int(ds or 1)
            i = np.argmin(abs(raw_time - t0))
            raw_time = np.roll(raw_time, -(i % ds))
            raw_data = np.roll(raw_data, -(i % ds))

        fny = 0.5/self.delta_t
        # Filter
        if flow and not fhigh:
            b, a = ss.butter(4, flow/fny, btype='highpass', output='ba')
        elif fhigh and not flow:
            b, a = ss.butter(4, fhigh/fny, btype='lowpass', output='ba')
        elif flow and fhigh:
            b, a = ss.butter(4, (flow/fny, fhigh/fny), btype='bandpass',
                              output='ba')

        if flow or fhigh:
            cond_data = ss.filtfilt(b, a, raw_data)
        else:
            cond_data = raw_data

        if ds and ds > 1:
            cond_data = ss.decimate(cond_data, ds, zero_phase=True)
            cond_time = raw_time[::ds]

        N = len(cond_data)
        istart = int(round(trim*N))
        iend = int(round((1-trim)*N))

        cond_time = cond_time[istart:iend]
        cond_data = cond_data[istart:iend]

        if remove_mean:
            cond_data -= np.mean(cond_data)

        return Data(cond_data, index=cond_time, ifo=self.ifo)


    def get_acf(self, **kws):
        dt = self.delta_t
        fs = 1/dt

        freq, psd = ss.welch(self.values, fs=fs, nperseg=fs)
        rho = 0.5*np.fft.irfft(psd) / self.delta_t
        return Data(rho, index=np.arange(len(rho))*dt)

