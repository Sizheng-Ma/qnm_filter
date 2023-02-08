__all__ = ['Network']

import numpy as np
from .gw_data import *
import h5py
import lal
import scipy.linalg as sl
import warnings

class Network(object):

    def __init__(self, **kws):
        self.oringal_data = {}
        self.filtered_data = {}
        self.acfs = {}
        self.start_times = {}
        self.cholesky_L = {}
        self.inverse_cholesky_L = {}

        self.ra = kws.get('ra', None)
        self.dec = kws.get('dec', None)
        self.t_init = kws.get('t_init', None)
        self.window_width = kws.get('window_width', None)

    def import_ligo_data(self, filename):

        with h5py.File(filename, 'r') as f:
            h = f['strain/Strain'][:]
            t_start = f['meta/GPSstart'][()]
            duration = f['meta/Duration'][()]
            ifo = str(f['meta/Detector'][()], 'utf-8')

            time = np.linspace(t_start, t_start+duration,\
                               num=len(h), endpoint=False)

            self.oringal_data[ifo] = Data(h, index=time, ifo=ifo)
            # return self.oringal_data #TODO: remove this return
    
    def import_data_array(self, attr_name, data, time, ifo):
        """TODO add comments"""
        getattr(self, attr_name)[ifo] = Data(data, index=time, ifo=ifo)

    def detector_alignment(self, **kwargs):
        t_init = kwargs.pop('t_init', None)
        if t_init==None:
            raise ValueError("t_init is not provided")
        tgps = lal.LIGOTimeGPS(t_init)

        for ifo, data in self.oringal_data.items():
            if self.ra==None or self.dec==None:
                self.start_times[ifo] = t_init
            else:
                location = lal.cached_detector_by_prefix[ifo].location
                dt_ifo = lal.TimeDelayFromEarthCenter(location,\
                                                self.ra, self.dec, tgps)
                self.start_times[ifo] = t_init + dt_ifo
            if self.start_times[ifo] < data.time[0] or\
                self.start_times[ifo] > data.time[-1]:
                raise ValueError("{} start time not in data".format(ifo))

    @property
    def start_indices(self) -> dict:
        i0_dict = {}
        for ifo, data in self.oringal_data.items():
            t0 = self.start_times[ifo]
            i0_dict[ifo] = abs(data.time - t0).argmin()
        return i0_dict

    @property
    def n_analyze(self):
        n_dict = {}
        for ifo, data in self.oringal_data.items():
            n_dict[ifo] = int(round(self.window_width/data.delta_t))
        if len(set(n_dict.values())) > 1:
            raise ValueError("Detectors have different sampling rates")

        return list(n_dict.values())[0]


    def truncate_data(self, network_data) -> dict:
        data = {}
        i0s = self.start_indices
        for i, d in network_data.items():
            data[i] = Data(d.iloc[i0s[i]:i0s[i] + self.n_analyze])
        return data

    def condition_data(self, attr_name, **kwargs):
        unconditioned_data = getattr(self, attr_name)
        for ifo, data in unconditioned_data.items():
            t0 = self.start_times[ifo]
            getattr(self, attr_name)[ifo] = data.condition(t0=t0, **kwargs)

    def compute_acfs(self, attr_name, **kws):
        noisy_data = getattr(self, attr_name)
        if self.acfs:
            warnings.warn("Overwriting ACFs")
        for ifo, data in noisy_data.items():
            self.acfs[ifo] = data.get_acf(**kws)

    def cholesky_decomposition(self):
        """Compute the Cholesky-decomposition of the covariance matrix.
        """
        for ifo, acf in self.acfs.items():
            truncated_acf = acf.iloc[:self.n_analyze].values
            L = np.linalg.cholesky(sl.toeplitz(truncated_acf))
            L_inv = np.linalg.inv(L)
            norm = np.sqrt(np.sum(abs(np.dot(L_inv,L)
                                  -np.identity(len(L)))**2))
            if abs(norm)>1e-8:
                raise ValueError("Inverse of L is not correct")

            self.cholesky_L[ifo] = L
            self.inverse_cholesky_L[ifo] = L_inv

    def compute_likelihood(self, apply_filter = True):
        likelihood = 0

        if not apply_filter:
            truncation = self.truncate_data(self.oringal_data)
        else:
            truncation = self.truncate_data(self.filtered_data)

        for ifo, data in truncation.items():
            wd = np.dot(self.inverse_cholesky_L[ifo], data)
            likelihood -= 0.5*np.dot(wd, wd)
        return likelihood

    def add_filter(self,  **kwargs):
        for ifo, data in self.oringal_data.items():
            data_in_freq = data.fft_data
            freq = data.fft_freq
            filter_in_freq = Filter(**kwargs).total_filter(freq)
            ifft = np.fft.irfft(filter_in_freq*data_in_freq,\
                                                norm='ortho', n=len(data))
            self.filtered_data[ifo] = Data(ifft, index=data.index, ifo=ifo)

    def likelihood_vs_mass_spin(self, M_est, chi_est, **kwargs) -> float:
        """Compute likelihood for the given mass and spin
        
        Parameters
        ----------
        M_est : float
            in solar mass, mass of rational filters
        chi_est : float
            dimensionless spin of rational filters

        Returns
        -------
        The corresponding likelihood.
        """
        model_list = kwargs.pop('model_list')
        self.add_filter(mass=M_est, chi=chi_est, model_list=model_list)
        return self.compute_likelihood(apply_filter=True)

    def compute_SNR(self, data, template, ifo, optimal) -> float:
        template_w = np.dot(self.inverse_cholesky_L[ifo], template)
        data_w = np.dot(self.inverse_cholesky_L[ifo], data)
        snr_opt = np.sqrt(np.dot(template_w, template_w))
        if optimal:
            return snr_opt
        else:
            snr = np.dot(data_w, template_w)/snr_opt
            return snr