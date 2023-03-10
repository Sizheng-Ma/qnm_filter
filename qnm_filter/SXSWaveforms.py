"""TO be added
"""

__all__ = ['SXSWaveforms']

from .gw_data import *
import sxs
import numpy as np
import warnings


class SXSWaveforms():

    def __init__(self, **kws) -> None:
        self.original_data = {}
        self.padded_data = {}
        self.filtered_data = {}

        self.filename = kws.get('filename', None)
        self.mf = None
        self.chif = None

    def import_sxs_data(self, l, m, interpolate, extrapolation_order=2,
                        download=False, ti=None, tf=None, delta_t=None) -> None:

        waveform = sxs.load(self.filename+"/Lev/rhOverM",
                            extrapolation_order=extrapolation_order, download=download)
        tp = waveform.max_norm_time()
        waveform_lm = waveform[:, waveform.index(l, m)]

        if ti == None:
            t_interp_i = 0
        else:
            t_interp_i = ti + tp
        if tf == None:
            t_interp_f = waveform_lm.t[-1]
        else:
            t_interp_f = tf + tp

        if interpolate:
            if delta_t == None:
                raise ValueError("Invalid delta_t: {}".format(delta_t))

            ts = np.arange(t_interp_i, t_interp_f, delta_t)
            interplated_waveform = waveform_lm.interpolate(ts).data
            self.original_data[str(
                l)+str(m)] = Data(interplated_waveform, index=ts-tp)
        else:
            if delta_t != None:
                warnings.warn("delta_t: {} is not used".format(delta_t))
            index_i = waveform_lm.index_closest_to(t_interp_i)
            index_f = waveform_lm.index_closest_to(t_interp_f)
            waveform_lm_trunc = waveform_lm[index_i:index_f]
            self.original_data[str(
                l)+str(m)] = Data(waveform_lm_trunc.data, index=waveform_lm_trunc.t-tp)

    def get_meta_data(self, download=False) -> None:
        metadata = sxs.load(
            self.filename+"/Lev/metadata.json", download=download)
        if self.mf != None:
            warnings.warn("Overwriting mf: {}".format(self.mf))
        if self.chif != None:
            warnings.warn("Overwriting chif: {}".format(self.chif))
        self.mf = metadata['remnant_mass']
        self.chif = metadata['remnant_dimensionless_spin'][-1]

    def pad_data(self, partition, len_pow) -> None:
        for lm, data in self.original_data.items():
            self.padded_data[lm] = self.pad_data2(data, partition, len_pow)
            
    @staticmethod
    def pad_data2(data, partition, len_pow) -> None:
        padlen = 2**(len_pow+int(np.ceil(np.log2(len(data)))))-len(data)
        data_pad = np.pad(data.values, (padlen//partition, padlen-(padlen//partition)),
                            'constant', constant_values=(0, 0))

        delta_t = data.time_interval
        end1 = data.index[-1] + (padlen-(padlen//partition)) * delta_t
        end2 = data.index[0] - (padlen//partition) * delta_t

        tpad = np.pad(data.index, (padlen//partition, padlen-(padlen//partition)),
                        'linear_ramp', end_values=(end2, end1))
        return Data(data_pad, index=tpad)

    @staticmethod
    def trunc_pad(data, before, after, partition, len_pow):
        truncated_data = data.truncate_data(before=before, after=after)
        return SXSWaveforms.pad_data2(truncated_data, partition, len_pow)
        

    def add_filter(self, model_list):
        for lm, data in self.padded_data.items():
            data_in_freq = data.complex_fft_data(False)
            freq = data.complex_fft_freq(False)
            filter_in_freq = Filter(
                chi=self.chif, mass=self.mf, model_list=model_list).NR_filter(freq)
            data_in_time = np.fft.fft(filter_in_freq*data_in_freq,
                                      norm='ortho')
            self.filtered_data[lm] = Data(data_in_time, index=data.index)
