"""Utilities to manipulate GW data and rational filters.
"""
__all__ = ["Data", "Filter", "Noise"]

import astropy.constants as c
import qnm
import pandas as pd
import numpy as np
import scipy.signal as ss
import warnings
import bilby

T_MSUN = c.M_sun.value * c.G.value / c.c.value**3


class Filter:
    """Container for rational filters.

    Attributes
    ----------
    chi : float
        remnant dimensionless spin.
    mass : float
        remnant mass, in solar mass.
    model_list : a list of dictionaries
        quasinormal modes to be filtered.
    """

    def __init__(self, chi=None, mass=None, model_list=None):
        """Constructor"""
        self.chi = chi
        self.mass = mass  # in solar mass

        self.model_list = []
        for l, m, n in model_list:
            self.model_list.append(dict(l=l, m=m, n=n))

    @property
    def get_spin(self) -> float:
        """Return :attr:`Filter.chi`."""
        return self.chi

    @property
    def get_mass(self) -> float:
        """Return :attr:`Filter.mass`."""
        return self.mass

    @property
    def get_model_list(self) -> list[dict]:
        """Return :attr:`Filter.model_list`."""
        return self.model_list

    @staticmethod
    def mass_unit(mass) -> float:
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
        return (
            (normalized_freq - omega)
            / (normalized_freq - np.conj(omega))
            * (normalized_freq + np.conj(omega))
            / (normalized_freq + omega)
        )

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
                raise ValueError(
                    f"Mass = {self.mass}" f" and Spin = {self.chi} are needed"
                )
        normalized_freq = freq * self.mass * T_MSUN
        for mode in self.model_list:
            final_rational_filter *= self.single_filter(
                -normalized_freq, mode["l"], mode["m"], mode["n"]
            )
        return final_rational_filter


class Data(pd.Series):
    """Container for gravitational data.

    Attributes
    ----------
    ifo : str
        name of interferometer.
    """

    def __init__(self, *args, ifo=None, **kwargs):
        super(Data, self).__init__(*args, **kwargs)
        self.ifo = ifo

    @property
    def time(self):
        """Time stamps."""
        return self.index.values

    @property
    def time_interval(self) -> float:
        """Interval of the time stamps."""
        return self.index[1] - self.index[0]

    @property
    def fft_span(self) -> float:
        """Span of FFT."""
        return 1.0 / self.time_interval

    @property
    def fft_freq(self):
        """FFT angular frequency stamps."""
        return np.fft.rfftfreq(len(self), d=self.time_interval) * 2 * np.pi

    @property
    def fft_data(self):
        """FFT of gravitational-wave data."""
        return np.fft.rfft(self.values, norm="ortho")

    def condition(
        self,
        t0=None,
        srate=None,
        flow=None,
        fhigh=None,
        trim=0.25,
        remove_mean=True,
        **kwargs,
    ):
        """Condition data.

        Credit: This function is from `git@github.com:maxisi/ringdown.git`.

        Arguments
        ---------
        flow : float
            lower frequency for high passing.
        fhigh : float
            higher frequency for low passing.
        srate : int
            sampling frequency after downsampling.
        remove_mean : bool
            explicitly remove mean from time series after conditioning.
        trim : float
            fraction of data to trim from edges after conditioning, to avoid
            spectral issues if filtering.

        Returns
        -------
        cond_data : Data
            conditioned data object.
        """

        srate = kwargs.pop("srate", srate)
        flow = kwargs.pop("flow", flow)
        raw_data = self.values
        raw_time = self.index.values

        ds = int(round(self.fft_span / srate))

        if t0 is not None:
            ds = int(ds or 1)
            i = np.argmin(abs(raw_time - t0))
            raw_time = np.roll(raw_time, -(i % ds))
            raw_data = np.roll(raw_data, -(i % ds))

        fny = 0.5 * self.fft_span
        # Filter
        if flow and not fhigh:
            b, a = ss.butter(4, flow / fny, btype="highpass", output="ba")
        elif fhigh and not flow:
            b, a = ss.butter(4, fhigh / fny, btype="lowpass", output="ba")
        elif flow and fhigh:
            b, a = ss.butter(
                4, (flow / fny, fhigh / fny), btype="bandpass", output="ba"
            )

        if flow or fhigh:
            cond_data = ss.filtfilt(b, a, raw_data)
        else:
            cond_data = raw_data
            cond_time = raw_time

        if ds and ds > 1:
            cond_data = ss.decimate(cond_data, ds, zero_phase=True)
            cond_time = raw_time[::ds]

        N = len(cond_data)
        istart = int(round(trim * N))
        iend = int(round((1 - trim) * N))

        cond_time = cond_time[istart:iend]
        cond_data = cond_data[istart:iend]

        if remove_mean:
            cond_data -= np.mean(cond_data)

        return Data(cond_data, index=cond_time, ifo=self.ifo)


class Noise:
    """Container for noise

    Attributes
    ----------
    ifo : str
        name of interferometer
    psd : Data
        one-sided power spectral density
    asd : Data
        amplitude spectral density
    acf : Data
        autocorrelation function
    signal : Data
        time-domain noisy signal
    """

    def __init__(self, ifo=None, **kwargs) -> None:
        self.ifo = ifo
        if "psd" in kwargs:
            freq = kwargs.pop("freq")
            self.psd = Data(kwargs.get("psd"), index=freq, ifo=ifo)
        if "asd" in kwargs:
            freq = kwargs.pop("freq")
            self.asd = Data(kwargs.get("asd"), index=freq, ifo=ifo)
        if "acf" in kwargs:
            time = kwargs.pop("time")
            self.acf = Data(kwargs.get("acf"), index=time, ifo=ifo)
        if "signal" in kwargs:
            time = kwargs.pop("time")
            self.signal = Data(kwargs.get("signal"), index=time, ifo=ifo)

    def load_noise_curve(self, attr_name, filename, ifo=None):
        """Read a txt/dat file and store the data in target attribute
        :attr:`attr_name`. The file should have two columns.

        Parameters
        ----------
        attr_name : string
            name of target attribute, could be psd, asd, or acf.
        filename : string
            the file name to be read.
        ifo : string, optional
            name of interferometer, by default None
        """
        filereader = np.loadtxt(filename)
        setattr(
            self, attr_name, Data(filereader[:, 1], index=filereader[:, 0], ifo=ifo)
        )

    def __psd_to_acf(self, psd):
        """Inverse FFT PSD to ACF

        Parameters
        ----------
        psd : Data
            one-sided power spectral density

        Returns
        -------
        Data
            autocorrelation function
        """
        fs = 2 * (psd.index[-1] - psd.index[0])
        rho = 0.5 * np.fft.irfft(psd) * fs
        return Data(rho, index=np.arange(len(rho)) / fs, ifo=self.ifo)

    def from_psd(self):
        """Compute ASD and ACF from PSD"""
        self.asd = Data(np.sqrt(self.psd.values), index=self.psd.time, ifo=self.ifo)
        self.acf = self.__psd_to_acf(self.psd)

    def from_asd(self):
        """Compute PSD and ACF from ASD"""
        self.psd = Data(self.asd.values**2, index=self.asd.time, ifo=self.ifo)
        self.acf = self.__psd_to_acf(self.psd)

    def from_acf(self):
        """Compute PSD and ASD from ACF"""
        dt = self.acf.time_interval
        freq_samp = np.fft.rfftfreq(len(self.acf), d=dt)
        psd_temp = 2 * dt * np.fft.rfft(self.acf)
        # TODO: This could go wrong. PSD is supposed to be real. Better way?
        psd_temp = np.real(psd_temp)
        self.psd = Data(psd_temp, index=freq_samp, ifo=self.ifo)
        self.asd = np.sqrt(self.psd)

    def welch(self, **kws):
        """Estimate PSD from data using Welch's method."""
        fs = self.signal.fft_span
        nperseg = fs / kws.get("sampling_rate", 1)

        freq, psd = ss.welch(self.signal, fs=fs, nperseg=nperseg)
        self.psd = Data(psd, index=freq, ifo=self.ifo)

    @property
    def bilby_psd(self):
        """Construct a Bilby `PowerSpectralDensity` instance."""
        return bilby.gw.detector.PowerSpectralDensity(
            frequency_array=self.psd.time, psd_array=self.psd.values
        )
