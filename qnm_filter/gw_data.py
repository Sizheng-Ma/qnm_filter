"""Utilities to manipulate GW data and rational filters.
"""
__all__ = ["RealData", "ComplexData", "Filter", "Noise"]

from .utility import pad_data_for_fft
import astropy.constants as c
import qnm
import pandas as pd
import numpy as np
import scipy.signal as ss
import copy
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
        if model_list != None:
            for l, m, n, p in model_list:
                self.model_list.append(dict(l=l, m=m, n=n, p=p))

    @property
    def get_freq_list(self) -> list:
        """Return a list of QNM frequencies stored in :attr:`Filter.model_list`."""
        freq_list = {}
        for mode in self.model_list:
            this_l = mode["l"]
            this_m = mode["m"]
            this_n = mode["n"]
            this_p = mode["p"]
            omega = qnm.modes_cache(s=-2, l=this_l, m=this_m, n=this_n)(a=self.chi)[0]
            if this_p == "p":
                freq_list[str(this_l) + str(this_m) + str(this_n)] = omega
            elif this_p == "r":
                freq_list[str(this_l) + str(this_m) + str(this_n)] = -np.conj(omega)
        return freq_list

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

    def pos_filter(self, normalized_freq, l, m, n):
        r"""The positive rational filter:

        .. math::
            \frac{\omega-\omega_{lmn}}{\omega-\omega_{lmn}^*}

        Parameters
        ----------
        normalized_freq : array
            in remnant mass, frequencies that rational filters are evaluated at.
        l : int
            angular index
        m : int
            angular index
        n : int
            overtone index

        Returns
        -------
        array
        """
        omega = qnm.modes_cache(s=-2, l=l, m=m, n=n)(a=self.chi)[0]
        return (normalized_freq - omega) / (normalized_freq - np.conj(omega))

    def neg_filter(self, normalized_freq, l, m, n):
        r"""The negative rational filter:

        .. math::
            \frac{\omega+\omega_{lmn}^*}{\omega+\omega_{lmn}}

        Parameters
        ----------
        normalized_freq : array
            in remnant mass, frequencies that rational filters are evaluated at.
        l : int
            angular index
        m : int
            angular index
        n : int
            overtone index

        Returns
        -------
        array
        """
        omega = qnm.modes_cache(s=-2, l=l, m=m, n=n)(a=self.chi)[0]
        return (normalized_freq + np.conj(omega)) / (normalized_freq + omega)

    def single_filter(self, normalized_freq, l, m, n):
        r"""A combination of the negative and postive rational filters

        .. math::
            \frac{\omega-\omega_{lmn}}{\omega-\omega_{lmn}^*}\frac{\omega+\omega_{lmn}^*}{\omega+\omega_{lmn}}

        Parameters
        ----------
        normalized_freq : array
            in remnant mass, frequencies that rational filters are evaluated at.
        l : int
            angular index
        m : int
            angular index
        n : int
            overtone index

        Returns
        -------
        array
        """
        return self.neg_filter(normalized_freq, l, m, n) * self.pos_filter(
            normalized_freq, l, m, n
        )

    def NR_filter(self, freq):
        """Rational filters for numerical-relativity waveforms, removing the modes stored in :attr:`Filter.model_list`.

        Parameters
        ----------
        freq : array
            the unit should be the same as :attr:`Filter.mass`

        Raises
        ------
        ValueError
            When :attr:`Filter.mass` or :attr:`Filter.chi` is not provided
        """
        final_rational_filter = 1
        if not bool(self.model_list):
            return final_rational_filter
        else:
            if (self.mass is None) or (self.chi is None):
                raise ValueError(
                    f"Mass = {self.mass}" f" and Spin = {self.chi} are needed"
                )
        normalized_freq = freq * self.mass
        for mode in self.model_list:
            if mode["p"] == "p":
                final_rational_filter *= self.pos_filter(
                    normalized_freq, mode["l"], mode["m"], mode["n"]
                )
            elif mode["p"] == "r":
                final_rational_filter *= self.neg_filter(
                    normalized_freq, mode["l"], mode["m"], mode["n"]
                )
        return final_rational_filter

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


class DataBase(pd.Series):
    """A Base container for time-domain gravitational data

    Parameters
    ----------
    ifo : str
        name of interferometer.
    """

    def __init__(self, *args, ifo=None, **kwargs):
        super(DataBase, self).__init__(*args, **kwargs)
        self.ifo = ifo

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __add__(self, other):
        if all(self.index.values != other.index.values):
            raise ValueError("Two arrays don't have the same time")
        return DataBase(
            self.values + other.values, index=self.index.values, ifo=self.ifo
        )

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


class ComplexData(DataBase):
    """Container for complex-valued time-domain numerical-relativity waveforms.

    Parameters
    ----------
    ifo : str
        name of interferometer.
    """

    def __add__(self, other):
        if all(self.index.values != other.index.values):
            raise ValueError("Two arrays don't have the same time")
        return ComplexData(
            self.values + other.values, index=self.index.values, ifo=self.ifo
        )

    def pad_complex_data_for_fft(self, partition, len_pow):
        """Pad zeros on both sides for FFT

        Parameters
        ----------
        partition : int
            fraction of zeros to be padded on the left
        len_pow : int
            the final length of padded data is :math:`2^{\textrm{len_pow}}`

        Returns
        -------
        ComplexData
            padded data
        """
        tpad, data_pad = pad_data_for_fft(self, partition, len_pow)
        return ComplexData(data_pad, index=tpad, ifo=self.ifo)

    @property
    def fft_freq(self):
        """FFT angular frequency stamps."""
        fft_freq = np.fft.fftfreq(len(self), d=self.time_interval) * 2 * np.pi
        return fft_freq

    @property
    def shifted_fft_freq(self):
        """Shifted FFT angular frequency stamps, with the zero-frequency component being at the center."""
        fft_freq = np.fft.fftfreq(len(self), d=self.time_interval) * 2 * np.pi
        return np.fft.fftshift(fft_freq)

    @property
    def fft_data(self):
        """FFT of the NR waveform"""
        fft_data = np.fft.ifft(self.values, norm="ortho")
        return fft_data

    @property
    def shifted_fft_data(self):
        """Shifted FFT of the NR waveform, with the zero-frequency component being at the center."""
        fft_data = np.fft.ifft(self.values, norm="ortho")
        return np.fft.fftshift(fft_data)

    def truncate_data(self, before=None, after=None, copy=None):
        """Truncate data before and after some index value

        Parameters
        ----------
        before : double, optional
            truncate all data before this index value, by default None
        after : double, optional
            truncate all data after this index value, by default None
        copy : copy, optional
            return a copy of the truncated data, by default None

        Returns
        -------
        ComplexData
            truncated data
        """
        truncated_waveform = self.truncate(before, after, copy)
        return ComplexData(
            truncated_waveform.values, index=truncated_waveform.index, ifo=self.ifo
        )


class RealData(DataBase):
    """Container for real-valued time-domain gravitational data.

    Parameters
    ----------
    ifo : str
        name of interferometer.
    """

    def __add__(self, other):
        if all(self.index.values != other.index.values):
            raise ValueError("Two arrays don't have the same time")
        return RealData(
            self.values + other.values, index=self.index.values, ifo=self.ifo
        )

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
        t0 : float
            target time to be preserved after downsampling.
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

        raw_data = self.values
        raw_time = self.index.values

        if srate:
            ds = int(round(self.fft_span / srate))
        else:
            ds = 1

        if t0 is not None:
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

        return RealData(cond_data, index=cond_time, ifo=self.ifo)


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
            self.psd = RealData(kwargs.get("psd"), index=freq, ifo=ifo)
        if "asd" in kwargs:
            freq = kwargs.pop("freq")
            self.asd = RealData(kwargs.get("asd"), index=freq, ifo=ifo)
        if "acf" in kwargs:
            time = kwargs.pop("time")
            self.acf = RealData(kwargs.get("acf"), index=time, ifo=ifo)
        if "signal" in kwargs:
            time = kwargs.pop("time")
            self.signal = RealData(kwargs.get("signal"), index=time, ifo=ifo)

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
            self, attr_name, RealData(filereader[:, 1], index=filereader[:, 0], ifo=ifo)
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
        fs = 2 * psd.index[-1]
        if psd.index[0] != 0:
            raise ValueError("PSD frequency-series is expected to start at 0 Hz")
        deltaf = np.diff(psd.index)
        if not all(abs(deltaf - deltaf[0]) < 1e-8):
            raise ValueError("PSD frequency-series is expected to be evenly spaced")
        rho = 0.5 * np.fft.irfft(psd) * fs
        return RealData(rho, index=np.arange(len(rho)) / fs, ifo=self.ifo)

    def from_psd(self):
        """Compute ASD and ACF from PSD"""
        self.asd = RealData(np.sqrt(self.psd.values), index=self.psd.time, ifo=self.ifo)
        self.acf = self.__psd_to_acf(self.psd)

    def from_asd(self):
        """Compute PSD and ACF from ASD"""
        self.psd = RealData(self.asd.values**2, index=self.asd.time, ifo=self.ifo)
        self.acf = self.__psd_to_acf(self.psd)

    def from_acf(self):
        """Compute PSD and ASD from ACF"""
        dt = self.acf.time_interval
        freq_samp = np.fft.rfftfreq(len(self.acf), d=dt)
        psd_temp = 2 * dt * np.fft.rfft(self.acf)
        # TODO: This could go wrong. PSD is supposed to be real. Better way?
        psd_temp = np.real(psd_temp)
        self.psd = RealData(psd_temp, index=freq_samp, ifo=self.ifo)
        self.asd = np.sqrt(self.psd)

    def welch(self, **kws):
        """Estimate PSD from data using Welch's method."""
        fs = self.signal.fft_span
        nperseg = fs / kws.get("sampling_rate", 1)

        freq, psd = ss.welch(self.signal, fs=fs, nperseg=nperseg)
        self.psd = RealData(psd, index=freq, ifo=self.ifo)

    @property
    def bilby_psd(self):
        """Construct a Bilby `PowerSpectralDensity` instance."""
        return bilby.gw.detector.PowerSpectralDensity(
            frequency_array=self.psd.time, psd_array=self.psd.values
        )
