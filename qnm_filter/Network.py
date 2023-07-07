"""Defining the core :class:`Network` class.
"""

__all__ = ["Network"]

from .gw_data import *
from .utility import time_shift_from_sky
import h5py
import numpy as np
import scipy.linalg as sl
import warnings


class Network(object):
    """Perform a ringdown filter analysis. Stores all the needed information.

    Example usage::

        import qnm_filter
        input = dict(model_list = [(2, 2, 0, 'p')], #l, m, n, prograde/retrograde
                    # trucation time (geocenter, in second)
                    t_init = 1126259462.4083147+2.0*1e-3,
                    # length of the analysis segment (in second)
                    segment_length = 0.2,
                    # sampling rate after conditioning (in Hz)
                    srate = 2048,
                    # sky localization
                    ra = 1.95, dec = -1.27,
                    # lower limit of the high-pass filter (in Hz)
                    flow = 20)
        fit = qnm_filter.Network(**input)
        fit.import_data('H-H1_GWOSC_16KHZ_R1-1126259447-32.hdf5')
        fit.detector_alignment()
        fit.condition_data('original_data', **input)
        fit.compute_acfs('original_data')
        fit.cholesky_decomposition()
        fit.first_index()
        fit.add_filter(mass=68.5, chi=0.69, **input)
        final_likelihood = fit.compute_likelihood(apply_filter=True)

    Attributes
    ----------
    original_data : dict
        dictionary containing unfiltered data for each detector.
    filtered_data : dict
        dictionary containing filtered data for each detector.
    acfs : dict
        dictionary containing autocovariance functions for each detector.
    start_times : dict
        dictionary containing trucation time (start time of analysis segment)
        for each detector, determined by specified sky location.
    cholesky_L : dict
        dictionary containing Cholesky factorizations of detectors.
    i0_dict : dict
        dictionary containing the array index of the first time of the analysis segment.
        The computation of :attr:`i0_dict` needs to be after the conditioning part.
    ra : float
        source right ascension, in radian.
    dec : float
        source declination, in radian.
    t_init : float
        trucation time (start time of analysis segment) at geocenter.
    segment_length : float
        width of analysis segment
    """

    def __init__(self, **kws) -> None:
        """Constructor"""
        self.original_data = {}
        self.filtered_data = {}
        self.acfs = {}
        self.start_times = {}
        self.cholesky_L = {}
        self.i0_dict = {}

        self.ra = kws.get("ra", None)
        self.dec = kws.get("dec", None)
        self.t_init = kws.get("t_init", None)
        self.srate = kws.get("srate", None)
        self.segment_length = kws.get("segment_length", None)

    def import_ligo_data(self, filename) -> None:
        """Read data from disk and store data in :attr:`Network.original_data`.

        Supports only HDF5 files downloaded from https://www.gw-openscience.org.

        Parameters
        ----------
        filename : string
            name of file
        """

        with h5py.File(filename, "r") as f:
            h = f["strain/Strain"][:]
            t_start = f["meta/GPSstart"][()]
            duration = f["meta/Duration"][()]
            ifo = str(f["meta/Detector"][()], "utf-8")

            time = np.linspace(t_start, t_start + duration, num=len(h), endpoint=False)

            self.original_data[ifo] = RealData(h, index=time, ifo=ifo)

    def import_data_array(self, attr_name, data, time, ifo) -> None:
        """Add the inputted data to a dynamic/existing attribute.

        Parameters
        ----------
        attr_name : string
            Name of the dynamic attribute
        data : ndarray
            Inputted data
        time : ndarray
            Time stamps
        ifo : string
            Name of interferometer
        """
        getattr(self, attr_name)[ifo] = RealData(data, index=time, ifo=ifo)

    def detector_alignment(self) -> None:
        """Set the start times of analysis segment at different
        interferometers :attr:`Network.start_times` using sky location.
        """
        if self.t_init == None:
            raise ValueError("t_init is not provided")

        for ifo, data in self.original_data.items():
            if self.ra == None or self.dec == None:
                shifted_time = self.t_init
            else:
                dt_ifo = time_shift_from_sky(ifo, self.ra, self.dec, self.t_init)
                shifted_time = self.t_init + dt_ifo
            self.start_times[ifo] = shifted_time
            if not (data.time[0] < shifted_time < data.time[-1]):
                raise ValueError("Invalid start time for {}".format(ifo))

    def first_index(self):
        """Find the index of a data point that is closet to the choosen
        start time :attr:`Network.start_times` for each interferometer."""
        for ifo, data in self.original_data.items():
            t0 = self.start_times[ifo]
            self.i0_dict[ifo] = abs(data.time - t0).argmin()

    def shift_first_index(self, n):
        """Shift the first index of the analysis segment by `n`."""
        for ifo, _ in self.original_data.items():
            self.i0_dict[ifo] += n

    @property
    def sampling_n(self) -> int:
        """Number of data points in analysis segment.

        Returns
        -------
        int
            Length of truncated data array
        """
        return int(round(self.segment_length * self.srate))

    def truncate_data(self, network_data) -> dict:
        """Select segments of the given data that are in analysis segment.

        Parameters
        ----------
        network_data : dictionary
            Network GW data to be truncated.

        Returns
        -------
        data : dictionary
            Truncated GW data for all interferometers.
        """
        data = {}
        i0s = self.i0_dict
        for i, d in network_data.items():
            if abs(d.fft_span / self.srate - 1) > 1e-8:
                raise ValueError("Sampling rate is not correct: {}".format(d.fft_span))
            data[i] = RealData(d.iloc[i0s[i] : i0s[i] + self.sampling_n])
        return data

    def condition_data(self, attr_name, **kwargs) -> None:
        """Condition data for all interferometers.

        Parameters
        ----------
        attr_name : string
            Name of data to be conditioned
        """
        unconditioned_data = getattr(self, attr_name)
        for ifo, data in unconditioned_data.items():
            t0 = self.start_times[ifo]
            getattr(self, attr_name)[ifo] = data.condition(
                t0=t0,
                srate=self.srate,
                flow=kwargs.get("flow"),
                fhigh=kwargs.get("fhigh"),
                trim=kwargs.get("trim", 0.25),
                remove_mean=kwargs.get("remove_mean", True),
            )

    def compute_acfs(self, attr_name, **kws) -> None:
        """Compute ACFs with data named `attr_name`.

        Parameters
        ----------
        attr_name : string
            Name of data for ACF estimation
        """
        noisy_data = getattr(self, attr_name)
        if self.acfs:
            warnings.warn("Overwriting ACFs")
        for ifo, data in noisy_data.items():
            noise = Noise(time=data.time, signal=data.values)
            noise.welch(**kws)
            noise.from_psd()
            self.acfs[ifo] = noise.acf

    def cholesky_decomposition(self) -> None:
        """Compute the Cholesky-decomposition of covariance matrix :math:`C = L^TL`,
        and the inverse of :math:`L`.
        """
        for ifo, acf in self.acfs.items():
            if self.sampling_n > len(self.acfs[ifo]) / 2:
                raise ValueError("The sampling_n is more than half the acf length")
            if abs(acf.fft_span / self.srate - 1) > 1e-8:
                raise ValueError(
                    "Sampling rate is not correct: {}".format(acf.fft_span)
                )
            truncated_acf = acf.iloc[: self.sampling_n].values
            L = np.linalg.cholesky(sl.toeplitz(truncated_acf))
            self.cholesky_L[ifo] = L

    def compute_likelihood(self, apply_filter=True) -> float:
        """Compute likelihood for interferometer network.

        Arguments
        ---------
        apply_filter : bool
            option to apply rational filters (default True).

        Returns
        -------
        likelihood : float
            The likelihood of interferometer network
        """
        likelihood = 0

        if not apply_filter:
            truncation = self.truncate_data(self.original_data)
        else:
            truncation = self.truncate_data(self.filtered_data)

        for ifo, data in truncation.items():
            wd = sl.cho_solve((self.cholesky_L[ifo], True), data)
            likelihood -= 0.5 * np.dot(data, wd)
        return likelihood

    def add_filter(self, **kwargs):
        """Apply rational filters to :attr:`Network.original_data` and store
        the filtered data in :attr:`Network.filtered_data`."""
        for ifo, data in self.original_data.items():
            data_in_freq = data.fft_data
            freq = data.fft_freq
            filter_in_freq = Filter(**kwargs).total_filter(freq)
            ifft = np.fft.irfft(
                filter_in_freq * data_in_freq, norm="ortho", n=len(data)
            )
            self.filtered_data[ifo] = RealData(ifft, index=data.index, ifo=ifo)

    def likelihood_vs_mass_spin(self, M_est, chi_est, **kwargs) -> float:
        """Compute likelihood for the given mass and spin.

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
        model_list = kwargs.pop("model_list")
        self.add_filter(mass=M_est, chi=chi_est, model_list=model_list)
        return self.compute_likelihood(apply_filter=True)

    def compute_SNR(self, data, template, ifo, optimal) -> float:
        """Compute matched-filter/optimal SNR.

        Parameters
        ----------
        data : ndarray
            Time-series data
        template : ndarray
            Ringdown template
        ifo : string
            Name of interferometer
        optimal: bool
            Compute optimal SNR
        """
        template_w = sl.cho_solve((self.cholesky_L[ifo], True), template)
        snr_opt = np.sqrt(np.dot(template, template_w))
        if optimal:
            return snr_opt
        else:
            snr = np.dot(data, template_w) / snr_opt
            return snr
