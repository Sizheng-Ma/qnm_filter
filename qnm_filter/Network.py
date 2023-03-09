"""Defining the core :class:`Network` class.
"""

__all__ = ["Network"]

from .gw_data import *
from .utility import *
import h5py
import lal
import numpy as np
import scipy.linalg as sl
import warnings
from scipy.special import logsumexp



class Network(object):
    """Perform a ringdown filter analysis. Stores all the needed information.

    Example usage::

        import qnm_filter
        input = dict(model_list = [(2, 2, 0)], #l, m, n
                    # trucation time (geocenter, in second)
                    t_init = 1126259462.4083147+2.0*1e-3,
                    # length of the analysis window (in second)
                    window_width = 0.2,
                    # sampling rate after conditioning (in Hz)
                    srate = 2048,
                    # sky localization
                    ra = 1.95, dec = -1.27,
                    # lower limit of the high-pass filter (in Hz)
                    flow = 20)
        fit = qnm_filter.Network(**input)
        fit.import_data('H-H1_GWOSC_16KHZ_R1-1126259447-32.hdf5')
        fit.detector_alignment(**input)
        fit.condition_data('original_data', **input)
        fit.compute_acfs('original_data')
        fit.cholesky_decomposition()
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
        dictionary containing trucation time (start time of analysis window)
        for each detector, determined by specified sky location.
    cholesky_L : dict
        dictionary containing Cholesky-decomposition of covariance matrix for each detector.
    inverse_cholesky_L : dict
        dictionary containing the inverse of Cholesky-decomposition.
    ra : float
        source right ascension, in radian.
    dec : float
        source declination, in radian.
    t_init : float
        trucation time (start time of analysis window) at geocenter.
    window_width : float
        width of analysis window
    """

    def __init__(self, **kws) -> None:
        """Constructor"""
        self.original_data = {}
        self.conditioned_data = {}
        self.filtered_data = {}
        self.acfs = {}
        self.start_times = {}
        self.cholesky_L = {}
        self.inverse_cholesky_L = {}

        self.ra = kws.get("ra", None)
        self.dec = kws.get("dec", None)
        self.t_init = kws.get("t_init", None)
        self.window_width = kws.get("window_width", None)

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

            self.original_data[ifo] = Data(h, index=time, ifo=ifo)

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
        getattr(self, attr_name)[ifo] = Data(data, index=time, ifo=ifo)

    def detector_alignment(self, **kwargs) -> None:
        """Set the start times of analysis window at different
        interferometers :attr:`Network.start_times` using sky location.

        Parameters
        ----------
        t_init : float
            The start time of analysis window at the geocenter.
        """
        t_init = kwargs.pop("t_init", None)
        if t_init == None:
            raise ValueError("t_init is not provided")
        tgps = lal.LIGOTimeGPS(t_init)

        for ifo, data in self.original_data.items():
            if self.ra == None or self.dec == None:
                shifted_time = t_init
            else:
                location = lal.cached_detector_by_prefix[ifo].location
                dt_ifo = lal.TimeDelayFromEarthCenter(location, self.ra, self.dec, tgps)
                shifted_time = t_init + dt_ifo
            self.start_times[ifo] = shifted_time
            if not (data.time[0] < shifted_time < data.time[-1]):
                raise ValueError("Invalid start time for {}".format(ifo))
    
    @property
    def first_index(self) -> dict:
        """Find the index of a data point that is closet to the choosen
        start time :attr:`Network.start_times` for each interferometer.

        Returns
        -------
        i0_dict : dictionary
            dictionary containing the start indices for all interferometers.
        """
        i0_dict = {}
        for ifo, data in self.conditioned_data.items():
            t0 = self.start_times[ifo]
            i0_dict[ifo] = abs(data.time - t0).argmin()
        return i0_dict
    
    @property
    def sampling_n(self) -> int:
        """Number of data points in analysis window.

        Should be the same for all interferometers.

        Returns
        -------
        Length of truncated data array
        """
        n_dict = {}
        for ifo, data in self.conditioned_data.items():
            n_dict[ifo] = int(round(self.window_width / data.time_interval))
        if len(set(n_dict.values())) > 1:
            raise ValueError("Detectors have different sampling rates")

        return list(n_dict.values())[0]
    
    def truncate_data(self, network_data) -> dict:
        """Select segments of the given data that are in analysis window.

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
        i0s = self.first_index
        for i, d in network_data.items():
            data[i] = Data(d.iloc[i0s[i] : i0s[i] + self.sampling_n])
        return data
            
    def condition_data(self, attr_name, trim=0.0, **kwargs) -> None:
        """Condition data for all interferometers.

        Parameters
        ----------
        attr_name : string
            Name of data to be conditioned
        """
        unconditioned_data = getattr(self, attr_name)
        for ifo, data in unconditioned_data.items():
            self.conditioned_data[ifo] = data.condition(trim = trim, **kwargs)
            
    def roll_condition(self, attr_name, trim=0.0, **kwargs) -> None:
        """Condition data for all interferometers.

        Parameters
        ----------
        attr_name : string
            Name of data to be conditioned
        """
        unconditioned_data = getattr(self, attr_name)
        
        for ifo, data in unconditioned_data.items():
            fft_span = data.fft_span
            srate = kwargs.get('srate', None)
            t0 = self.start_times[ifo]
            
            ds = int(round(fft_span/srate))
            i = np.argmin(abs(data.index.values - t0))
            
            rolled_time = np.roll(data.index.values, -(i % ds))
            rolled_data = np.roll(data.values, -(i % ds))
            self.conditioned_data[ifo] = Data(rolled_data, index=rolled_time, ifo=ifo)
        self.condition_data('conditioned_data', trim=trim, **kwargs)

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
            # TODO: Warning: this assumes acf has the same time step as data, which could go wrong.
            truncated_acf = acf.iloc[: self.sampling_n].values
            L = np.linalg.cholesky(sl.toeplitz(truncated_acf))
            L_inv = np.linalg.inv(L)
            norm = np.sqrt(np.sum(abs(np.dot(L_inv, L) - np.identity(len(L))) ** 2))
            if abs(norm) > 1e-8:
                raise ValueError("Inverse of L is not correct")

            self.cholesky_L[ifo] = L
            self.inverse_cholesky_L[ifo] = L_inv

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
            truncation = self.truncate_data(self.conditioned_data)
        else:
            truncation = self.truncate_data(self.filtered_data)

        for ifo, data in truncation.items():
            wd = np.dot(self.inverse_cholesky_L[ifo], data)
            likelihood -= 0.5 * np.dot(wd, wd)
        return likelihood

    def add_filter(self, **kwargs):
        """Apply rational filters to :attr:`Network.original_data` and store
        the filtered data in :attr:`Network.filtered_data`."""
        for ifo, data in self.conditioned_data.items():
            data_in_freq = data.fft_data
            freq = data.fft_freq
            filter_in_freq = Filter(**kwargs).total_filter(freq)
            ifft = np.fft.irfft(
                filter_in_freq * data_in_freq, norm="ortho", n=len(data)
            )
            self.filtered_data[ifo] = Data(ifft, index=data.index, ifo=ifo)

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
    

    def t_init_evidence(self, new_t, massspace, chispace, num_cpu = -1, **kwargs) -> float:
        """ Calculates evidence for a given value of t_init (ms afer t0)
        """
        kwargs["t_init"] = new_t
        self.detector_alignment(**kwargs)
        self.roll_condition('original_data', **kwargs)
        likelihood_data = parallel_compute(self, massspace, chispace, num_cpu = num_cpu, **kwargs)
        evidence = logsumexp(likelihood_data, axis = (0,1))
        return evidence
        
    
    def compute_SNR(self, data, template, ifo, optimal=False) -> float:
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
        template_w = np.dot(self.inverse_cholesky_L[ifo], template)
        data_w = np.dot(self.inverse_cholesky_L[ifo], data)
        snr_opt = np.sqrt(np.dot(template_w, template_w))
        if optimal:
            return snr_opt
        else:
            snr = np.dot(data_w, template_w) / snr_opt
            return snr
