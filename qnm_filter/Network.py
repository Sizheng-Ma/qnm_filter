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
        self.filtered_ftau_data = {}
        self.acfs = {}
        self.start_times = {}
        self.cholesky_L = {}
        self.i0_dict = {}

        # Optional floor on the PSD below `flow`, as (flow_hz, frac), or None to
        # disable. High-passing drives the PSD to ~0 below `flow`, making the
        # covariance nearly singular there; flooring those bins to
        # frac * (in-band median) keeps C^{-1} bounded. Applied in compute_acfs.
        self.psd_floor = None
        # Raw Welch PSDs, reused by compute_acfs(acfs_update=True): the noise data
        # is fixed during sampling, only the calibration scaling changes.
        self._raw_psd_cache = {}
        # Covariance solver prepared by the last call to cholesky_decomposition()
        # ("cholesky") or compute_toeplitz() ("toeplitz"), and the diagonal
        # jitter (fraction of acf[0]) used to build it.
        self._toeplitz_col = {}
        self._solver = "cholesky"
        self._eps_frac = 0.0

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

    def compute_acfs(self, attr_name, cal_response=None, acfs_update=False, **kws) -> None:
        """Compute ACFs with data named `attr_name`.

        Parameters
        ----------
        attr_name : string
            Name of data for ACF estimation
        cal_response : dict, optional
            {ifo: R(omega)} calibration response. If given, each IFO's PSD is
            multiplied by ``|R(omega)|^2`` before the ACF is computed. IFOs not in
            the dict are left unchanged.
        acfs_update : bool
            Set True when recomputing the ACFs at every likelihood call (e.g.
            when sampling calibration). Reuses the cached raw Welch PSD instead of
            re-estimating it, and does not warn about overwriting ACFs.
        **kws
            Passed to :meth:`Noise.welch`.
        """
        noisy_data = getattr(self, attr_name)
        if self.acfs and not acfs_update:
            warnings.warn("Overwriting ACFs")
        for ifo, data in noisy_data.items():
            noise = Noise(time=data.time, signal=data.values)

            psd_key = (attr_name, ifo, tuple(sorted(kws.items())))
            if acfs_update and psd_key in self._raw_psd_cache:
                freqs, values = self._raw_psd_cache[psd_key]
                noise.psd = RealData(values, index=freqs, ifo=ifo)
            else:
                noise.welch(**kws)
                self._raw_psd_cache[psd_key] = (
                    np.asarray(noise.psd.time).copy(),
                    np.asarray(noise.psd.values).copy(),
                )

            if cal_response is not None and ifo in cal_response:
                freqs = np.asarray(noise.psd.time)
                scale = np.abs(cal_response[ifo](2 * np.pi * freqs)) ** 2
                noise.psd = RealData(noise.psd.values * scale, index=freqs, ifo=ifo)

            if self.psd_floor is not None:
                flow_hz, frac = self.psd_floor
                freqs = np.asarray(noise.psd.time)
                psd = np.asarray(noise.psd.values).copy()
                inband = psd[freqs >= flow_hz]
                if inband.size:
                    below = freqs < flow_hz
                    psd[below] = np.maximum(psd[below], frac * np.median(inband))
                    noise.psd = RealData(psd, index=freqs, ifo=ifo)

            noise.from_psd()
            self.acfs[ifo] = noise.acf

    def _truncated_acf(self, ifo, eps_frac):
        """First sampling_n entries of the ACF, with diagonal jitter
        ``eps_frac * acf[0]`` added to the zero lag."""
        acf = self.acfs[ifo]
        if self.sampling_n > len(acf) / 2:
            raise ValueError("The sampling_n is more than half the acf length")
        if abs(acf.fft_span / self.srate - 1) > 1e-8:
            raise ValueError("Sampling rate is not correct: {}".format(acf.fft_span))
        truncated_acf = acf.iloc[: self.sampling_n].values
        if eps_frac:
            truncated_acf = np.array(truncated_acf, dtype=float, copy=True)
            truncated_acf[0] += eps_frac * truncated_acf[0]
        return truncated_acf

    def cholesky_decomposition(self, eps_frac=0.0) -> None:
        """Compute the Cholesky-decomposition of covariance matrix :math:`C = L^TL`,
        and the inverse of :math:`L`.

        Parameters
        ----------
        eps_frac : float
            Diagonal jitter added to the covariance, as a fraction of acf[0]
            (default 0). Also used by later :meth:`compute_toeplitz` calls.
        """
        for ifo in self.acfs:
            truncated_acf = self._truncated_acf(ifo, eps_frac)
            L = np.linalg.cholesky(sl.toeplitz(truncated_acf))
            self.cholesky_L[ifo] = L
        self._eps_frac = eps_frac
        self._solver = "cholesky"

    def compute_toeplitz(self, eps_frac=None) -> None:
        """Prepare a Toeplitz (Levinson) solve of the covariance instead of a
        Cholesky factorisation.

        :meth:`compute_likelihood` then evaluates :math:`d^T C^{-1} d` with
        ``scipy.linalg.solve_toeplitz`` (O(N^2)) rather than a Cholesky factor
        (O(N^3)); the result is the same to machine precision. Used when the
        covariance changes at every likelihood call (``acfs_update=True``).
        Call :meth:`cholesky_decomposition` to switch back.

        Parameters
        ----------
        eps_frac : float, optional
            Diagonal jitter as a fraction of acf[0]. Defaults to the value last
            passed to :meth:`cholesky_decomposition`.
        """
        if eps_frac is None:
            eps_frac = self._eps_frac
        for ifo in self.acfs:
            self._toeplitz_col[ifo] = self._truncated_acf(ifo, eps_frac)
        self._solver = "toeplitz"

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
            if self._solver == "toeplitz":
                col = self._toeplitz_col[ifo]
                d = np.array(data, dtype=float)  # writable copy: levinson rejects read-only buffers
                x = sl.solve_toeplitz((col, col), d)
                likelihood -= 0.5 * np.dot(d, x)
            else:
                y = sl.solve_triangular(self.cholesky_L[ifo], data, lower=True)
                likelihood -= 0.5 * np.dot(y, y)
        return likelihood

    def compute_likelihood_ftau(self) -> float:
        """Compute likelihood for interferometer network.

        Arguments

        Returns
        -------
        likelihood : float
            The likelihood of interferometer network
        """
        likelihood = 0

        truncation = self.truncate_data(self.filtered_ftau_data)

        for ifo, data in truncation.items():
            y = sl.solve_triangular(self.cholesky_L[ifo], data, lower=True)
            likelihood -= 0.5 * np.dot(y, y)
        return likelihood

    def add_filter(self, window='Tukey', alpha=0.2, cal_response=None, **kwargs):
        """Apply rational filters to :attr:`Network.original_data` and store
        the filtered data in :attr:`Network.filtered_data`.

        If ``cal_response`` ({ifo: R(omega)}) is given, the data of each IFO in it
        is also multiplied by R(omega) in the frequency domain."""
        for ifo, data in self.original_data.items():
            data_in_freq = data.fft_data(window=window, alpha=alpha)
            freq = data.fft_freq
            filter_in_freq = Filter(**kwargs).total_filter(freq)
            prod = filter_in_freq * data_in_freq
            if cal_response is not None and ifo in cal_response:
                prod = prod * cal_response[ifo](freq)
            ifft = np.fft.irfft(
                prod, norm="ortho", n=len(data)
            )
            self.filtered_data[ifo] = RealData(ifft, index=data.index, ifo=ifo)

    def add_ftau_filter(self, f, tau, window="Tukey", alpha=0.2):
        """Apply rational filters to prefiltered data :attr:`Network.filtered_data` and store
        the filtered data in :attr:`Network.filtered_ftau_data`."""
        for ifo, data in self.filtered_data.items():
            data_in_freq = data.fft_data(window=window, alpha=alpha)
            freq = data.fft_freq
            filter_in_freq = Filterftau().total_filter(freq, f, tau)
            ifft = np.fft.irfft(
                filter_in_freq * data_in_freq, norm="ortho", n=len(data)
            )
            self.filtered_ftau_data[ifo] = RealData(ifft, index=data.index, ifo=ifo)

    def likelihood_vs_mass_spin(self, M_est, chi_est, **kwargs) -> float:
        """Compute likelihood for the given mass and spin.

        Parameters
        ----------
        M_est : float
            in solar mass, mass of rational filters
        chi_est : float
            dimensionless spin of rational filters
        cal_response : dict, optional
            {ifo: R(omega)} calibration response applied to the data
            (see :meth:`add_filter`).

        Returns
        -------
        The corresponding likelihood.
        """
        model_list = kwargs.pop("model_list")
        window = kwargs.get("window", 'Tukey')
        alpha = kwargs.get("alpha", 0.2)
        cal_response = kwargs.get("cal_response", None)
        self.add_filter(mass=M_est, chi=chi_est, model_list=model_list, window=window, alpha=alpha,
                        cal_response=cal_response)
        return self.compute_likelihood(apply_filter=True)

    def likelihood_vs_f_tau(self, f, tau) -> float:
        """Compute likelihood for the frequency and damping time.

        Parameters
        ----------
        f : float
            in rad/s
        tau : float
            in seconds

        Returns
        -------
        The corresponding likelihood.
        """
        self.add_ftau_filter(f=f, tau=tau)
        return self.compute_likelihood_ftau()

    def cached_add_filter(self, fft_freq_dict=None, fft_data_dict = None, **kwargs):
        """Apply rational filters to :attr:`Network.original_data` and store
        the filtered data in :attr:`Network.filtered_data`."""
        for ifo, data in self.original_data.items():
            data_in_freq = fft_data_dict[ifo] #data.fft_data
            freq = fft_freq_dict[ifo] #data.fft_freq
            filter_in_freq = cached_Filter(**kwargs).total_filter(freq)
            ifft = np.fft.irfft(
                filter_in_freq * data_in_freq, norm="ortho", n=len(data)
            )
            self.filtered_data[ifo] = RealData(ifft, index=data.index, ifo=ifo)

    def cached_likelihood_vs_mass_spin(self, M_est, chi_est, cached_omega, fft_freq_dict, fft_data_dict, **kwargs) -> float:
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
        self.cached_add_filter(mass=M_est, chi=chi_est, model_list=model_list, 
            cached_omega=cached_omega, fft_freq_dict = fft_freq_dict, fft_data_dict = fft_data_dict)
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
