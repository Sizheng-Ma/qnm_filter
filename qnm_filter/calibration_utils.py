
__all__ = [
    "Calibration_ErrUnc",
    "CalibrationModel",
    "CalibrationLogProb",
    "build_response_map",
    "compute_loglike_network",
]

import numpy as np
from scipy import interpolate

class Calibration_ErrUnc:
    """
    Handle complex calibration response and its (magnitude, phase) uncertainties.
    Provides:
      - mean complex response R_cal(f)
      - 1-sigma uncertainties as functions of frequency
      - builder for MCMC
    """

    def __init__(self, freq, mag, radian, mag16, mag84, radian16, radian84, ifo, f_edge=16384.0):
        # Input arrays (1D, vs frequency)
        #   freq      : frequency grid [Hz]
        #   mag       : median (or mean) amplitude response
        #   radian    : median (or mean) phase response [rad]
        #   mag16/84  : 16th / 84th percentile amplitude
        #   radian16/84: 16th / 84th percentile phase [rad]
        #   ifo       : detector name (e.g. 'H1', 'L1')
        #   f_edge    : (optional) max frequency used as edge for extrapolation
        

        self.freq = np.asarray(freq, float)
        self.mag = np.asarray(mag, float)
        self.mag16 = np.asarray(mag16, float)
        self.mag84 = np.asarray(mag84, float)
        self.radian = np.asarray(radian, float)
        self.radian16 = np.asarray(radian16, float)
        self.radian84 = np.asarray(radian84, float)
        self.ifo = ifo
        self.f_edge = f_edge

        f_grid_in = np.asarray(self.freq, float)
        mag_in = self.mag
        rad_in = self.radian

        re_in = np.real(mag_in * np.exp(1j * rad_in))
        im_in = np.imag(mag_in * np.exp(1j * rad_in))

        self._fr = interpolate.interp1d(
            f_grid_in, re_in, kind="cubic",
            bounds_error=False, fill_value=(1.0, 1.0), assume_sorted=True
        )
        self._fi = interpolate.interp1d(
            f_grid_in, im_in, kind="cubic",
            bounds_error=False, fill_value=(0.0, 0.0), assume_sorted=True
        )
        mag_sigma = self.mag_1sigma() 
        rad_sigma = self.radian_1sigma()             

        self._sig_mag = interpolate.interp1d(
            f_grid_in, mag_sigma, kind="linear",
            bounds_error=False, fill_value=(0.0, 0.0), assume_sorted=True
        )
        self._sig_rad = interpolate.interp1d(
            f_grid_in, rad_sigma, kind="linear",
            bounds_error=False, fill_value=(0.0, 0.0), assume_sorted=True
        )

    def mag_1sigma(self):
        """
        Symmetrize the (16th, 84th) percentile amplitude errors into 1-sigma.
        Uses:
            sigma^2 = 0.5 * [ (mag84 - mag)^2 + (mag - mag16)^2 ]
        """
        return np.sqrt(
            ((self.mag84 - self.mag) ** 2 + (self.mag - self.mag16) ** 2) * 0.5
        )
    
    def radian_1sigma(self):
        """
        Same as mag_1sigma(), but for phase [rad].
        """
        return np.sqrt(
            ((self.radian84 - self.radian) ** 2 + (self.radian - self.radian16) ** 2) * 0.5
        )
    
    def cal_response_mean(self, omega):
        """
        Return mean complex calibration response R_mean(omega) or R_median(omega).

        Parameters
        ----------
        omega : array_like
            Angular frequency [rad/s].

        Returns
        -------
        R_mean (R_median) : complex or ndarray of complex
            Interpolated mean response at |omega|.
        """
        f = np.abs(np.asarray(omega)) / (2 * np.pi)  # [Hz]
        return self._fr(f) + 1j * self._fi(f)

    def sigma_at(self, omega):
        """
        Evaluate 1-sigma uncertainties at given angular frequencies.

        Parameters
        ----------
        omega : array_like
            Angular frequency [rad/s].

        Returns
        -------
        (sigma_amp, sigma_phase_rad)
            sigma_amp        : 1-sigma on amplitude (absolute, not relative)
            sigma_phase_rad  : 1-sigma on phase [rad]
        """
        f = np.abs(np.asarray(omega)) / (2 * np.pi)
        return self._sig_mag(f), self._sig_rad(f)

   
    def make_spline_response(self, f_ctrl_hz, use_logf=False, include_mean=False, pin_boundaries=True, spline_kind="quadratic"):
        """
        Create a function that, given control-point offsets (delta_amp, delta_phase),
        returns a calibration response function R(omega).

        delta_amp is the fractional amplitude offset and delta_phase the phase
        offset [rad] at each control point; both are interpolated in frequency.

        Parameters
        ----------
        f_ctrl_hz : array_like
            Control point frequencies [Hz] where delta_amp, delta_phase are defined.
        use_logf : bool
            If True, interpolate delta_amp, delta_phase in log(f). If False, in linear f.
        include_mean : bool
            If True, multiply uncertainty factor by mean response.
            If False, return only the uncertainty factor R_unc.
        pin_boundaries : bool
            If True, the builder expects interior-only delta arrays (length K-2)
            and prepends/appends 0 at both ends before interpolating.

        Returns
        -------
        builder : callable
            builder(omega, delta_amp, delta_phase) -> R(omega)
        """
        f_ctrl_hz = np.asarray(f_ctrl_hz, float)
        xctrl = np.log(f_ctrl_hz) if use_logf else f_ctrl_hz

        def builder(omega, delta_amp, delta_phase):
            if pin_boundaries:
                delta_amp = np.concatenate([[0.0], delta_amp, [0.0]])
                delta_phase = np.concatenate([[0.0], delta_phase, [0.0]])
            f = np.abs(np.asarray(omega)) / (2*np.pi)
            f_safe = np.where(f > 0.0, f, 1e-6)
            x = np.log(f_safe) if use_logf else f_safe

            amp_spline = interpolate.interp1d(
                xctrl, np.asarray(delta_amp, float), kind=spline_kind,
                bounds_error=False, fill_value=(0.0, 0.0), assume_sorted=True
            )
            phase_spline = interpolate.interp1d(
                xctrl, np.asarray(delta_phase, float), kind=spline_kind,
                bounds_error=False, fill_value=(0.0, 0.0), assume_sorted=True
            )
            amp = np.clip(1.0 + amp_spline(x), 1e-6, None)
            R_unc = amp * np.exp(1j * phase_spline(x))

            return self.cal_response_mean(2*np.pi*f) * R_unc if include_mean else R_unc
        return builder


    def prior_sigma_on_ctrl(self, f_ctrl_hz):
        """
        Compute prior 1-sigma for control points at f_ctrl_hz.

        Returns
        -------
        (sigma_relative_amp, sigma_phase_rad)

        where:
            sigma_relative_amp = sigma_mag / |R_mean|
            sigma_phase_rad    = sigma_phase
        """
        omega_ctrl = 2 * np.pi * np.asarray(f_ctrl_hz, float)
        mag_sigma, rad_sigma =  self.sigma_at(omega_ctrl)
        mean_mag = np.abs(self.cal_response_mean(omega_ctrl))
        mag_sigma = mag_sigma / np.clip(mean_mag, 1e-12, None)
        return mag_sigma, rad_sigma

# --------------------------------------------------------------------------
# Response map and network likelihood
# --------------------------------------------------------------------------

def build_response_map(calib_dict, fctrl_dict, deltas, use_logf=False, include_mean=False, pin_boundaries=True, spline_kind="quadratic"):
    """
    Build a mapping {ifo: R_ifo(omega)} from the calibration control-point offsets.

    Parameters
    ----------
    calib_dict : dict
        {ifo: Calibration_ErrUnc instance}
    fctrl_dict : dict
        {ifo: f_ctrl_array}
    deltas : dict
        {ifo: (delta_amp_array, delta_phase_array)}
    use_logf, include_mean, pin_boundaries, spline_kind :
        passed to make_spline_response()

    Returns
    -------
    cal_map : dict
        {ifo: callable(omega) -> complex response}
    """
    cal_map = {}
    for ifo, cal_obj in calib_dict.items():
        f_ctrl = fctrl_dict[ifo]
        builder = cal_obj.make_spline_response(
            f_ctrl_hz=f_ctrl, use_logf=use_logf, include_mean=include_mean,
            pin_boundaries=pin_boundaries, spline_kind=spline_kind,
        )
        delta_amp, delta_phase = deltas[ifo]

        def make_fixed_response(builder, delta_amp, delta_phase):
            def response(omega):
                return builder(omega, delta_amp, delta_phase)
            return response

        cal_map[ifo] = make_fixed_response(builder, delta_amp, delta_phase)

    return cal_map

def compute_loglike_network(fit, M, chi, cal_response, model_list, window="Tukey", alpha=0.2, acfs_update=True, noise_name="pure_noise", sampling_rate=0.5):
    """
    Compute network log-likelihood given intrinsic parameters and calibration map.

    Parameters
    ----------
    fit : Network object
    M, chi : float
        Intrinsic parameters (e.g. total mass, spin).
    cal_response : dict
        {ifo: R_ifo(omega)} calibration response functions.
    model_list : list
        Waveform / detector model specification for fit.likelihood_vs_mass_spin().
    window, alpha : passed to likelihood routine.
    acfs_update : bool
        Whether the calibration response is also applied to the noise model.

        - True (use this for real data): applies calibration error to data and noise at every step.
        - False: applies calibration error to data only. Much faster, fine for white-noise
          tests and for quick evidence / Laplace estimates; not the same
          likelihood as the real-data runs.
    noise_name : str
        Attribute of ``fit`` holding the off-source noise used to re-estimate the
        PSD. Only used when acfs_update=True.
    sampling_rate : float
        Passed to ``fit.compute_acfs()``. Only used when acfs_update=True.

    Returns
    -------
    ll : float
        Log-likelihood value.
    """
    if acfs_update:
        fit.compute_acfs(noise_name, sampling_rate=sampling_rate, cal_response=cal_response, acfs_update=acfs_update)
        fit.compute_toeplitz()
    return fit.likelihood_vs_mass_spin(
        M, chi, model_list=model_list, window=window, alpha=alpha,
        cal_response=cal_response, acfs_update=acfs_update,
    )


class CalibrationModel:
    """
    Parameter layout, prior and walker initialisation for ringdown + calibration
    sampling.

    The default samples everything (M, chi, amplitude and phase calibration for
    every IFO), which is what you use on real data:

        theta = [M, chi, (H1: delta_amp[K], delta_phase[K]), (L1: delta_amp[K], delta_phase[K]), ...]

    Any block can be switched off and held at a fixed value instead (used for
    tests, e.g. calibration-only or amplitude-only recovery). Blocks that are off
    are simply dropped from theta; the ordering of the remaining blocks is
    unchanged.

    Parameters
    ----------
    calib_dict : dict
        {ifo: Calibration_ErrUnc instance}
    fctrl_dict : dict
        {ifo: f_ctrl_array} control-point frequencies [Hz].
    priors : dict, optional
        Must contain "M_min", "M_max", "chi_min", "chi_max" when sample_mchi=True.
    sample_mchi, sample_amp, sample_phase : bool
        Whether (M, chi), the amplitude control points delta_amp and the phase
        control points delta_phase are sampled. Default: all True.
    M_fixed, chi_fixed : float, optional
        Values of M and chi used when sample_mchi=False.
    delta_amp_fixed, delta_phase_fixed : float or dict
        Values of delta_amp / delta_phase used when that block is not sampled. A float is
        applied to every IFO and control point; a dict is {ifo: array} with one
        entry per free control point. Default 0 (nominal calibration).
    pin_boundaries : bool
        If True, the first and last control point of each IFO are fixed at 0 and
        only the K-2 interior points are free. Default True (as in the real-data fits).
    """

    def __init__(self, calib_dict, fctrl_dict, priors=None, *,
                 sample_mchi=True, sample_amp=True, sample_phase=True,
                 M_fixed=None, chi_fixed=None, delta_amp_fixed=0.0, delta_phase_fixed=0.0,
                 pin_boundaries=True):
        self.calib_dict = calib_dict
        self.fctrl_dict = fctrl_dict
        self.priors = priors
        self.sample_mchi = sample_mchi
        self.sample_amp = sample_amp
        self.sample_phase = sample_phase
        self.pin_boundaries = pin_boundaries

        if sample_mchi:
            missing = {"M_min", "M_max", "chi_min", "chi_max"} - set(priors or {})
            if missing:
                raise ValueError(f"priors is missing {sorted(missing)} (needed when sample_mchi=True)")
        elif M_fixed is None or chi_fixed is None:
            raise ValueError("M_fixed and chi_fixed are required when sample_mchi=False")
        self.M_fixed = M_fixed
        self.chi_fixed = chi_fixed

        # Free control points per IFO, and the prior sigmas on them (fixed, so
        # computed once here rather than at every likelihood call).
        self._K = {}
        self._sigma = {}
        for ifo, f_ctrl in fctrl_dict.items():
            self._K[ifo] = len(f_ctrl) - 2 if pin_boundaries else len(f_ctrl)
            f_free = f_ctrl[1:-1] if pin_boundaries else f_ctrl
            amp_sigma, phase_sigma = calib_dict[ifo].prior_sigma_on_ctrl(f_free)
            self._sigma[ifo] = (np.asarray(amp_sigma, float), np.asarray(phase_sigma, float))

        self.delta_amp_fixed = self._fixed_delta(delta_amp_fixed)
        self.delta_phase_fixed = self._fixed_delta(delta_phase_fixed)

    def _fixed_delta(self, value):
        if isinstance(value, dict):
            return {ifo: np.asarray(value[ifo], float) for ifo in self.fctrl_dict}
        return {ifo: np.full(K, float(value)) for ifo, K in self._K.items()}

    @property
    def ndim(self):
        n_cal = (int(self.sample_amp) + int(self.sample_phase)) * sum(self._K.values())
        return 2 * int(self.sample_mchi) + n_cal

    @property
    def param_names(self):
        """Labels for each entry of theta, in order (handy for corner plots)."""
        names = ["M", "chi"] if self.sample_mchi else []
        for ifo, K in self._K.items():
            if self.sample_amp:
                names += [f"{ifo}_delta_amp_{k}" for k in range(K)]
            if self.sample_phase:
                names += [f"{ifo}_delta_phase_{k}" for k in range(K)]
        return names

    def unpack(self, theta):
        """
        Split theta into (M, chi, deltas), filling in fixed values for blocks that
        are not sampled.

        Returns
        -------
        M, chi : float
        deltas : dict
            {ifo: (delta_amp_array, delta_phase_array)}
        """
        if self.sample_mchi:
            M, chi = theta[0], theta[1]
            pos = 2
        else:
            M, chi = self.M_fixed, self.chi_fixed
            pos = 0
        deltas = {}
        for ifo, K in self._K.items():
            if self.sample_amp:
                delta_amp = np.asarray(theta[pos:pos+K]); pos += K
            else:
                delta_amp = self.delta_amp_fixed[ifo]
            if self.sample_phase:
                delta_phase = np.asarray(theta[pos:pos+K]); pos += K
            else:
                delta_phase = self.delta_phase_fixed[ifo]
            deltas[ifo] = (delta_amp, delta_phase)
        return M, chi, deltas

    def log_prior(self, theta):
        """
        Uniform prior on (M, chi) within the bounds in ``priors``; Gaussian prior
        on each sampled calibration control point, with sigma from
        Calibration_ErrUnc.prior_sigma_on_ctrl. Returns -inf outside the bounds.
        """
        M, chi, deltas = self.unpack(theta)
        if self.sample_mchi:
            p = self.priors
            if not (p["M_min"] <= M <= p["M_max"] and p["chi_min"] <= chi <= p["chi_max"]):
                return -np.inf

        lp = 0.0
        for ifo in self._K:
            delta_amp, delta_phase = deltas[ifo]
            amp_sigma, phase_sigma = self._sigma[ifo]
            if self.sample_amp:
                lp += _gaussian_lnpdf(delta_amp, np.clip(amp_sigma, 1e-12, None))
            if self.sample_phase:
                lp += _gaussian_lnpdf(delta_phase, np.clip(phase_sigma, 1e-12, None))
        return lp

    def init_walkers(self, nwalkers=None, seed=42):
        """
        Initial walker positions: (M, chi) uniform over the prior box, calibration
        control points drawn from their Gaussian prior.

        Parameters
        ----------
        nwalkers : int, optional
            Defaults to 4 * ndim.
        seed : int

        Returns
        -------
        p0 : ndarray, shape (nwalkers, ndim)
        """
        if nwalkers is None:
            nwalkers = 4 * self.ndim
        rng = np.random.default_rng(seed)
        p0 = np.empty((nwalkers, self.ndim), float)

        pos = 0
        if self.sample_mchi:
            p0[:, 0] = rng.uniform(self.priors["M_min"], self.priors["M_max"], size=nwalkers)
            p0[:, 1] = rng.uniform(self.priors["chi_min"], self.priors["chi_max"], size=nwalkers)
            pos = 2
        for ifo, K in self._K.items():
            amp_sigma, phase_sigma = self._sigma[ifo]
            if self.sample_amp:
                p0[:, pos:pos+K] = rng.normal(0.0, amp_sigma, size=(nwalkers, K)); pos += K
            if self.sample_phase:
                p0[:, pos:pos+K] = rng.normal(0.0, phase_sigma, size=(nwalkers, K)); pos += K
        return p0


def _gaussian_lnpdf(x, sigma):
    return -0.5*np.sum((x/sigma)**2 + np.log(2*np.pi*sigma**2))


class CalibrationLogProb:
    """
    log_prob(theta) = log_prior + log_likelihood, for samplers such as emcee.

    A module-level class rather than a closure so it can be pickled and sent to
    multiprocessing workers.

    Parameters
    ----------
    model : CalibrationModel
    fit : Network
        Network with data loaded and the ACFs / Cholesky factors computed once
        before sampling (with acfs_update=True they are then recomputed at
        every call). As in the real-data fits::

            fit.psd_floor = (flow, 1.0)           # if the data were high-passed
            fit.compute_acfs("pure_noise", sampling_rate=0.5)
            fit.cholesky_decomposition(eps_frac=1e-6)
            fit.first_index()
    model_list : list
        Mode list passed to fit.likelihood_vs_mass_spin().
    window, alpha : passed to the likelihood.
    use_logf, include_mean, spline_kind : calibration response options, passed
        to Calibration_ErrUnc.make_spline_response().
    acfs_update, noise_name, sampling_rate : passed to compute_loglike_network().
        Use acfs_update=True for real data (calibration applied to both data and
        noise model); False is a faster approximation for tests.
    """

    def __init__(self, model, fit, model_list, window="Tukey", alpha=0.2,
                 use_logf=False, include_mean=False, spline_kind="quadratic",
                 acfs_update=True, noise_name="pure_noise", sampling_rate=0.5):
        self.model = model
        self.fit = fit
        self.model_list = model_list
        self.window = window
        self.alpha = alpha
        self.use_logf = use_logf
        self.include_mean = include_mean
        self.spline_kind = spline_kind
        self.acfs_update = acfs_update
        self.noise_name = noise_name
        self.sampling_rate = sampling_rate

    def __call__(self, theta):
        lp = self.model.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf

        M, chi, deltas = self.model.unpack(theta)
        cal_map = build_response_map(
            self.model.calib_dict, self.model.fctrl_dict, deltas,
            use_logf=self.use_logf, include_mean=self.include_mean,
            pin_boundaries=self.model.pin_boundaries, spline_kind=self.spline_kind,
        )
        ll = compute_loglike_network(
            self.fit, M, chi, cal_map, model_list=self.model_list,
            window=self.window, alpha=self.alpha,
            acfs_update=self.acfs_update, noise_name=self.noise_name,
            sampling_rate=self.sampling_rate,
        )
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll
