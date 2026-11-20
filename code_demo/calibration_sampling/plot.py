"""Plotting helpers for the calibration-sampling demo.

Lives next to the demo notebook rather than in the package (it needs corner).
Use it from this folder with ``import plot``.

Every function draws in the paper style (serif fonts, the palette in
``_STYLE``) inside a ``matplotlib.rc_context``, so importing or calling this
module never changes your global matplotlib settings. Each function saves the
figure if ``save_path`` is given, closes it, and returns it (a returned figure
still displays in a notebook).
"""
import functools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qnm")


__all__ = [
    'plot_ACF',
    'plot_PSD',
    'plot_noise',
    'plot_data',
    'plot_likelihood_vs_mass_spin',
    'plot_corner',
    'plot_sampling_density_vs_mass_spin',
    'plot_cal_response',
]

import numpy as np
import matplotlib.pyplot as plt
from qnm_filter.utility import parallel_compute, find_credible_region
import corner
from scipy import stats
from qnm_filter import Calibration_ErrUnc

# ---------------------------------------------------------------------------
# Paper style: colours used by the plotting functions, and the rcParams they
# are drawn with.
# ---------------------------------------------------------------------------
_STYLE = {
    "primary": "deeppink",      # posterior / main accent
    "primary_dark": "#c2185b",  # MAP/central line, darker than the band fill
    "secondary": "#70888c",     # data / truth marker (teal)
    "tertiary": "blue",         # injected-error / reference line
    "prior": "grey",            # prior curves
    "inj": "green",             # injected marker (green +)
    "rec": "white",             # recovered marker face (white X, black edge)
    "cmap": "Spectral",
    "alpha": 0.6,               # posterior density shading
    "corner_label_size": 14,    # corner panels are small, so smaller text
    "corner_title_size": 12,
}

_PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    # Render $...$ math in serif too, so $M_f$, $\chi_f$, $\delta A$ match the text.
    "mathtext.fontset": "dejavuserif",
    "mathtext.rm": "serif",
    "font.size": 13,
    "axes.labelsize": 17,
    "axes.titlesize": 16,
    "figure.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
    "image.cmap": _STYLE["cmap"],
    "axes.prop_cycle": plt.cycler(color=[_STYLE["primary"], _STYLE["secondary"], "#e8800c"]),
}


def _paper_style(func):
    """Run a plotting function inside the paper-style rc_context."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with plt.rc_context(_PAPER_RC):
            return func(*args, **kwargs)
    return wrapper


def _finish(fig, save_path, dpi=200):
    """Save (if a path is given), close and return the figure."""
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


@_paper_style
def plot_ACF(fit, save_path=None, ifos=None, figsize=None, normalize=True):
    """Plot the autocorrelation function for each IFO.

    Parameters
    ----------
    fit : Network
        Network object with a populated ``acfs`` dict.
    save_path : str, optional
        File path to save the figure. If None the figure is returned but not saved.
    ifos : list of str, optional
        Which IFOs to plot. Defaults to all keys in ``fit.acfs``.
    figsize : tuple, optional
        Figure size ``(width, height)``.
    normalize : bool
        If True, divide each ACF by its zero-lag value before plotting.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if not fit.acfs:
        raise ValueError("fit.acfs is empty – call fit.compute_acfs() first.")
    ifo_list = ifos if ifos is not None else sorted(fit.acfs.keys())
    n = len(ifo_list)
    if figsize is None:
        figsize = (8, 3 * n)
    fig, axes = plt.subplots(n, 1, figsize=figsize, squeeze=False)
    for ax, ifo in zip(axes[:, 0], ifo_list):
        acf = fit.acfs[ifo]
        lags = acf.index.values
        vals = acf.values
        if normalize:
            vals = vals / vals[0]
        ax.plot(lags, vals, lw=0.8)
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.set_xlabel('Lag (s)')
        ax.set_ylabel('ACF' + (' (normalised)' if normalize else ''))
        ax.set_title(ifo)
        ax.set_xlim(lags[0], lags[-1])
    fig.tight_layout()
    return _finish(fig, save_path)


@_paper_style
def plot_PSD(fit, save_path=None, ifos=None, figsize=None, asd=True, f_min=None, f_max=None):
    """Plot the power (or amplitude) spectral density for each IFO.

    Uses ``fit.raw_psds`` if available, otherwise reconstructs from ``fit.acfs``
    via an rFFT of the ACF values.

    Parameters
    ----------
    fit : Network
        Network object with ``raw_psds`` or ``acfs`` populated.
    save_path : str, optional
        File path to save the figure.
    ifos : list of str, optional
        Which IFOs to plot. Defaults to all available IFOs.
    figsize : tuple, optional
        Figure size ``(width, height)``.
    asd : bool
        If True plot ASD (sqrt of PSD); if False plot PSD.
    f_min, f_max : float, optional
        Frequency limits in Hz for the x-axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    use_raw = hasattr(fit, 'raw_psds') and fit.raw_psds
    if not use_raw and not fit.acfs:
        raise ValueError("Neither fit.raw_psds nor fit.acfs is populated.")

    ifo_list = ifos if ifos is not None else sorted(
        fit.raw_psds.keys() if use_raw else fit.acfs.keys()
    )
    n = len(ifo_list)
    if figsize is None:
        figsize = (8, 8)
    fig, axes = plt.subplots(n, 1, figsize=figsize, squeeze=False)

    for ax, ifo in zip(axes[:, 0], ifo_list):
        if use_raw and ifo in fit.raw_psds:
            psd_obj = fit.raw_psds[ifo]
            freqs = psd_obj.index.values   # Hz
            psd_vals = psd_obj.values
        else:
            acf = fit.acfs[ifo]
            dt = acf.time_interval
            N = len(acf)
            fs = 1.0 / dt
            freqs = np.fft.rfftfreq(N, d=dt)           # Hz
            psd_vals = (2.0 / fs) * np.abs(np.fft.rfft(acf.values))

        y = np.sqrt(psd_vals) if asd else psd_vals
        ylabel = r'ASD (strain/$\sqrt{\mathrm{Hz}}$)' if asd else r'PSD (strain$^2$/Hz)'

        mask = np.ones(len(freqs), dtype=bool)
        if f_min is not None:
            mask &= freqs >= f_min
        if f_max is not None:
            mask &= freqs <= f_max

        ax.loglog(freqs[mask], y[mask], lw=0.8)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(ylabel)
        ax.set_title(ifo)
        if f_min is not None or f_max is not None:
            ax.set_xlim(
                f_min if f_min is not None else freqs[mask][0],
                f_max if f_max is not None else freqs[mask][-1],
            )

    fig.tight_layout()
    return _finish(fig, save_path)


@_paper_style
def plot_noise(fit, save_path=None, ifos=None, figsize=None,
               t_start=None, t_end=None):
    """Plot a noise time-series stored on *fit*.

    Parameters
    ----------
    fit : Network
        Network object that has an attribute ``noise_attr`` (a dict of RealData).
    noise_attr : str
        Name of the attribute to read, e.g. ``'pure_noise'``.
    save_path : str, optional
        File path to save the figure.
    ifos : list of str, optional
        Which IFOs to plot.
    figsize : tuple, optional
        Figure size ``(width, height)``.
    t_start, t_end : float, optional
        GPS time limits for the x-axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    noise_dict = fit.pure_noise
    # if noise_dict is None:
    #     raise AttributeError(f"fit has no attribute '{noise_attr}'.")
    ifo_list = ifos if ifos is not None else sorted(noise_dict.keys())
    n = len(ifo_list)
    if figsize is None:
        figsize = (10, 3 * n)
    fig, axes = plt.subplots(n, 1, figsize=figsize, squeeze=False)
    for ax, ifo in zip(axes[:, 0], ifo_list):
        d = noise_dict[ifo]
        t = d.index.values
        v = d.values
        mask = np.ones(len(t), dtype=bool)
        if t_start is not None:
            mask &= t >= t_start
        if t_end is not None:
            mask &= t <= t_end
        ax.plot(t[mask], v[mask], lw=0.6)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strain')
        ax.set_title(f'{ifo} – Noise')
    fig.tight_layout()
    return _finish(fig, save_path)


@_paper_style
def plot_data(fit, save_path=None, ifos=None, figsize=None, plot_filtered=True,
              t_start=None, t_end=None):
    """Plot the original (conditioned) data and optionally the filtered residual.

    Parameters
    ----------
    fit : Network
        Network object with ``original_data`` (and optionally ``filtered_data``) populated.
    save_path : str, optional
        File path to save the figure.
    ifos : list of str, optional
        Which IFOs to plot.
    figsize : tuple, optional
        Figure size ``(width, height)``.
    plot_filtered : bool
        If True and ``fit.filtered_data`` is non-empty, overlay the filtered data.
    t_start, t_end : float, optional
        GPS or relative time limits for the x-axis (same units as ``data.index``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if not fit.original_data:
        raise ValueError("fit.original_data is empty.")
    has_filtered = plot_filtered and bool(fit.filtered_data)
    ifo_list = ifos if ifos is not None else sorted(fit.original_data.keys())
    n = len(ifo_list)
    if figsize is None:
        figsize = (10, 3 * n)
    fig, axes = plt.subplots(n, 1, figsize=figsize, squeeze=False)
    for ax, ifo in zip(axes[:, 0], ifo_list):
        orig = fit.original_data[ifo]
        t = orig.index.values
        v = orig.values

        mask = np.ones(len(t), dtype=bool)
        if t_start is not None:
            mask &= t >= t_start
        if t_end is not None:
            mask &= t <= t_end

        ax.plot(t[mask], v[mask], lw=0.7, label='data', color=_STYLE['secondary'])

        if has_filtered and ifo in fit.filtered_data:
            filt = fit.filtered_data[ifo]
            tf = filt.index.values
            vf = filt.values
            fmask = np.ones(len(tf), dtype=bool)
            if t_start is not None:
                fmask &= tf >= t_start
            if t_end is not None:
                fmask &= tf <= t_end
            ax.plot(tf[fmask], vf[fmask], lw=0.7, label='filtered', color=_STYLE['primary'], alpha=0.85)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strain')
        ax.set_title(ifo)
        ax.legend()

    fig.tight_layout()
    return _finish(fig, save_path)


@_paper_style
def plot_likelihood_vs_mass_spin(fit, M, chi, M_min, M_max, chi_min, chi_max, n_points, save_path=None,
                                 cal_response=None, num_cpu=4, acfs_update=False, likelihood_data=None,
                                 massspace=None, chispace=None, truth_label='Injected', figsize=(6.1, 7),
                                 **data_input):
    """Plot the log-likelihood over a (mass, spin) grid.

    Parameters
    ----------
    fit : Network
        Network with data and noise model prepared.
    M, chi : float
        Injected (or reference) mass [solar masses] and spin, drawn as a "+".
    M_min, M_max, chi_min, chi_max : float
        Grid limits.
    n_points : int
        Number of grid points in each dimension.
    save_path : str, optional
        Where to save the figure.
    cal_response : dict, optional
        {ifo: R(omega)} calibration response applied to the data.
    num_cpu : int
        Parallel workers for the grid.
    likelihood_data, massspace, chispace : ndarray, optional
        A precomputed grid, to skip recomputing it.
    figsize : tuple
        Figure size.
    **data_input :
        Passed to ``fit.likelihood_vs_mass_spin`` (e.g. ``model_list``).

    Returns
    -------
    fig, massspace, chispace, likelihood_data
    """
    if massspace is None:
        massspace = np.linspace(M_min, M_max, n_points)
    if chispace is None:
        chispace = np.linspace(chi_min, chi_max, n_points)
    if likelihood_data is None:
        likelihood_data, _ = parallel_compute(fit, massspace, chispace, num_cpu=num_cpu,
                                              acfs_update=acfs_update, cal_response=cal_response,
                                              **data_input)

    credible_region = find_credible_region(likelihood_data)
    mass_grid, chi_grid = np.meshgrid(massspace, chispace)

    fig, ax = plt.subplots(figsize=figsize)
    contours = ax.contourf(mass_grid, chi_grid, likelihood_data, 20, cmap=_STYLE['cmap'],
                           origin='lower', alpha=_STYLE['alpha'])
    ax.contour(mass_grid, chi_grid, likelihood_data, [credible_region], colors='red',
               linestyles='--')
    ax.scatter(x=M, y=chi, s=270, marker='+', c='black', linewidths=5)
    ax.scatter(x=M, y=chi, s=255, marker='+', c=_STYLE['inj'], linewidths=4, label=truth_label)

    idx = np.unravel_index(np.nanargmax(likelihood_data), likelihood_data.shape)
    ax.scatter(mass_grid[idx], chi_grid[idx], s=120, marker='X', c=_STYLE['rec'],
               edgecolor='black', zorder=10, label='Recovered')
    ax.set_xlabel(r'Final Mass ($M_\mathrm{f}/M_\odot$)')
    ax.set_ylabel(r'Final Spin ($\chi_\mathrm{f}$)')
    ax.legend()

    cbar = fig.colorbar(contours, orientation="horizontal", pad=0.15, format='%3.2f')
    cbar.set_label(r'Log likelihood')
    cbar.set_ticks(np.linspace(np.min(likelihood_data), np.max(likelihood_data), 5))

    return _finish(fig, save_path), massspace, chispace, likelihood_data


def _corner_defaults(ndim_plot, labels, truths, figsize=None):
    """Shared corner.corner styling for the calibration corner plots."""
    # Auto-scale figsize based on number of parameters plotted
    if figsize is None:
        side = max(8, ndim_plot * 1.5)
        figsize = (side, side)
    return {
        'labels': labels,
        'figsize': figsize,
        'show_titles': True,
        'title_kwargs': {'fontsize': _STYLE['corner_title_size']},
        'label_kwargs': {'fontsize': _STYLE['corner_label_size']},
        'smooth': 1.0,
        'smooth1d': 1.0,
        'bins': 30,
        'hist_bin_factor': 2,
        'color': _STYLE['primary'],
        'truth_color': _STYLE['secondary'],
        'max_n_ticks': 4,
        'truths': truths,
        'data_kwargs': {'alpha': _STYLE['alpha']},
    }


def _strip_title_labels(fig, ndim_plot, title_kwargs):
    """Drop the "label = " prefix from each panel's title, leaving just the
    median/error text (corner.corner always prepends the axis label)."""
    axes_arr = np.array(fig.axes).reshape((ndim_plot, ndim_plot))
    for i in range(ndim_plot):
        ax = axes_arr[i, i]
        t = ax.get_title()
        if '=' in t:
            ax.set_title(t.split('=', 1)[1].strip(), **title_kwargs)


@_paper_style
def plot_corner(flat_samples, fctrl_dict, save_path, labels=None,
                truths=None, figsize=None, model=None,
                sample_mchi=True, sample_amp=True, sample_phase=True,
                pin_boundaries=True, truth_txt=None, show_suptitle=True,
                show_title_labels=True, **corner_kwargs):
    """
    Create a corner plot from MCMC flat samples.

    Parameters
    ----------
    flat_samples : ndarray
        Flattened MCMC samples, shape (nsamples, ndim), laid out as
        :class:`qnm_filter.CalibrationModel` lays out theta:
        ``[M, chi, (ifo1: delta_amp[K], delta_phase[K]), (ifo2: ...), ...]``,
        in ``fctrl_dict`` order, with any block that was not sampled left out.
    fctrl_dict : dict
        {ifo: f_ctrl_array} Control frequencies for each IFO.
    save_path : str
        Path to save the corner plot.
    labels : list, optional
        Custom labels for parameters. If None, generates automatic labels.
    truths : dict or list, optional
        True values to plot. If dict, may contain 'M', 'chi', and scalar
        'cal_amp' / 'cal_phase' (used for every control point when truth_txt is
        None). If list, must match the column order.
    figsize : tuple, optional
        Figure size. Default scales with the number of parameters.
    model : CalibrationModel, optional
        If given, sample_mchi / sample_amp / sample_phase / pin_boundaries are
        taken from it (recommended: guarantees the layout matches the sampler).
    sample_mchi, sample_amp, sample_phase : bool
        Which blocks are present in flat_samples. Default: all (real-data fit).
    pin_boundaries : bool
        If True (default, as in the real-data fits), the first and last control
        points are fixed at 0 and only the K-2 interior points are in the chain.
        Labels use the index in the full control-point array.
    truth_txt : str, dict or None, optional
        Injected calibration error file (columns: freq, mag, rad, ...), shared by
        all detectors, or ``{ifo: path}`` for a per-detector injection.
        Per-control-point truths are interpolated at each control frequency:
        delta_amp_truth = mag(f_ctrl) - 1, delta_phase_truth = rad(f_ctrl).
        Overrides scalar 'cal_amp'/'cal_phase' from truths dict.
    **corner_kwargs : dict
        Additional keyword arguments to pass to corner.corner().

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if model is not None:
        sample_mchi, sample_amp = model.sample_mchi, model.sample_amp
        sample_phase, pin_boundaries = model.sample_phase, model.pin_boundaries

    ifo_names = list(fctrl_dict)  # same order as CalibrationModel
    label_offset = 1 if pin_boundaries else 0

    # Per-control-point truths from an injected calibration error file:
    # delta_amp = mag - 1 (fractional amplitude), delta_phase = rad (radians).
    amp_truths, phase_truths = {}, {}
    if truth_txt is not None:
        truth_paths = (truth_txt if isinstance(truth_txt, dict)
                       else {ifo: truth_txt for ifo in ifo_names})
        for ifo in ifo_names:
            freq_t, mag_t, rad_t = np.loadtxt(truth_paths[ifo]).T[:3]
            f_ctrl = fctrl_dict[ifo]
            f_eval = f_ctrl[1:-1] if pin_boundaries else f_ctrl
            amp_truths[ifo] = np.interp(f_eval, freq_t, mag_t - 1.0)
            phase_truths[ifo] = np.interp(f_eval, freq_t, rad_t)

    def _truth(table, key, ifo, k):
        if table:
            return float(table[ifo][k])
        return truths.get(key, 0.0) if isinstance(truths, dict) else None

    labels_list, truths_list = [], []
    if sample_mchi:
        labels_list += [r'$M_\mathrm{f}$ $(M_\odot)$', r'$\chi_\mathrm{f}$']
        if isinstance(truths, dict):
            truths_list += [truths.get('M'), truths.get('chi')]
        else:
            truths_list += [None, None]
    for ifo in ifo_names:
        k_free = len(fctrl_dict[ifo]) - 2 if pin_boundaries else len(fctrl_dict[ifo])
        if sample_amp:
            for k in range(k_free):
                labels_list.append(rf'{ifo} $\delta A_{{{label_offset + k}}}$')
                truths_list.append(_truth(amp_truths, 'cal_amp', ifo, k))
        if sample_phase:
            for k in range(k_free):
                labels_list.append(rf'{ifo} $\delta\phi_{{{label_offset + k}}}$')
                truths_list.append(_truth(phase_truths, 'cal_phase', ifo, k))

    if len(labels_list) != flat_samples.shape[1]:
        raise ValueError(
            f"flat_samples has {flat_samples.shape[1]} columns but the layout "
            f"implies {len(labels_list)}; check sample_*/pin_boundaries or pass model=")

    if labels is None:
        labels = labels_list
    if isinstance(truths, (list, tuple, np.ndarray)):
        truths_array = list(truths)
    else:
        truths_array = truths_list if any(t is not None for t in truths_list) else None

    default_kwargs = _corner_defaults(flat_samples.shape[1], labels, truths_array, figsize)
    default_kwargs.update(corner_kwargs)

    fig = corner.corner(flat_samples, **default_kwargs)

    if not show_title_labels:
        _strip_title_labels(fig, flat_samples.shape[1], default_kwargs.get('title_kwargs', {}))

    if show_suptitle:
        if sample_mchi:
            title = 'Parameter Posteriors'
        elif sample_amp and sample_phase:
            title = 'Calibration Parameter Posteriors'
        elif sample_amp:
            title = 'Amplitude Calibration Parameter Posteriors'
        else:
            title = 'Phase Calibration Parameter Posteriors'
        fig.suptitle(title, y=0.995)

    return _finish(fig, save_path)

@_paper_style
def plot_sampling_density_vs_mass_spin(flat_samples, M, chi, M_min, M_max, chi_min, chi_max, CI_level,
                                       save_path=None, log_density=False, log_floor_CI=99.90,
                                       truth_label='Injected', figsize=(6.1, 7)):
    """Plot the KDE sampling density over mass-spin.

    Parameters
    ----------
    flat_samples : ndarray
        Chain whose first two columns are M and chi.
    M, chi : float
        Injected (or reference) values, drawn as a "+".
    M_min, M_max, chi_min, chi_max : float
        Plot limits.
    CI_level : float
        Credible level [%] of the red contour.
    save_path : str, optional
        Where to save the figure.
    log_density : bool
        If True, colour the density on a logarithmic scale. Unlike a linear
        density (whose exponential peak-to-tail ratio eats the whole colour
        range), log scaling spreads the colourmap across the body of the
        posterior, mirroring the readable look of the ln-likelihood plot.
        Default False (linear).
    log_floor_CI : float
        Colour floor for ``log_density``, as a credible-mass percentage: the
        colourmap spans from the density threshold enclosing this fraction of
        the probability up to the peak. This bounds the dynamic range to the
        region that carries probability, so the numerically-tiny tails don't
        wash out the structure. Default 99.0 (i.e. floor at the 99% contour).
    """
    mcmc_M = flat_samples[:,0]; mcmc_chi = flat_samples[:,1]

    kde = stats.gaussian_kde(np.vstack([mcmc_M, mcmc_chi]))

    X, Y = np.mgrid[M_min:M_max:200j, chi_min:chi_max:200j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kde(positions), X.shape)

    Z_flat = Z.ravel()
    Z_sort = np.sort(Z_flat)[::-1]
    cdf = np.cumsum(Z_sort)
    cdf /= cdf[-1]
    level = Z_sort[cdf > CI_level/100][0]

    imax, jmax = np.unravel_index(np.argmax(Z), Z.shape)


    fig, ax = plt.subplots(figsize=figsize)
    if log_density:
        from matplotlib.colors import LogNorm
        vmax = Z.max()
        # Floor the colour range at the density threshold enclosing
        # log_floor_CI% of the probability, so the tails don't set the scale.
        vmin = Z_sort[cdf > log_floor_CI / 100][0]
        levels = np.logspace(np.log10(vmin), np.log10(vmax), 20)
        # Clamp to the floor so below-floor cells fill with the darkest colour
        # (no transparent gaps) without a colourbar clipping arrow.
        Zc = np.clip(Z, vmin, vmax)
        contours = ax.contourf(X, Y, Zc, levels, cmap=_STYLE['cmap'],
                           norm=LogNorm(vmin=vmin, vmax=vmax),
                           origin='lower', alpha=_STYLE['alpha'])
    else:
        contours = ax.contourf(X, Y, Z, 20, cmap=_STYLE['cmap'],
                               origin='lower', alpha=_STYLE['alpha'])
    ax.contour(X, Y, Z, levels=[level], colors="red", linewidths=2, linestyles="--")
    ax.scatter(M, chi, marker='+', s=255,
               c=_STYLE['inj'], linewidths=4, label=truth_label, zorder=10)
    ax.scatter(X[imax, jmax], Y[imax, jmax], s=120, marker='X', c=_STYLE['rec'], edgecolor='black', zorder=10, label='Recovered')
    ax.set_xlabel(r'Final Mass ($M_\mathrm{f}/M_\odot$)')
    ax.set_ylabel(r'Final Spin ($\chi_\mathrm{f}$)')
    ax.legend()
    cbar = fig.colorbar(contours, orientation="horizontal", pad=0.15, format='%3.2f')
    cbar.set_label(r'MCMC posterior density')
    if not log_density:
        cbar.set_ticks(np.linspace(np.min(Z), np.max(Z), 5))

    return _finish(fig, save_path), X[imax, jmax], Y[imax, jmax]
    
def compute_M_and_pow2_lt_M(n: int):
    MAX_ELEMS = 1_000_000_000  # = 16_000_000_000 bytes / 16 bytes per complex128
    M = MAX_ELEMS // n
    if M <= 1:
        pow2_lt_M = 0
    else:
        pow2_lt_M = 1 << ((M - 1).bit_length() - 1)
    return M, pow2_lt_M

def _cal_response_bands(Rf, delta_amp, delta_phase, filter_f, tag="", verbose=False):
    """Percentile bands of |R(f)| and arg R(f) over the MCMC samples.

    Evaluates the calibration response for every sample at every frequency in
    memory-safe chunks, then reduces over samples.

    Returns
    -------
    mag_bands, deg_bands : dict
        Each keyed by percentile (5, 16, 50, 84, 95) -> array over ``filter_f``.
    """
    n_samp_used = delta_amp.shape[0]
    n_freq = len(filter_f)
    pcts = (5, 16, 50, 84, 95)
    mag_bands = {p: np.empty(n_freq, dtype=float) for p in pcts}
    deg_bands = {p: np.empty(n_freq, dtype=float) for p in pcts}

    _, CHUNK = compute_M_and_pow2_lt_M(n_samp_used)
    if verbose:
        print(f'{tag}chunk={CHUNK}, sample_number: {n_samp_used}')

    for s in range(0, n_freq, CHUNK):
        e = min(s + CHUNK, n_freq)
        omega_chunk = 2 * np.pi * filter_f[s:e]

        vals = np.empty((n_samp_used, e - s), dtype=complex)
        for i, (g_i, r_i) in enumerate(zip(delta_amp, delta_phase)):
            vals[i, :] = Rf(omega_chunk, g_i, r_i)
            if verbose and np.mod(i, max(1, int(n_samp_used / 1000))) == 0:
                print(f"{tag}({s}, {i})")

        _mag = np.percentile(np.abs(vals), pcts, axis=0)
        _deg = np.percentile(np.angle(vals, deg=True), pcts, axis=0)
        for j, p in enumerate(pcts):
            mag_bands[p][s:e] = _mag[j]
            deg_bands[p][s:e] = _deg[j]

    return mag_bands, deg_bands


@_paper_style
def plot_cal_response(
        flat_samples, fctrl_dict,
        prior_txt, save_path,
        xscale="linear", p_fmin=100, p_fmax=500, p1_ymin=None, p1_ymax=None, p2_ymin=None, p2_ymax=None,
        truth=None, sample_mchi=True, pin_boundaries=True,
        use_logf=False, spline_kind="quadratic", max_samples=2000,
        truth_label=None, map_index=None, truth_band=True, amp_pct=False, verbose=False):
    """Plot the recovered calibration response against prior (and truth).

    The number of detectors is taken from ``fctrl_dict``: one amplitude panel
    and one phase panel are stacked per detector, so a two-detector run yields
    four panels sharing the frequency axis.

    Parameters
    ----------
    flat_samples : ndarray
        Flattened MCMC chain. Calibration blocks are laid out per detector in
        ``fctrl_dict`` order, matching ``CalibrationModel``:
        ``[M, chi, (ifo1: delta_amp[K], delta_phase[K]), (ifo2: delta_amp[K], delta_phase[K]), ...]``
        (the leading ``M, chi`` are absent when ``sample_mchi=False``).
    fctrl_dict : dict
        ``{ifo: f_ctrl_array}``. Its keys set the detectors that are plotted.
    prior_txt : str, Calibration_ErrUnc or dict
        Calibration prior: a file (columns freq, mag, rad, mag16, rad16, mag84,
        rad84) or a :class:`Calibration_ErrUnc`, shared by all detectors, or
        ``{ifo: ...}`` per detector.
    truth : str, array, dict or None
        Injected error: a file or an array whose rows are (freq, mag, rad) and
        optionally (mag16, rad16, mag84, rad84), shared or ``{ifo: ...}``.
    verbose : bool
        Print progress while evaluating the response for each sample.
    sample_mchi : bool
        Whether the chain starts with M, chi (default True).
    pin_boundaries, use_logf, spline_kind :
        Must match the fit (defaults are the real-data settings). The plotted
        response always includes the prior median (``include_mean=True``) so it
        can be compared with the prior band.
    """
    ifo_names = list(fctrl_dict)
    n_ifo = len(ifo_names)

    # A single path is shared by every detector; a dict gives per-detector files.
    if not isinstance(prior_txt, dict):
        prior_txt = {ifo: prior_txt for ifo in ifo_names}
    if truth is not None and not isinstance(truth, dict):
        truth = {ifo: truth for ifo in ifo_names}

    filter_f = np.arange(1, 4096+1, 1.)
    filter_f = np.r_[1e-6, filter_f]

    data = flat_samples

    # Thin samples before the (expensive) per-sample response evaluation.
    # The response is evaluated for every sample at every frequency, so a
    # 32000-step run is unusably slow; subsample evenly to at most
    # ``max_samples`` rows (set max_samples=None to disable).
    if max_samples is not None and len(data) > max_samples:
        idx = np.linspace(0, len(data) - 1, max_samples).astype(int)
        data = data[idx]
        if verbose:
            print(f"Thinned {len(flat_samples)} -> {len(data)} samples for cal-response plot")

    # Smaller, paper-column-sized figure (was 10x6*n_ifo): that made the
    # figure so large relative to a printed column that it had to be shrunk
    # far more than plot_mchi_combined's 6.4x6.0, making the same absolute
    # font sizes look tiny once placed in the paper.
    fig, axes = plt.subplots(2 * n_ifo, 1, figsize=(7, 5 * n_ifo),
                             squeeze=False, sharex=True)
    axes = axes[:, 0]

    _posterior = _STYLE['primary']
    _posterior_line = _STYLE['primary_dark']
    _prior = _STYLE['prior']
    _inj = _STYLE['tertiary']

    # Walk the per-detector calibration blocks in fctrl_dict order. `pos` tracks
    # the column offset into the chain; each detector contributes delta_amp then
    # delta_phase, so the stride is 2*k_free.
    pos = 2 if sample_mchi else 0
    for i, ifo in enumerate(ifo_names):
        f_ctrl_hz = fctrl_dict[ifo]
        k_free = len(f_ctrl_hz) - 2 if pin_boundaries else len(f_ctrl_hz)

        pos_start = pos
        delta_amp = data[:, pos:pos + k_free]
        delta_phase = data[:, pos + k_free:pos + 2 * k_free]
        pos += 2 * k_free

        if isinstance(prior_txt[ifo], Calibration_ErrUnc):
            cal_obj = prior_txt[ifo]
        else:
            freq, mag, rad, mag16, rad16, mag84, rad84 = np.loadtxt(prior_txt[ifo]).T
            cal_obj = Calibration_ErrUnc(
                freq, mag, rad, mag16, mag84, rad16, rad84, ifo=ifo,
            )
        freq, mag, rad = cal_obj.freq, cal_obj.mag, cal_obj.radian
        mag16, mag84, rad16, rad84 = cal_obj.mag16, cal_obj.mag84, cal_obj.radian16, cal_obj.radian84
        Rf = cal_obj.make_spline_response(
            f_ctrl_hz, use_logf=use_logf, include_mean=True,
            pin_boundaries=pin_boundaries, spline_kind=spline_kind)

        mag_b, deg_b = _cal_response_bands(Rf, delta_amp, delta_phase, filter_f,
                                           tag=f"[{ifo}] ", verbose=verbose)

        # Amplitude error as a percentage deviation from unity response
        # (centred at 0%) rather than the raw response magnitude (centred
        # at 1).
        _amp = (lambda m: (np.asarray(m) - 1.0) * 100.0) if amp_pct else (lambda m: m)

        ax1, ax2 = axes[2 * i], axes[2 * i + 1]

        # Central line: the MAP response (evaluated at the highest-posterior
        # sample) when map_index is given -- for consistency with the mass-spin
        # plots' MAP marker -- otherwise the pointwise median. Credible bands are
        # percentiles either way.
        if map_index is not None:
            g_map = flat_samples[map_index, pos_start:pos_start + k_free]
            r_map = flat_samples[map_index, pos_start + k_free:pos_start + 2 * k_free]
            R_map = Rf(2 * np.pi * filter_f, g_map, r_map)
            mag_c, deg_c = np.abs(R_map), np.angle(R_map, deg=True)
        else:
            mag_c, deg_c = mag_b[50], deg_b[50]

        ax1.plot(filter_f, _amp(mag_c), color=_posterior_line, label="Posterior")
        ax1.fill_between(filter_f, _amp(mag_b[16]), _amp(mag_b[84]), color=_posterior, alpha=0.2)
        ax1.fill_between(filter_f, _amp(mag_b[5]), _amp(mag_b[95]), color=_posterior, alpha=0.2)
        ax1.plot(freq, _amp(mag), color=_prior, linestyle="--", label="Prior")
        ax1.fill_between(freq, _amp(mag16), _amp(mag84), color=_prior, alpha=0.2)

        ax2.plot(filter_f, deg_c, color=_posterior_line, label="Posterior")
        ax2.fill_between(filter_f, deg_b[16], deg_b[84], color=_posterior, alpha=0.2)
        ax2.fill_between(filter_f, deg_b[5], deg_b[95], color=_posterior, alpha=0.2)
        ax2.plot(freq, rad*180/np.pi, color=_prior, linestyle="--", label="Prior")
        ax2.fill_between(freq, rad16*180/np.pi, rad84*180/np.pi,
                         color=_prior, alpha=0.2)

        if truth is not None:
            tcols = (np.loadtxt(truth[ifo]).T if isinstance(truth[ifo], str)
                     else np.asarray(truth[ifo], dtype=float))
            freq_true, mag_true, rad_true = tcols[0], tcols[1], tcols[2]
            # A 7-column file carries 16th/84th percentiles -> draw it as a band
            # (the real calibration envelope). A 3-column file is just a median
            # injected-error line (legacy behaviour).
            has_band = tcols.shape[0] >= 7
            # Caller-supplied label wins; else a neutral default (the old
            # hardcoded "C00 envelope" was misleading for non-C00 overlays).
            if truth_label is not None:
                tlabel = truth_label
            else:
                tlabel = "Reference" if has_band else "Injected Error"
            ax1.plot(freq_true, _amp(mag_true), color=_inj, linestyle="--", label=tlabel)
            ax2.plot(freq_true, rad_true*180/np.pi, color=_inj, linestyle="--",
                     label=tlabel)
            # Shade the 16/84 band only when the truth carries an uncertainty
            # (a real envelope) AND truth_band is on. For an EXACT known error
            # (e.g. an NR injection) set truth_band=False to draw the line alone.
            if has_band and truth_band:
                mag16, rad16, mag84, rad84 = tcols[3], tcols[4], tcols[5], tcols[6]
                ax1.fill_between(freq_true, _amp(mag16), _amp(mag84), color=_inj, alpha=0.3)
                ax2.fill_between(freq_true, rad16*180/np.pi, rad84*180/np.pi,
                                 color=_inj, alpha=0.3)

        ax1.set_xlim(p_fmin, p_fmax)
        ax2.set_xlim(p_fmin, p_fmax)
        ax1.set_ylim(p1_ymin, p1_ymax)
        ax2.set_ylim(p2_ymin, p2_ymax)
        ax1.legend(loc='lower right')
        ax2.legend(loc='upper right')
        amp_label = 'Amplitude [%]' if amp_pct else 'Amplitude'
        ax1.set_ylabel(f'{ifo} {amp_label}' if n_ifo > 1 else amp_label)
        ax2.set_ylabel(f'{ifo} phase [deg]' if n_ifo > 1 else 'Phase [deg]')
        ax1.grid()
        ax2.grid()
        ax1.set_xscale(xscale)
        ax2.set_xscale(xscale)

    axes[-1].set_xlabel('Frequency [Hz]')
    fig.tight_layout()
    return _finish(fig, save_path)
