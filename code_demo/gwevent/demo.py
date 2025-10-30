#!/usr/bin/env python3

"""
Sample code to compare real GWOSC data (GW250114) with a NR surrogate
reconstruction for the original and a QNM-removed (filtered) versions 
using qnm_filter.

Dependencies:
    qnm_filter, h5py, matplotlib, numpy, scipy, gwpy, gwsurrogate
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import copy
import h5py
import numpy as np
import matplotlib.pyplot as pl
import scipy.linalg as sl

from gwpy.timeseries import TimeSeries
from scipy.interpolate import interp1d

import gwsurrogate
import qnm_filter

import os
import urllib.request


# ----------------------------- Plot/Style Config ----------------------------- #

pl.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "xtick.labelsize": "medium",
        "ytick.labelsize": "medium",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.labelsize": "large",
        "axes.titlesize": "large",
        "axes.grid": False,
        "grid.alpha": 0.73,
        "lines.markersize": 12,
        "legend.borderpad": 0.2,
        "legend.fancybox": True,
        "legend.fontsize": 12,
        "legend.framealpha": 0.7,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.2,
        "legend.loc": "best",
        "savefig.dpi": 150,
        "pdf.compression": 9,
    }
)

# ------------------------------- Paths/Constants ----------------------------- #

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

POSTERIOR_H5 = f"{DATA_DIR}/posterior_samples_NRSur7dq4.h5"  # user-provided path

# Local filenames for GWOSC frames (auto-downloaded if missing)
GWOSC_GWF = {
    "H1": f"{DATA_DIR}/H-H1_GWOSC_O4b3Disc_16KHZ_R1-1420877824-4096.gwf",
    "L1": f"{DATA_DIR}/L-L1_GWOSC_O4b3Disc_16KHZ_R1-1420877824-4096.gwf",
}

# Source URLs to fetch the frames if missing
GWOSC_URLS = {
    "H1": "https://gwosc.org/archive/data/O4b3Disc_16KHZ_R1/1420820480/H-H1_GWOSC_O4b3Disc_16KHZ_R1-1420877824-4096.gwf",
    "L1": "https://gwosc.org/archive/data/O4b3Disc_16KHZ_R1/1420820480/L-L1_GWOSC_O4b3Disc_16KHZ_R1-1420877824-4096.gwf",
}

GWOSC_CHANNEL = "{ifo}:GWOSC-16KHZ_R1_STRAIN"


# ----------------------------- Data Structures ------------------------------ #

@dataclass
class RemnantIMR:
    """Final remnant mass and spin from IMR PE (posterior)."""
    mass: float
    chi: float


@dataclass
class MaxLContext:
    """Container of frequently used event-level values at max-likelihood."""
    ra: float
    dec: float
    t_geo: float
    psi: float
    mass_unit: float
    IMR: RemnantIMR
    posterior: np.ndarray
    maxL_idx: int


# ----------------------------- Utility: Downloads ---------------------------- #

def _ensure_file(local_path: str, url: str) -> None:
    """
    Ensure a file exists at `local_path`; if not, download it from `url`.
    Retries are minimal; for robust workflows consider adding checksums.
    """
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return
    print(f"[info] Downloading {os.path.basename(local_path)} from GWOSC ...")
    try:
        urllib.request.urlretrieve(url, local_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download {url} -> {local_path}: {e}")

def ensure_gwosc_frames_present() -> None:
    """Check for local GWOSC .gwf files and download them if missing."""
    for ifo, path in GWOSC_GWF.items():
        _ensure_file(path, GWOSC_URLS[ifo])

def ensure_posterior_present() -> None:
    """Ensure the posterior_samples_NRSur7dq4.h5 file exists in ./data; download from Zenodo if missing."""
    if not (os.path.exists(POSTERIOR_H5) and os.path.getsize(POSTERIOR_H5) > 0):
        print("[info] Downloading posterior_samples_NRSur7dq4.h5 from Zenodo ...")
        try:
            urllib.request.urlretrieve(
                "https://zenodo.org/api/records/16877102/files/posterior_samples_NRSur7dq4.h5/content",
                POSTERIOR_H5
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download posterior samples: {e}")
        

# ----------------------------- Posterior Utilities -------------------------- #

def load_posterior_maxL(
    posterior_h5: str = POSTERIOR_H5,
    group: str = "bilby-NRSur7dq4_prod-reweighted",
    dataset: str = "posterior_samples",
) -> MaxLContext:
    """Load posterior samples and extract max-likelihood sample plus convenient fields."""
    ensure_posterior_present()
    with h5py.File(posterior_h5, "r") as f:
        posterior = f[group][dataset][...]
    maxL_idx = int(np.argmax(posterior["log_likelihood"]))

    # Remnant properties (prefer evolved keys; fall back to *_non_evolved)
    try:
        m_final = posterior["final_mass"][maxL_idx]
        chi_final = posterior["final_spin"][maxL_idx]
    except (KeyError, ValueError):
        m_final = posterior["final_mass_non_evolved"][maxL_idx]
        chi_final = posterior["final_spin_non_evolved"][maxL_idx]

    mass_unit = qnm_filter.Filter.mass_unit(m_final)  # geometric time unit for scaling
    ctx = MaxLContext(
        ra=float(posterior["ra"][maxL_idx]),
        dec=float(posterior["dec"][maxL_idx]),
        t_geo=float(posterior["geocent_time"][maxL_idx]),
        psi=float(posterior["psi"][maxL_idx]),
        mass_unit=mass_unit,
        IMR=RemnantIMR(mass=m_final, chi=chi_final),
        posterior=posterior,
        maxL_idx=maxL_idx,
    )
    return ctx


# ----------------------------- GWOSC Data Readers --------------------------- #

def read_gwosc_timeseries(
    ifo: str,
    start: float,
    end: float,
    channel_tmpl: str = GWOSC_CHANNEL,
) -> TimeSeries:
    """
    Read a GWOSC frame segment for a given interferometer (ifo) and time window.
    If the file is not present locally, download it first.
    """
    ensure_gwosc_frames_present()
    url = GWOSC_GWF.get(ifo)
    if url is None:
        raise ValueError(f"No GWF path configured for IFO={ifo}.")
    channel = channel_tmpl.format(ifo=ifo)
    return TimeSeries.read(url, channel, start=start, end=end)


def gwosc_setup(
    detectors: Iterable[str] = ("H1", "L1"),
    psd_seconds: int = 64,
) -> Tuple[MaxLContext, Dict[str, "qnm_filter.RealData"], Dict[str, "qnm_filter.RealData"]]:
    """
    Prepare real (event) and off-source (noise) data around the max-likelihood trigger.

    Returns
    -------
    ctx : MaxLContext
        Max-likelihood context (posterior, sky location, masses, etc).
    real_data_dict : dict
        Per-IFO real strain as qnm_filter.RealData objects (+/-2 s window).
    noise_data_dict : dict
        Per-IFO off-source data (used for PSD and whitening, length=psd_seconds).
    """
    ctx = load_posterior_maxL()

    # Choose short windows around the event for analysis and a later segment for PSD
    event_start, event_end = (ctx.t_geo - 2.0, ctx.t_geo + 2.0)
    noise_start, noise_end = (ctx.t_geo + 1.0, ctx.t_geo + 1.0 + float(psd_seconds))

    real_data_dict: Dict[str, qnm_filter.RealData] = {}
    noise_data_dict: Dict[str, qnm_filter.RealData] = {}

    for ifo in detectors:
        data = read_gwosc_timeseries(ifo, event_start, event_end)
        noise = read_gwosc_timeseries(ifo, noise_start, noise_end)
        real_data_dict[ifo] = qnm_filter.RealData(data.value, index=data.times.value)
        noise_data_dict[ifo] = qnm_filter.RealData(noise.value, index=noise.times.value)

    return ctx, real_data_dict, noise_data_dict


# --------------------------- Antenna / Orientation -------------------------- #

def antenna_factors_for_ifos(
    ctx: MaxLContext,
    ifos: Iterable[str],
    sampling_frequency: int,
    duration: float,
    start_time: float,
) -> Dict[str, Tuple[float, float, object]]:
    """
    Compute plus/cross antenna patterns at geocenter time for each IFO.

    Returns
    -------
    mapping : dict
        { 'H1': (F_plus, F_cross, bilby_ifo), 'L1': (...), ... }
    """
    mapping: Dict[str, Tuple[float, float, object]] = {}
    for ifo in ifos:
        bilby_ifo = qnm_filter.set_bilby_predefined_ifo(
            ifo, sampling_frequency, duration, start_time=start_time
        )
        fp = bilby_ifo.antenna_response(ctx.ra, ctx.dec, ctx.t_geo, ctx.psi, "plus")
        fc = bilby_ifo.antenna_response(ctx.ra, ctx.dec, ctx.t_geo, ctx.psi, "cross")
        mapping[ifo] = (fp, fc, bilby_ifo)
    return mapping


# --------------------------- Surrogate Waveform ------------------------------ #

def build_surrogate_signal(ctx: MaxLContext, sampling_frequency: int = 4 * 4096) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate NRSur7dq4 surrogate strain h(t) at max-likelihood parameters.

    Returns
    -------
    time : np.ndarray
        Time samples for the surrogate.
    h_complex : np.ndarray (complex)
        Complex strain h = h_plus - i h_cross
    """
    # Load surrogate model
    sur = gwsurrogate.LoadSurrogate("NRSur7dq4")

    # Extract injection parameters from the posterior (maxL sample)
    p = dict(zip(ctx.posterior.dtype.names, ctx.posterior[ctx.maxL_idx]))

    # Reference/minimum frequency comes from the config in the HDF5
    with h5py.File(POSTERIOR_H5, "r") as f:
        rf = f["bilby-NRSur7dq4_prod-reweighted"]["config_file"]["config"]["reference-frequency"][0]
    f_ref = float(rf.decode("utf-8"))

    q = 1.0 / float(p["mass_ratio"])
    chiA = [float(p["spin_1x"]), float(p["spin_1y"]), float(p["spin_1z"])]
    chiB = [float(p["spin_2x"]), float(p["spin_2y"]), float(p["spin_2z"])]
    f_low = 20.0
    M = float(p["total_mass"])
    dist_mpc = float(p["luminosity_distance"])
    inclination = float(p["iota"])
    phi_ref = float(p["phase"])

    dt = 1.0 / float(sampling_frequency)

    # Surrogate evaluate: returns time array t, complex h(t), and dynamics
    t, h, _dyn = sur(
        q, chiA, chiB, dt=dt, f_low=f_low, f_ref=f_ref, M=M, dist_mpc=dist_mpc,
        inclination=inclination, phi_ref=phi_ref, units="mks",
    )
    return t, h


def inject_into_ifos(
    t: np.ndarray,
    h_complex: np.ndarray,
    ifo_map: Dict[str, Tuple[float, float, object]],
) -> Dict[str, "qnm_filter.RealData"]:
    """
    Project surrogate strain onto each IFO using their antenna patterns.

    Returns
    -------
    dict ifo->RealData for strain projected and re-sampled on IFO time grid.
    """
    # Split complex strain: h = h_plus - i h_cross
    hplus = np.real(h_complex)
    hcross = -np.imag(h_complex)

    # Interpolants on the surrogate time grid
    hplus_fn = interp1d(t, hplus, bounds_error=False, fill_value=0.0)
    hcross_fn = interp1d(t, hcross, bounds_error=False, fill_value=0.0)

    out: Dict[str, qnm_filter.RealData] = {}
    for ifo, (fp, fc, bilby_ifo) in ifo_map.items():
        time_grid = bilby_ifo.strain_data.time_array
        s = hplus_fn(time_grid) * fp + hcross_fn(time_grid) * fc
        out[ifo] = qnm_filter.RealData(s, index=time_grid)
    return out


# ------------------------------- Whitening/BP ------------------------------- #

def whiten_and_bandpass(
    fit: "qnm_filter.Network",
    ifo: str,
    attr: Dict[str, "qnm_filter.RealData"],
    fmin: float = 50.0,
    fmax: float = 2000.0,
) -> "qnm_filter.RealData":
    """
    Triangular-solve whitening using Cholesky factor from the network,
    then band-pass in GWpy for visualization.
    """
    truncated = fit.truncate_data(attr)
    # Solve L * y = x  --> y = L^{-1} x
    white = sl.solve_triangular(fit.cholesky_L[ifo], truncated[ifo], lower=True)
    white_ts = TimeSeries(white, times=truncated[ifo].time)
    bp = white_ts.bandpass(fmin, fmax)
    return qnm_filter.RealData(bp, index=truncated[ifo].time)


# ------------------------------- Main Pipeline ------------------------------ #

def build_network(
    data: Dict[str, "qnm_filter.RealData"],
    noise: Dict[str, "qnm_filter.RealData"],
    model_list: List[Tuple[int, int, int, str]],
    t_init: float,
    segment_length: float,
    srate: int,
    ra: float | None,
    dec: float | None,
    flow: float = 20.0,
    trim: float = 0.1,
    mass: float = 0.0,
    chi: float = 0.0,
) -> "qnm_filter.Network":
    """Set up a qnm_filter.Network with given data/noise and filter options."""
    cfg = dict(
        model_list=model_list,
        t_init=t_init,
        segment_length=segment_length,
        srate=srate,
        ra=ra,
        dec=dec,
        flow=flow,
        trim=trim,
    )
    net = qnm_filter.Network(**cfg)
    net.original_data = copy.deepcopy(data)
    net.pure_noise = copy.deepcopy(noise)
    net.detector_alignment()
    net.condition_data("original_data", **cfg)
    net.condition_data("pure_noise", **cfg)
    net.add_filter(mass=mass, chi=chi, model_list=model_list)
    net.compute_acfs("pure_noise")
    net.cholesky_decomposition()
    net.first_index()
    del net.pure_noise
    return net


def main() -> None:
    # ------------------------ Load posterior & data ------------------------ #
    # Example IFO: H1
    ifo = "H1"
    ctx, real_data, noise_data = gwosc_setup(detectors=[ifo], psd_seconds=64)

    # Antenna responses on high-rate grids for plotting overlays
    ifo_map_hi = antenna_factors_for_ifos(
        ctx, [ifo], sampling_frequency=16384, duration=4.0, start_time=-2.0
    )

    # ------------------------ Surrogate (NRSur7dq4) ------------------------ #
    t_sur, h_sur = build_surrogate_signal(ctx, sampling_frequency=4 * 4096)

    # Pad complex strain for FFT stability (follow original intent)
    complexData = qnm_filter.ComplexData(h_sur, index=t_sur)
    padded = complexData.pad_complex_data_for_fft(2, 2)

    # Inject surrogate into the same IFO geometry for comparison
    nrsur_data = inject_into_ifos(padded.time, padded.values, ifo_map_hi)

    # ------------------------ QNM model list & timing ---------------------- #
    # Example modes: (l, m, n, prograde/retrograde)
    model_list = [(2, 2, 0, "p"), (2, 2, 1, "p"), (2, 2, 2, "p")]
    tshift = qnm_filter.compute_filter_time_shift(ctx.IMR.chi, model_list, True, ctx.IMR.mass)

    # ------------------------ Build plotting networks ---------------------- #
    # Real-event network 
    plotting_cfg = dict(
        model_list=model_list,
        t_init=ctx.t_geo - 0.1,
        segment_length=0.4,
        srate=8192,
        ra=ctx.ra,
        dec=ctx.dec,
        flow=20.0,
        trim=0.1,
        mass=ctx.IMR.mass,
        chi=ctx.IMR.chi,
    )
    fit_plotting = build_network(real_data, noise_data, **plotting_cfg)

    # Surrogate network 
    nrsur_cfg = dict(
        model_list=model_list,
        t_init=-0.1,
        segment_length=0.4,
        srate=8192,
        ra=None,
        dec=None,
        flow=20.0,
        trim=0.1,
        mass=ctx.IMR.mass,
        chi=ctx.IMR.chi,
    )
    fit_nrsur = build_network(nrsur_data, noise_data, **nrsur_cfg)

    # ----------------------------- Plotting -------------------------------- #
    fig, axs = pl.subplots(nrows=2, sharex=True, sharey=True, figsize=[8, 5.5])

    # Real event: whitened + bandpassed (original vs QNM-removed)
    original = whiten_and_bandpass(fit_plotting, ifo, fit_plotting.original_data)
    original_time = original.time - float(ctx.posterior[f"{ifo}_time"][ctx.maxL_idx])

    filtered = whiten_and_bandpass(fit_plotting, ifo, fit_plotting.filtered_data)
    filtered_time = filtered.time + tshift - float(ctx.posterior[f"{ifo}_time"][ctx.maxL_idx])

    axs[0].plot(original_time, original.values, label="Hanford data", alpha=0.5, lw=3)
    axs[1].plot(filtered_time, filtered.values, label="Hanford data", alpha=0.5, lw=3)

    # Surrogate: whitened + bandpassed (original vs QNM-removed)
    nrsur_original = whiten_and_bandpass(fit_nrsur, ifo, fit_nrsur.original_data)
    nrsur_filtered = whiten_and_bandpass(fit_nrsur, ifo, fit_nrsur.filtered_data)

    axs[0].plot(nrsur_original.time, nrsur_original.values,
                label="Waveform model reconstruction", alpha=0.7, lw=3)
    axs[1].plot(nrsur_filtered.time + tshift, nrsur_filtered.values,
                label="Waveform model reconstruction", alpha=0.7, lw=3)

    # Cosmetics
    axs[0].set_xlim(-0.06, 0.04)
    axs[0].legend(loc="lower left", handlelength=1.4, frameon=False)

    axs[-1].set_xlabel("$t$ [s]")
    fig.supylabel("Whitened Strain", x=0.02)

    axs[0].text(0.038, 5.2, "Original", fontsize=12,
                ha="right", bbox=dict(facecolor="none", edgecolor="k"))
    axs[1].text(0.038, 5.2, "Quasinormal modes removed", fontsize=12,
                ha="right", bbox=dict(facecolor="none", edgecolor="k"))

    pl.tight_layout()
    pl.subplots_adjust(hspace=0)
    pl.savefig("plot.pdf")
    print("Saved figure: plot.pdf")


# ------------------------------- Entrypoint --------------------------------- #

if __name__ == "__main__":
    main()
