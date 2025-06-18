"""Wrapping up some Bilby functions for convenience
"""
__all__ = [
    "bilby_get_strain",
    "set_bilby_ifo",
    "set_bilby_predefined_ifo",
    "bilby_injection",
    "bilby_construct_noise_from_file",
]

from .gw_data import *
import copy
import bilby
import numpy as np
from scipy.interpolate import interp1d


def bilby_construct_noise_from_file(
    filename, duration, sampling_frequency, fhigh, delta_f=0.1, flow=0
):
    filereader = np.loadtxt(filename)
    freq_target = np.arange(flow, fhigh + delta_f, delta_f)
    value_interp = interp1d(
        filereader[:, 0],
        filereader[:, 1],
        bounds_error=False,
        fill_value=(filereader[:, 1][0], filereader[:, 1][-1]),
    )(freq_target)
    bilby_psd = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=freq_target, psd_array=value_interp**2
    )
    ifo = bilby.gw.detector.Interferometer(
        name=None,
        length=0,
        latitude=0,
        longitude=0,
        elevation=0,
        xarm_azimuth=0,
        yarm_azimuth=90,
        power_spectral_density=bilby_psd,
        minimum_frequency=flow,
        maximum_frequency=fhigh,
    )
    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=-duration / 2,
    )

    noise_class = Noise(
        signal=ifo.strain_data.time_domain_strain,
        time=ifo.strain_data.time_array,
    )
    noise_class.welch()
    noise_class.from_psd()
    return noise_class


def bilby_get_strain(ifo, time_offset):
    """Get strain data from Bilby's `Interferometer` and store the result in `Data`.

    Parameters
    ----------
    ifo : bilby.gw.detector.Interferometer
        An instance of `bilby.gw.detector.Interferometer`
    time_offset : float
        The time offset applied to data

    Returns
    -------
    Data
        strain data
    """
    return RealData(
        ifo.strain_data.time_domain_strain,
        index=ifo.strain_data.time_array - time_offset,
        ifo=ifo.name,
    )


def set_bilby_predefined_ifo(
    name,
    sampling_frequency,
    duration,
    start_time,
    zero_noise=False,
):
    """Construct a Bilby's `Interferometer` instance with its internal PSD.

    Parameters
    ----------
    name : string
        name of interferometer, e.g., "H1" and "L1"
    sampling_frequency : float
        in Hz
    duration : float
        in second. The duration of the generated time series
    start_time : float
        in second. The start time of the generated time series
    zero_noise : bool, optional
        if ture, returns pure waveform strain w/o noise, by default False

    Returns
    -------
    bilby.gw.detector.Interferometer
        An instance of `bilby.gw.detector.Interferometer`
    """
    ifo = bilby.gw.detector.InterferometerList([name])[0]
    if zero_noise:
        ifo.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
        )
    else:
        ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
        )
    return ifo


def set_bilby_ifo(
    Noise,
    sampling_frequency,
    duration,
    start_time,
    minimum_frequency,
    maximum_frequency,
    zero_noise=False,
    name=None,
    length=0,
    latitude=0,
    longitude=0,
    elevation=0,
    xarm_azimuth=0,
    yarm_azimuth=90,
):
    """Construct a Bilby's `Interferometer` instance given the :class:`Noise` class.

    Parameters
    ----------
    Noise : :class:`Noise`
        An instance of :class:`Noise`
    sampling_frequency : float
        in Hz
    duration : float
        in second. The duration of the generated time series
    start_time : float
        in second. The start time of the generated time series
    minimum_frequency : float
        in Hz. Minimum frequency to analyse for detector.
    maximum_frequency : float
        in Hz. Maximum frequency to analyse for detector.
    zero_noise : bool, optional
        if ture, returns pure waveform strain w/o noise, by default False
    name : string, optional
        name of interferometer, by default None
    length : int, optional
        length of interferometer, by default 0
    latitude : int, optional
        latitude of interferometer, by default 0
    longitude : int, optional
        longitude of interferometer, by default 0
    elevation : int, optional
        elevation of interferometer, by default 0
    xarm_azimuth : int, optional
        azimuth angle of the x-arm of interferometer, by default 0
    yarm_azimuth : int, optional
        azimuth angle of the y-arm  of interferometer, by default 90

    Returns
    -------
    bilby.gw.detector.Interferometer
        An instance of `bilby.gw.detector.Interferometer`
    """
    ifo = bilby.gw.detector.Interferometer(
        power_spectral_density=Noise.bilby_psd,
        name=name,
        length=length,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        xarm_azimuth=xarm_azimuth,
        yarm_azimuth=yarm_azimuth,
    )
    if zero_noise:
        ifo.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
        )
    else:
        ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
        )

    return ifo


def bilby_injection(ifo, NR_injection_into_Bilby, **injection_parameters):
    """Inject a GW strain into noise

    Parameters
    ----------
    ifo : bilby.gw.detector.Interferometer
        An instance of `bilby.gw.detector.Interferometer`
    NR_injection_into_Bilby : function
        waveform to be injected.

        Example::

            def NR_injection_into_Bilby(time, **waveform_kwargs):
                return {'plus': foo, 'cross': bar}

    Returns
    -------
    bilby.gw.detector.Interferometer
        An instance of `bilby.gw.detector.Interferometer`
    """
    waveform = bilby.gw.waveform_generator.WaveformGenerator(
        duration=ifo.duration,
        sampling_frequency=ifo.sampling_frequency,
        time_domain_source_model=NR_injection_into_Bilby,
        parameters=injection_parameters,
        start_time=ifo.start_time,
    )

    ifo_new = copy.deepcopy(ifo)

    ifo_new.inject_signal(waveform_generator=waveform, parameters=injection_parameters)
    return ifo_new
