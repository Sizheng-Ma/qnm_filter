"""Wrapping up some Bilby functions for convenience
"""
__all__ = [
    "bilby_get_strain",
    "set_bilby_ifo",
    "set_bilby_predefined_ifo",
    "bilby_injection",
]

from .gw_data import *
import copy
import bilby


def bilby_get_strain(ifo):
    """Get strain data from Bilby's `Interferometer` and store the result in `Data`.

    Parameters
    ----------
    ifo : bilby.gw.detector.Interferometer
        An instance of `bilby.gw.detector.Interferometer`

    Returns
    -------
    Data
        strain data
    """
    return Data(
        ifo.strain_data.time_domain_strain,
        index=ifo.strain_data.time_array,
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
        minimum_frequency=min(Noise.psd.time),
        maximum_frequency=max(Noise.psd.time),
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
