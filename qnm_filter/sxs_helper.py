"""Wrapping up some SXS (numerical relativity) functions for convenience
"""
__all__ = ["SXSWaveforms"]

from .gw_data import *
from .utility import *
import sxs
import lal
import numpy as np
import warnings
import astropy.constants as c

MPC = c.kpc.value * 1e3 / c.c.value


class SXSWaveforms:
    """Container for numerical relativity waveforms, downloaded from
    SXS catalog https://data.black-holes.org/waveforms/index.html

    Attributes
    ----------
    original_data : dictionary
        dictionary that stores all the GW harmonics
    padded_data : dictionary
        dictionary that stores all the GW harmonics padded with 0 on both sides.
        This is mainly for FFT so the time stamp should be evenly spaced, and the total length needs to be a power of 2.
    data_in_si : dictionary
        dictionary that stores all the GW harmonics that have SI units
    filename : string
        SXS ID, in the format of `SXS:BBH:XXXX`
    mf : float
        remnant mass, in the unit of BBH's total mass
    chif : float
        remnant dimensionless spin
    """

    def __init__(self, **kws) -> None:
        """Constructor"""
        self.original_data = {}
        self.padded_data = {}
        self.filtered_data = {}
        self.data_in_si = {}

        self.filename = kws.get("filename", None)
        self.mf = None
        self.chif = None

    def import_sxs_data(
        self,
        l,
        m,
        interpolate,
        extrapolation_order=2,
        download=False,
        ti=None,
        tf=None,
        delta_t=None,
    ) -> None:
        """Download/Load SXS waveforms.

        Parameters
        ----------
        l : int
            GW harmonic index
        m : int
            GW harmonic index
        interpolate : bool
            interpolate to desired time stamps if ture, otherwise use sxs's original data
        extrapolation_order : int, optional
            waveform extraploation order, used internally by `sxs`, by default 2
        download : bool, optional
            download GWs from SXS catalog if ture, by default False
        ti : float, optional
            the initial time of desired time stamps, use the first time stamp by default.
            The strain's peak is assumed to be at 0.
        tf : float, optional
            the final time of desired time stamps, use the last time stamp by default.
            The strain's peak is assumed to be at 0.
        delta_t : float, optional
            the step size of desired time stamps, use sxs's original data by default,
            whose time stamps may not be even spacing

        Raises
        ------
        ValueError
            when the time interpolator cannot find `delta_t`
        """
        waveform = sxs.load(
            self.filename + "/Lev/rhOverM",
            extrapolation_order=extrapolation_order,
            download=download,
        )
        tp = waveform.max_norm_time()
        waveform_lm = waveform[:, waveform.index(l, m)]

        if ti == None:
            t_interp_i = 0
        else:
            t_interp_i = ti + tp
        if tf == None:
            t_interp_f = waveform_lm.t[-1]
        else:
            t_interp_f = tf + tp

        if interpolate:
            if delta_t == None:
                raise ValueError("Invalid delta_t: {}".format(delta_t))

            ts = np.arange(t_interp_i, t_interp_f, delta_t)
            interplated_waveform = waveform_lm.interpolate(ts).data
            self.original_data[str(l) + str(m)] = Data(
                interplated_waveform, index=ts - tp
            )
        else:
            if delta_t != None:
                warnings.warn("delta_t: {} is not used".format(delta_t))
            index_i = waveform_lm.index_closest_to(t_interp_i)
            index_f = waveform_lm.index_closest_to(t_interp_f)
            waveform_lm_trunc = waveform_lm[index_i:index_f]
            self.original_data[str(l) + str(m)] = Data(
                waveform_lm_trunc.data, index=waveform_lm_trunc.t - tp
            )

    def get_meta_data(self, download=False) -> None:
        """Get meta data, including remnant mass and dimensionless spin.
        Note only spin's length is returned.

        Parameters
        ----------
        download : bool, optional
            download meta data from SXS catalog if ture, by default False
        """
        metadata = sxs.load(self.filename + "/Lev/metadata.json", download=download)
        if self.mf != None:
            warnings.warn("Overwriting mf: {}".format(self.mf))
        if self.chif != None:
            warnings.warn("Overwriting chif: {}".format(self.chif))
        self.mf = metadata["remnant_mass"]
        spinvec = metadata["remnant_dimensionless_spin"]
        self.chif = np.sqrt(np.sum(np.array(spinvec) ** 2))

    def pad_data(self, partition, len_pow) -> None:
        r"""Pad zeros on both sides of GW harmonics :attr:`self.original_data`,
        the final length is :math:`2^{\textrm{len\_pow}}`

        Parameters
        ----------
        partition : int
            fraction of zeros to be padded on the left
        len_pow : int
            the final length of padded data is :math:`2^{\textrm{len\_pow}}`
        """
        for lm, data in self.original_data.items():
            self.padded_data[lm] = pad_data_for_fft(data, partition, len_pow)

    def scale_to_si(self, attr_name, mass, distance) -> None:
        """Convert GW waveforms stored in `attr_name` from numerical-relativity's units to SI units.

        Parameters
        ----------
        attr_name : string
            the name of attribute
        mass : float
            binary's total mass, in solar mass. Not to be confused with the remnant mass.
        distance : float
            binary's luminous intensity, in MPC.
        """
        for lm, data in getattr(self, attr_name).items():
            scaled_time = data.time * Filter.mass_unit(mass)
            scaled_waveform = data.values * Filter.mass_unit(mass) / distance / MPC
            self.data_in_si[lm] = Data(scaled_waveform, index=scaled_time, ifo=data.ifo)

    def harmonics_to_polarizations(self, attr_name, iota, beta, model_list) -> None:
        """Compute two polarizations from GW harmonics stored in `attr_name`

        Parameters
        ----------
        attr_name : string
            the name of attribute
        iota : float
            inclination angle of propagation direction, in rad.
        beta : float
            azimuthal angle of propagation direction, in rad.
        model_list : a list of dictionaries
            harmonics to be added

        Returns
        -------
        Dictionary
            plus and cross polarizations
        """
        strain = 0
        for l, m in model_list:
            ylm = lal.SpinWeightedSphericalHarmonic(iota, beta, -2, l, m)
            strain += getattr(self, attr_name)[str(l) + str(m)] * ylm
        time = getattr(self, attr_name)[str(l) + str(m)].time
        ifo = getattr(self, attr_name)[str(l) + str(m)].ifo
        hp = np.real(strain)
        hc = -np.imag(strain)
        return {
            "plus": Data(hp, index=time, ifo=ifo),
            "cross": Data(hc, index=time, ifo=ifo),
        }
