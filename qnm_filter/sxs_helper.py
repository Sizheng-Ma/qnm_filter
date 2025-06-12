"""Wrapping up some SXS (numerical relativity) functions for convenience
"""
__all__ = ["SXSWaveforms"]

from .gw_data import *
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
        self.data_in_si = {}

        self.filename = kws.get("filename", None)
        self.mf = None
        self.chif = None
        self.chif_vec = None

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
            self.filename,
            extrapolation_order=extrapolation_order,
            download=download,
        ).Strain
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
            interplated_waveform = np.array(waveform_lm.interpolate(ts))
            self.original_data[str(l) + str(m)] = ComplexData(
                interplated_waveform, index=ts - tp
            )
        else:
            if delta_t != None:
                warnings.warn("delta_t: {} is not used".format(delta_t))
            index_i = waveform_lm.index_closest_to(t_interp_i)
            index_f = waveform_lm.index_closest_to(t_interp_f)
            waveform_lm_trunc = waveform_lm[index_i:index_f]
            self.original_data[str(l) + str(m)] = ComplexData(
                np.array(waveform_lm_trunc), index=waveform_lm_trunc.t - tp
            )

    def get_remnant_data(self, download=False) -> None:
        """Get remnant mass and dimensionless spin from SXS catalog.
        Note only spin's length is returned.

        Parameters
        ----------
        download : bool, optional
            download meta data from SXS catalog if ture, by default False
        """
        metadata = sxs.load(self.filename, download=download)
        if self.mf != None:
            warnings.warn("Overwriting mf: {}".format(self.mf))
        if self.chif != None:
            warnings.warn("Overwriting chif: {}".format(self.chif))
        self.mf = metadata.load_metadata()["remnant_mass"]
        spinvec = metadata.load_metadata()["remnant_dimensionless_spin"]
        self.chif_vec = spinvec
        self.chif = np.sqrt(np.sum(np.array(spinvec) ** 2))

    @property
    def get_bbh_spin1(self):
        """Get the spin vector of the first BH (at a reference time during inspiral)"""
        metadata = sxs.load(self.filename).load_metadata()
        return metadata["reference_dimensionless_spin1"]

    @property
    def get_bbh_spin2(self):
        """Get the spin vector of the second BH (at a reference time during inspiral)"""
        metadata = sxs.load(self.filename).load_metadata()
        return metadata["reference_dimensionless_spin2"]

    @property
    def get_bbh_m1(self):
        """Get the mass of the first BH (at a reference time during inspiral)"""
        metadata = sxs.load(self.filename).load_metadata()
        return metadata["reference_mass1"]

    @property
    def get_bbh_m2(self):
        """Get the mass of the second BH (at a reference time during inspiral)"""
        metadata = sxs.load(self.filename).load_metadata()
        return metadata["reference_mass2"]

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
            self.padded_data[lm] = data.pad_complex_data_for_fft(partition, len_pow)

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
            self.data_in_si[lm] = ComplexData(
                scaled_waveform, index=scaled_time, ifo=data.ifo
            )

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
            "plus": RealData(hp, index=time, ifo=ifo),
            "cross": RealData(hc, index=time, ifo=ifo),
        }

    def add_filter(self, lm, model_list):
        r"""Apply rational filters listed in `model_list` to the :math:`lm` harmonic of the NR waveform.

        Parameters
        ----------
        lm : string
            the :math:`lm` harmonic to be filtered
        model_list : a list of dictionaries
            quasinormal modes to be filtered.

        Returns
        -------
        ComplexData
            filtered harmonic
        """
        data = self.padded_data[lm]
        data_in_freq = data.fft_data
        freq = data.fft_freq
        filter_in_freq = Filter(
            chi=self.chif, mass=self.mf, model_list=model_list
        ).NR_filter(freq)
        data_in_time = np.fft.fft(filter_in_freq * data_in_freq, norm="ortho")
        return ComplexData(data_in_time, index=data.time)

    @staticmethod
    def trunc_pad(data, before, after, partition, len_pow):
        truncated_data = data.truncate_data(before=before, after=after)
        return truncated_data.pad_complex_data_for_fft(partition, len_pow)
