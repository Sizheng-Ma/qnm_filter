"""Useful functions for calculating and plotting data
"""
__all__ = [
    "parallel_compute",
    "find_credible_region",
    "project_to_1d",
    "pad_data_for_fft",
    "evidence_parallel",
    "save_class",
    "load_class",
    "time_to_index",
    "time_shift_from_sky",
    "posterior_quantile_2d",
]

from joblib import Parallel, delayed
import matplotlib.pyplot as pl
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import warnings
import pickle
import lal


def parallel_compute(self, M_arr, chi_arr, num_cpu=-1, **kwargs):
    """Parallel computation of a function that takes 2 arguments

    Arguments
    ---------
    self : Network class instance
        An instance of a Network class that will have self.likelihood_vs_mass_spin computed.
    M_arr : array-like
        array of the values of remnant mass to calculate the likelihood function for.
    chi_arr : array-like
        array of the values of remnant spin to calculate the likelihood function for.
    num_cpu : int
        integer to be based to Parallel as n_jobs. NOTE: passing a positive integer leads to better performance than -1 but performance differs across machines.
    kwargs : dict
        dictionary of kwargs of the function

    Returns
    ---------
    reshaped_results : ndarray
        2d array of the results with shape (len(x_arr), len(y_arr))
    """
    flatten_array = [(i, j) for i in M_arr for j in chi_arr]
    results = Parallel(num_cpu)(
        delayed(self.likelihood_vs_mass_spin)(i, j, **kwargs) for i, j in flatten_array
    )
    reshaped_results = np.reshape(results, (len(M_arr), len(chi_arr))).T
    return reshaped_results, logsumexp(reshaped_results)


def evidence_parallel(
    self,
    index_spacing,
    num_iteration,
    initial_offset,
    M_arr,
    chi_arr,
    num_cpu=-1,
    apply_filter=True,
    verbosity=False,
    **kwargs,
):
    """Compute evidence curve, which is sampled at multiples of the post-downsampling rate `self.srate`,
    therefore there is no need to recondition the data set.

    Parameters
    ----------
    index_spacing : int
        the ratio between `self.srate` and the evidence's sampling rate
    num_iteration : int
        number of sampling points for the evidence curve
    initial_offset : int
        the index offset of the first evidence data point with respect to `self.i0_dict`
    M_arr : array-like
        array of the values of remnant mass to calculate the likelihood function for
    chi_arr : array-like
        array of the values of remnant spin to calculate the likelihood function for
    num_cpu : int, optional
        integer to be based to Parallel as n_jobs. NOTE: passing a positive integer leads to better performance than -1 but performance differs across machines, by default -1
    verbosity : bool, optional
        print more information, by default False

    Returns
    -------
    Two arrays
        time stamps, log-evidence
    """
    flatten_array = [(i, j) for i in M_arr for j in chi_arr]
    saved_log_evidence = []
    self.shift_first_index(initial_offset)
    if verbosity:
        print(self.i0_dict)
    for time_iter in range(num_iteration):
        if apply_filter:
            results = Parallel(num_cpu)(
                delayed(self.likelihood_vs_mass_spin)(i, j, **kwargs)
                for i, j in flatten_array
            )
        else:
            results = (
                [self.compute_likelihood(apply_filter=False)]
                * len(M_arr)
                * len(chi_arr)
            )
        log_evidence = logsumexp(results)
        saved_log_evidence.extend([log_evidence])
        self.shift_first_index(index_spacing)
        if verbosity:
            print(time_iter)
    t_array = (
        self.t_init
        + (initial_offset + np.arange(num_iteration) * index_spacing) / self.srate
    )
    return t_array, np.array(saved_log_evidence)


def time_to_index(self, index_spacing, tmin, tmax):
    """Estimate `initial_offset` and `num_iteration` for the evidence calculator given physical times `tmin` and `tmax`.

    Parameters
    ----------
    index_spacing : int
        the ratio between `self.srate` and the evidence's sampling rate
    tmin : float
        the start time of the evidence curve
    tmax : float
        the end time of the evidence curve

    Returns
    -------
    initial_offset : int
        the index offset of the first evidence data point with respect to `self.i0_dict`
    num_iteration : int
        number of sampling points for the evidence curve
    """
    initial_offset = int((tmin - self.t_init) * self.srate)
    num_iteration = (
        int(((tmax - self.t_init) * self.srate - initial_offset) / index_spacing) + 1
    )
    return initial_offset, num_iteration


def find_probability_difference(threshold, array2d):
    """Calculates the difference between the log probability of sampling array2d above the threshold and log target_probability

    Parameters
    ----------
    threshold : float
        value to consider the probability of sampling above
    array2d : ndarray
        2D array of sampling log likelihood as a function of mass and spin

    Returns
    -------
    float
        log probability of the given sampling array2d above the threshold
    """
    tot = logsumexp(array2d)
    region = array2d[array2d > threshold]
    if region.size == 0:
        prob = 0
    else:
        region_tot = logsumexp(region)
        prob = region_tot - tot
    return prob


def find_credible_region(array2d, num_cpu=-1, target_probability=0.9):
    """Compute the log likelihood contour that encloses the desired probability.

    Parameters
    ----------
    array2d : ndarray
        2D array of sampling log likelihood as a function of mass and spin
    num_cpu : int, optional
        number of CPUs used for parallelization, by default -1
    target_probability : float, optional
        desired probability, by default 0.9

    Returns
    -------
    result : float
        the log likelihood above which has the desired probability.

    Raises
    ------
    ValueError
        when the target log likelihood cannot be found.
    """
    # iterate over the inputted log likelihoods and compute the distance of their log probability from the desired value.
    sorted_likelihood = np.sort(array2d.flatten())

    sorted_probability = Parallel(num_cpu)(
        delayed(find_probability_difference)(i, array2d) for i in sorted_likelihood
    )
    sorted_probability = np.array(sorted_probability)

    # interpolation is preferred when the sample density is insufficient
    interp_probability = interp1d(sorted_probability, sorted_likelihood)
    return interp_probability(np.log(target_probability))


def posterior_quantile_2d(array2d, fit, mass, spin, model_list, num_cpu=-1):
    """Compute the posterior quantile of the queried mass and spin

    Parameters
    ----------
    array2d : ndarray
        2D array of sampling log likelihood as a function of mass and spin
    fit : Network
        a Network object
    mass : float
        the queried mass
    spin : float
        the queried spin
    model_list : a list of dictionaries
        quasinormal modes to be filtered
    num_cpu : int, optional
        integer to be based to Parallel as n_jobs, by default -1

    Returns
    -------
    float
        the computed posterior quantile
    """
    this_likelihood = fit.likelihood_vs_mass_spin(mass, spin, model_list=model_list)

    # iterate over the inputted log likelihoods and compute the distance of
    # their log probability from the desired value.
    sorted_likelihood = np.sort(array2d.flatten())

    sorted_probability = Parallel(num_cpu)(
        delayed(find_probability_difference)(i, array2d) for i in sorted_likelihood
    )
    sorted_probability = np.array(sorted_probability)
    interp_probability = interp1d(sorted_likelihood, sorted_probability)
    if min(sorted_likelihood) <= this_likelihood <= max(sorted_likelihood):
        return np.exp(interp_probability(this_likelihood))
    elif this_likelihood <= min(sorted_likelihood):
        return 1
    elif this_likelihood >= max(sorted_likelihood):
        return 0


def project_to_1d(array2d, delta_mass, delta_chi):
    """Project the 2D log likelihood to 1D probability density functions,
    whose integrations are normalized to be 1.

    Parameters
    ----------
    array2d : ndarray
        2D array of sampling log likelihood as a function of mass and spin
    delta_mass : float
        step size of mass
    delta_chi : float
        step size of chi

    Returns
    -------
    Two ndarrays
        probability density functions of mass and spin, both normalized to a total probability of 1.
    """
    log_evidence = logsumexp(array2d)
    normalized_mass = np.exp(logsumexp(array2d, axis=0) - log_evidence)
    normalized_chi = np.exp(logsumexp(array2d, axis=1) - log_evidence)

    normalized_mass /= np.sum(normalized_mass * delta_mass)
    normalized_chi /= np.sum(normalized_chi * delta_chi)
    return normalized_mass, normalized_chi


def pad_data_for_fft(data, partition, len_pow) -> None:
    r"""Pad zeros on both sides of `data`, the final length is :math:`2^{\textrm{len\_pow}}`

    Parameters
    ----------
    data : Data
        data to be padded
    partition : int
        fraction of zeros to be padded on the left
    len_pow : int
        the final length of padded data is :math:`2^{\textrm{len\_pow}}`

    Returns
    -------
    Data
        padded data
    """
    padlen = 2 ** (len_pow + int(np.ceil(np.log2(len(data))))) - len(data)
    data_pad = np.pad(
        data.values,
        (padlen // partition, padlen - (padlen // partition)),
        "constant",
        constant_values=(0, 0),
    )

    delta_t = data.time_interval
    end1 = data.index[-1] + (padlen - (padlen // partition)) * delta_t
    end2 = data.index[0] - (padlen // partition) * delta_t

    tpad = np.pad(
        data.index,
        (padlen // partition, padlen - (padlen // partition)),
        "linear_ramp",
        end_values=(end2, end1),
    )
    return tpad, data_pad


def save_class(cls, filename):
    """Dump a class object to a file

    Parameters
    ----------
    filename : string
        the file name to be dumped
    """
    with open(filename, "wb") as file:
        pickle.dump(cls, file)


def load_class(filename):
    """Read a class object from a file

    Parameters
    ----------
    filename : string
        the file name to be read

    Returns
    -------
    fit
        class object saved in the file
    """
    with open(filename, "rb") as file:
        fit = pickle.load(file)
    return fit


def time_shift_from_sky(ifo, ra, dec, t_init):
    """Get time offset with respect to geocenter given the information

    Parameters
    ----------
    ifo : str
        name of interferometer.
    ra : float
        source right ascension, in radian.
    dec : float
        source declination, in radian.
    t_init : float
        trucation time (start time of analysis segment) at geocenter.

    Returns
    -------
    dt_ifo : float
        the time offset
    """
    tgps = lal.LIGOTimeGPS(t_init)
    location = lal.cached_detector_by_prefix[ifo].location
    dt_ifo = lal.TimeDelayFromEarthCenter(location, ra, dec, tgps)
    return dt_ifo
