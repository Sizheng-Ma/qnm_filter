"""Useful functions for calculating and plotting data
"""
__all__ = [
    "parallel_compute",
    "find_credible_region",
    "project_to_1d",
    "pad_data_for_fft",
    "evidence_parallel",
]

from .gw_data import *
from joblib import Parallel, delayed
import matplotlib.pyplot as pl
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from sklearn.utils.extmath import cartesian


def parallel_compute(self, M_arr, chi_arr, num_cpu=-1, offset = 0, **kwargs):
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
    offset : integer
        offset from the index of t_init to compute the likelihood function for.
    kwargs : dict
        dictionary of kwargs of the function

    Returns
    ---------
    reshaped_results : ndarray
        2d array of the results with shape (len(x_arr), len(y_arr))
    """
    flatten_array = [(i, j) for i in M_arr for j in chi_arr]
    results = Parallel(num_cpu)(
        delayed(self.likelihood_vs_mass_spin)(offset, i, j, **kwargs) for i, j in flatten_array
    )
    reshaped_results = np.reshape(results, (len(M_arr), len(chi_arr))).T
    return reshaped_results, logsumexp(reshaped_results)


def evidence_parallel(
    self,
    offset_arr,
    M_arr,
    chi_arr,
    num_cpu=-1,
    apply_filter=True,
    **kwargs,
):
    """Compute evidence curve, which is sampled at multiples of the post-downsampling rate `self.srate`,
    therefore there is no need to recondition the data set.

    Parameters
    ----------
    offset_arr : array-like
        array of offset indexes (w.r.t the index of t_init) to calculate the evidence for
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
    flatten_array = cartesian((offset_arr, M_arr, chi_arr))
    if apply_filter:
        results = Parallel(num_cpu)(
            delayed(self.likelihood_vs_mass_spin)(offset, M, chi, **kwargs)
            for offset, M, chi in flatten_array
        )
    else:
        results = np.array(
            [[self.compute_likelihood(offset, apply_filter=False)] * len(M_arr)
            * len(chi_arr) for offset in offset_arr])
    results = np.reshape(results, (len(offset_arr), len(M_arr), len(chi_arr)))
    log_evidence = logsumexp(results, axis = (1,2))
    t_array = (
        self.t_init
        + offset_arr / self.srate
    )
    return t_array, np.array(log_evidence)


def find_probability_difference(threshold, array2d, target_probability=0.9):
    """Calculates the difference between the log probability of sampling array2d above the threshold and log target_probability

    Parameters
    ----------
    threshold : float
        value to consider the probability of sampling above
    array2d : ndarray
        2D array of sampling log likelihood as a function of mass and spin
    target_probability : float, optional
        function returns 0 if the probability of sampling above the level = target_probability, by default 0.9

    Returns
    -------
    float
        difference between the log probability of sampling array2d above the threshold and log target_probability
    """
    tot = logsumexp(array2d)
    region = array2d[array2d > threshold]
    region_tot = logsumexp(region)
    prob = region_tot - tot
    return prob - np.log(target_probability)


def sampling_probability(array2d, num_cpu=-1, target_probability=0.9):
    """Sort 2D likelihood array from minimum to maximum, and compute the difference between the corresponding log probability and log target_probability. The probability enclosed by the minimum likelihood contour is normalized to be 1.

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
    sorted_array : ndarray
        sorted 1D likelihood array
    sorted_probability : ndarray
        difference between log probability and log target_probability.
    """
    sorted_array = np.sort(array2d.flatten())[:-1]

    sorted_probability = Parallel(num_cpu)(
        delayed(find_probability_difference)(i, array2d, target_probability)
        for i in sorted_array
    )
    sorted_probability = np.array(sorted_probability)
    return sorted_array, sorted_probability


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
    sorted_likelihood, sorted_probability = sampling_probability(
        array2d, num_cpu, target_probability
    )
    # the minimum distance corresponds to the initial guess
    initial_guess = sorted_likelihood[abs(sorted_probability).argmin()]

    # interpolation is preferred when the sample density is insufficient
    interp_probability = interp1d(sorted_likelihood, sorted_probability)
    result = fsolve(interp_probability, initial_guess)
    root_distance = interp_probability(result)
    if abs(root_distance) > 1e-8:
        raise ValueError("Cannot find the root: {}".format(root_distance))
    return result


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
    evidence = logsumexp(array2d)
    normalized_mass = np.exp(logsumexp(array2d, axis=0) - evidence)
    normalized_chi = np.exp(logsumexp(array2d, axis=1) - evidence)

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
    return Data(data_pad, index=tpad)
