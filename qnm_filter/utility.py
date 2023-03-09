"""Useful functions for calculating and plotting data
"""
__all__ = ["parallel_compute", "find_credible_region"]

from joblib import Parallel, delayed
import matplotlib.pyplot as pl
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import fsolve
from scipy.interpolate import interp1d


def parallel_compute(self, M_arr, chi_arr, **kwargs):
    """Parallel computation of the likelihood as a function of mass and chi

    Arguments
    ---------
    self : Network class instance
        An instance of a Network class that will have self.likelihood_vs_mass_spin computed
    M_arr : array-like
        array of the values of remnant mass to calculate the likelihood function for
    chi_arr : array-like
        array of the values of remnant spin to calculate the likelihood function for
    kwargs : dict
        dictionary of kwargs of the function

    Returns
    ---------
    reshaped_results : ndarray
        2d array of the results with shape (len(x_arr), len(y_arr))
    """
    flatten_array = [(i, j) for i in M_arr for j in chi_arr]
    results = Parallel(-1)(
        delayed(self.likelihood_vs_mass_spin)(i, j, **kwargs) for i, j in flatten_array
    )
    reshaped_results = np.reshape(results, (len(M_arr), len(chi_arr))).T
    return reshaped_results


def find_probability_difference(threshold, array2d, target_probability=0.9):
    """Calculates the difference between the log probability of sampling array2d above the threshold and log target_probability

    Parameters
    ----------
    threshold : float
        value to consider the probability of sampling above
    array2d : ndarray
        2D array of sampling likelihood as a function of mass and spin
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
        2D array of sampling likelihood as a function of mass and spin
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
        2D array of sampling likelihood as a function of mass and spin
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
