"""Useful functions for calculating and plotting data
"""
__all__ = ["parallel_compute", "find_credible_region"]

from joblib import Parallel, delayed
import matplotlib.pyplot as pl
import numpy as np
from scipy.optimize import fsolve


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
    """Calculates the difference between the probability of sampling array2d above the threshold, and target_probability. Returns 0 if threshold is the target_probability credible interval

    Arguments
    ---------
    threshold : float
        value to consider the probability of sampling above
    array2d : array-like
        2D array of likelihood as a function of mass and spin
    target_probability : float
        function returns 0 if the probability of sampling above the level = target_probability
    """
    tot = np.sum(np.exp(array2d), axis=(0, 1))
    region = [x for index, x in np.ndenumerate(array2d) if x > threshold]
    region_tot = np.sum(np.exp(region))
    prob = region_tot / tot
    return prob - target_probability


def find_guess(array2d, target_probability):
    """Calculates an initial guess for the threshold of the target_probability credible interval by calculating the target_probability quartile value

    Arguments
    ---------
    array2d : array-like
        2D posterior of mass and spin
    target_probability : float
        target_probability credible interval threshold that we aim to approximate
    """
    array2d_normalised = (array2d - array2d.min()) / (array2d.max() - array2d.min())
    temp = np.abs(array2d_normalised - target_probability)
    index = np.unravel_index(temp.argmin(), temp.shape)
    return array2d[index]


def find_credible_region(
    array2d, target_probability=0.9, max_guesses=5, guess_step=0.05
):
    """Calculates the threshold of the target_probability credible region

    Arguments
    ---------
    array2d : array-like
        2D posterior of mass and spin
    percentage : float
        credible interval probability that we aim to find
    """
    guess_no = 0
    while guess_no < max_guesses:
        initial_guess = find_guess(array2d, target_probability - guess_step * guess_no)
        res = fsolve(
            find_probability_difference,
            initial_guess,
            args=(array2d, target_probability),
        )
        if abs(find_probability_difference(res, array2d, target_probability)) > 5e-3:
            guess_no += 1
        else:
            return res
    raise Exception("Wrong", res)
