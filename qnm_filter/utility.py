"""Useful functions for calculating and plotting data
"""
__all__ = ['parallel_compute', 'least_square_tempate', 'linear_fit']

from joblib import Parallel, delayed
import matplotlib.pyplot as pl
import numpy as np
import qnm

def parallel_compute(self, M_arr, chi_arr, **kwargs):
    """Parallel computation of a function that takes 2 arguments 
    
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
    flatten_array = [(i,j) for i in M_arr for j in chi_arr]
    results = Parallel(-1)(delayed(self.likelihood_vs_mass_spin)(i, j, **kwargs) 
                     for i,j in flatten_array)
    reshaped_results = np.reshape(results, (len(M_arr), len(chi_arr))).T
    return reshaped_results

def least_square_tempate(a, mass, model_list, t):
    htot=[]
    htot+=[np.exp(0*t)]
    for (l, m, n) in model_list:
        grav = qnm.modes_cache(s=-2,l=l,m=m,n=n)
        omega, _, _ = grav(a=a)
        htot+=[np.exp(-1j*omega/mass*(t-t[0]))]
    return np.array(htot).T

def linear_fit(data, t0, a, mass, model_list):
    truncated_data = data.truncate_data(before=t0,after=80)
    
    tcut=truncated_data.time
    hcut=truncated_data.values
    
    datalist=least_square_tempate(a,mass,model_list,tcut)
    linearfit=np.linalg.lstsq(datalist, hcut,rcond=None)[0]
    return np.sum(datalist*linearfit,axis=1), linearfit, truncated_data