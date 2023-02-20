#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 17:55:14 2023

@author: neil
"""


from joblib import Parallel, delayed
import matplotlib.pyplot as pl
import numpy as np
import time

def parallel_compute(function, x_arr, y_arr, kwargs = {}, show_time = True):
    """Parallel computation of a function that takes 2 arguments 
    
    Arguments
    ---------
    function : function
        function that takes 2 args and an arbitrary number of kwargs
    x_arr : array-like
        array of the values of the first arg of the function
    y_arr : array-like
        array of the values of the second arg of the function
    kwargs : dict
        dictionary of kwargs of the function
    show_time : bool
        whether to print time (in minutes) for the calculation
        
    Returns
    ---------
    reshaped_results : ndarray
        2d array of the results with shape (len(x_arr), len(y_arr))
    """
    tic = time.time()
    flatten_array = [(i,j) for i in x_arr for j in y_arr]
    results = Parallel(-1)(delayed(function)(i, j, **kwargs) 
                     for i,j in flatten_array)
    reshaped_results = np.reshape(results, (len(x_arr), len(y_arr))).T
    toc = time.time()
    if show_time:
        calc_time = (toc-tic)/60
        print("Calculation time = " + str(np.round(calc_time, 2)) + "mins")
    return reshaped_results