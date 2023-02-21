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

def plotter(X, Y, results, credible_region=None, show = True, save_fig = False, **kwarg_dict):
    pl.rc('figure', figsize=(6.1, 6.6))
    fig, ax = pl.subplots()
    contours = ax.contourf(X, Y, results, 20, cmap='Spectral',
                           origin='lower', alpha=1.0, linestyles='--')
    ax.scatter(x=68.5, y=0.69, s=255, marker='+', 
               c='white', linewidths=4, label='IMR')
    ax.axvline(x=68.5, c='k', ls = '--', alpha = 0.4, linewidth = 2)
    ax.axhline(y=0.69, c='k', ls = '--', alpha = 0.4, linewidth = 2)
    ax.grid(visible = True, alpha = 0.5)
    
    if not credible_region == None:
        dotted = ax.contour(X, Y, results, [credible_region], colors = 'red', \
                       linestyles ='--')

    # coloarbar
    cbar=fig.colorbar(contours, orientation="horizontal", 
                      pad=0.15, format='%3.2f')
    cbar.set_label(r'$\log_{10}$ likelihood', fontsize=15)
    cbar.set_ticks(np.linspace(np.min(results), np.max(results), 5))

    pl.xlabel(r'$M_f$', fontsize=13)
    pl.ylabel(r'$\chi_f$', fontsize=13)
    ax.set_box_aspect(1)

    time_str = str(np.round((kwarg_dict["t_init"] - 1126259462.4083147)*1e3, 2))
    title = r'Filters = '+ str(kwarg_dict["model_list"]) + '  $\Delta t_0 = $' + time_str
    ax.set_title(title);
    
    if save_fig:
        pl.savefig("Figures/"+title+".png");
    if not show:
        pl.close()