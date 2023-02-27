"""Useful functions for calculating and plotting data
"""
__all__ = ['parallel_compute', 'plotter']

from joblib import Parallel, delayed
import matplotlib.pyplot as pl
import numpy as np

def parallel_compute(function, M_arr, chi_arr, **kwargs):
    """Parallel computation of a function that takes 2 arguments 
    
    Arguments
    ---------
    function : function
        function that takes 2 args and an arbitrary number of kwargs
    M_arr : array-like
        array of the values of the first arg of the function
    chi_arr : array-like
        array of the values of the second arg of the function
    kwargs : dict
        dictionary of kwargs of the function
        
    Returns
    ---------
    reshaped_results : ndarray
        2d array of the results with shape (len(x_arr), len(y_arr))
    """
    flatten_array = [(i,j) for i in M_arr for j in chi_arr]
    results = Parallel(-1)(delayed(function)(i, j, **kwargs) 
                     for i,j in flatten_array)
    reshaped_results = np.reshape(results, (len(M_arr), len(chi_arr))).T
    return reshaped_results

def plotter(X_grid, Y_grid, results, IMR_result = (np.nan, np.nan), credible_region=None, \
            show = True, save_fig = False, fig_size = (6.1, 6.6), peak_time = 1126259462.4083147, \
            **kwarg_dict):
    pl.rc('figure', figsize=fig_size)
    fig, ax = pl.subplots()
    contours = ax.contourf(X_grid, Y_grid, results, 20, cmap='Spectral',
                           origin='lower', alpha=1.0, linestyles='--')
    ax.scatter(x=IMR_result[0], y=IMR_result[1], s=255, marker='+', 
               c='white', linewidths=4, label='IMR')
    ax.axvline(x=IMR_result[0], c='k', ls = '--', alpha = 0.4, linewidth = 2)
    ax.axhline(y=IMR_result[1], c='k', ls = '--', alpha = 0.4, linewidth = 2)
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

    time_str = str(np.round((kwarg_dict["t_init"] - peak_time)*1e3, 2))
    title = r'Filters = '+ str(kwarg_dict["model_list"]) + '  $\Delta t_0 = $' + time_str + " ms"
    ax.set_title(title);
    
    if save_fig:
        pl.savefig("Figures/"+title+".png");
    if not show:
        pl.close()