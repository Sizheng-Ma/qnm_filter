"""Useful functions for calculating and plotting data
"""
__all__ = ['parallel_compute', 'plotter']

from joblib import Parallel, delayed
import matplotlib.pyplot as pl
import numpy as np

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

def plotter(mass_grid, spin_grid, results, credible_region=None,
            **plotter_dict):
    """Generating a contour plot of the likelihood vs mass and spin
    
    Arguments
    ----------
    mass_grid : 2d array-like
        2d array of the mass values, generated via np.meshgrid
    spin_grid : 2d array-like
        2d array of the spin values, generated via np.meshgrid
    results : 2d array-like
        2d array of the computed likelihood values
    credible_region : float
        contour value which defines the credible region
    plotter_dict : dict
        dictionary of the kwargs of the function, notably can 
        include show_fig, save_fig, fig_size, IMR_result
    
    """
    pl.rc('figure', figsize=plotter_dict.get("fig_size"))
    fig, ax = pl.subplots()
    contours = ax.contourf(mass_grid, spin_grid, results, 20, cmap='Spectral',
                           origin='lower', alpha=1.0, linestyles='--')
    ax.grid(visible = True, alpha = 0.5)
    
    # colorbar
    cbar=fig.colorbar(contours, orientation="horizontal", 
                      pad=0.15, format='%3.2f')
    cbar.set_label(r'$\log_{10}$ likelihood', fontsize=15)
    cbar.set_ticks(np.linspace(np.min(results), np.max(results), 5))

    #formatting
    pl.xlabel(r'$M_f$', fontsize=13)
    pl.ylabel(r'$\chi_f$', fontsize=13)
    ax.set_box_aspect(1)

    title = r'Filters = '+ str(plotter_dict["model_list"]) + \
    '  $\Delta t_0 = $' + str(plotter_dict["delta_t"]*1e3) + " ms"
    ax.set_title(title);
    
    #optional elements
    if plotter_dict.get("IMR_result"):
        IMR_result = plotter_dict.get("IMR_result")
        ax.scatter(x=IMR_result[0], y=IMR_result[1], s=255, marker='+', 
                   c='white', linewidths=4, label='IMR')
        ax.axvline(x=IMR_result[0], c='k', ls = '--', alpha = 0.4, linewidth = 2)
        ax.axhline(y=IMR_result[1], c='k', ls = '--', alpha = 0.4, linewidth = 2)
    
    if not credible_region == None:
        dotted = ax.contour(X, Y, results, [credible_region], colors = 'red', \
                       linestyles ='--')
    
    if plotter_dict.get("save_fig"):
        pl.savefig("Figures/"+title+".png");
    if not plotter_dict.get("show_fig"):
        pl.close()