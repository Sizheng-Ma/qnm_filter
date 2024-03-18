#!/usr/bin/env python
# coding: utf-8
# %%


import numpy as np
from scipy.interpolate import interp1d
import qnm_filter
import sys
import qnm
import random
import argparse
from scipy.special import logsumexp
from pathlib import Path

argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--filename")

args = argParser.parse_args()

print("%s" % args.filename)


sampling_frequency = 4096 * 1  # in Hz
duration = 8  # in second

mass_in_solar = random.uniform(70, 120)
chi_inject = random.uniform(0.0, 0.9)
mass = qnm_filter.Filter.mass_unit(mass_in_solar)  # in solar mass
omega220, _, _ = qnm.modes_cache(s=-2, l=2, m=2, n=0)(a=chi_inject)
omega221, _, _ = qnm.modes_cache(s=-2, l=2, m=2, n=1)(a=chi_inject)
omega320, _, _ = qnm.modes_cache(s=-2, l=3, m=2, n=0)(a=chi_inject)
model_list = [(2, 2, 0, "p"), (2, 2, 1, "p")]

credibility220_list = []
snr_list = []
mmax = 8.4 * 1e-21
phase1 = random.uniform(0, 2 * np.pi)
phase2 = random.uniform(0, 2 * np.pi)
# phase3 = random.uniform(0, 2 * np.pi)

bilby_ifo = qnm_filter.set_bilby_predefined_ifo(
    "H1", sampling_frequency, duration, start_time=-duration / 2
)
signalH_noise = qnm_filter.bilby_get_strain(bilby_ifo, 0.0)

t_range = np.arange(-duration / 2, duration / 2, 1 / sampling_frequency)
N = 5
results = np.empty((N, 5))

for idx, val in enumerate(np.random.uniform(0.02, 0.5, N)):
    print(idx)

    A220x = mmax * np.cos(phase1) * val
    A220y = mmax * np.sin(phase1) * val

    A221x = mmax * np.cos(phase2) * val
    A221y = mmax * np.sin(phase2) * val

#    A320x = mmax * np.cos(phase3) * i
#    A320y = mmax * np.sin(phase3) * i

    signal = t_range * 0

    signal = np.real(
        np.exp(-1j * omega220 * np.abs(t_range / mass)) * (A220x + 1j * A220y)
        + np.exp(-1j * omega221 * np.abs(t_range / mass)) * (A221x + 1j * A221y)
#        + np.exp(-1j * omega320 * np.abs(t_range / mass)) * (A320x + 1j * A320y)
    )

    signalH_no_noise = qnm_filter.RealData(signal, index=t_range, ifo="H1")
    signalH = signalH_no_noise + signalH_noise

    fit = qnm_filter.Network(segment_length=0.2, srate=4096 * 1, t_init=3.0 * mass)

    fit.original_data["H1"] = signalH
    fit.detector_alignment()

    fit.pure_noise = {}
    fit.pure_noise["H1"] = signalH_noise

    fit.pure_nr = {}
    fit.pure_nr["H1"] = signalH_no_noise

    fit.condition_data("original_data")
    fit.condition_data("pure_noise")
    fit.condition_data("pure_nr")
    fit.compute_acfs("pure_noise")

    fit.cholesky_decomposition()

    delta_chi = 0.5
    delta_mass = 2.0
    massspace = np.arange(45, 150, delta_mass)
    chispace = np.arange(0.0, 0.95, delta_chi)
    mass_grid, chi_grid = np.meshgrid(massspace, chispace)

    fit.first_index()

    _, evidence220 = qnm_filter.parallel_compute(
        fit,
        massspace,
        chispace,
        num_cpu=-1,
        model_list=[(2, 2, 0, "p")],
    )
    
    _, evidence221 = qnm_filter.parallel_compute(
        fit,
        massspace,
        chispace,
        num_cpu=-1,
        model_list=[(2, 2, 1, "p")],
    )
    
    _, evidence220_221 = qnm_filter.parallel_compute(
        fit,
        massspace,
        chispace,
        num_cpu=-1,
        model_list=[(2, 2, 0, "p"),(2,2,1,'p')],
    )

    nofilter = logsumexp(
        np.array(
            [fit.compute_likelihood(apply_filter=False)]
            * len(massspace)
            * len(chispace)
        )
    )

    snr = fit.compute_SNR(
        fit.truncate_data(fit.original_data)["H1"],
        fit.truncate_data(fit.pure_nr)["H1"],
        "H1",
        False,
    )

    results[idx] = np.array([snr, nofilter, evidence220, evidence221, evidence220_221])

home_dir = 'home/neil.lu/Aspects_of_rational_filter/Sizheng_script/'
#np.savetxt(
#    "home_dir" + args.filename + ".dat",
#    np.ravel(results),
#)

#np.savetxt(
#    str(Path().absolute()) + args.filename + "pathlib.dat",
#    np.ravel(results),
#)

# %%
