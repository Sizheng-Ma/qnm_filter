# qnm_filter

Ringdown analysis with rational filters. The properties of the rational filters can be found in [Ma, et al. (2023a)](https://doi.org/10.1103/PhysRevLett.130.141401), [Ma, et al. (2023b)](https://doi.org/10.1103/PhysRevD.107.084010), and [Ma, et al. (2022)](https://doi.org/10.1103/PhysRevD.106.084036).

## Installation

Install this Python package in "editable mode":

```shell
git clone git@github.com:Sizheng-Ma/qnm_filter.git
cd qnm_filter
conda create --name qnm_filter python=3.11
conda activate qnm_filter
pip install -e .
```

## Examples

1. The analysis of GW150914 can be found in [code_demo/Filter.ipynb](code_demo/Filter.ipynb).
2. Two injection systems with a numerical-relativity waveform (GW150914-like) are available at [code_demo/NRInjection_bilby.ipynb](code_demo/NRInjection_bilby.ipynb) and [code_demo/NRInjection_customized_noise.ipynb](code_demo/NRInjection_customized_noise.ipynb).
3. The evidence curve is a useful tool for determining the statistical significance of a quasinormal mode and for limiting its start time. An example is at [code_demo/Evidence.ipynb](code_demo/Evidence.ipynb).

## Cite this code

Please cite these three papers if you use this code for academic research:

- Black Hole Spectroscopy by Mode Cleaning, Ma et al., [Phys. Rev. Lett. 130, 141401 (2023)](https://doi.org/10.1103/PhysRevLett.130.141401).
- Using rational filters to uncover the first ringdown overtone in GW150914, Ma et al., [Phys. Rev. D 107, 084010 (2023)](https://doi.org/10.1103/PhysRevD.107.084010).
- Quasinormal-mode filters: A new approach to analyze the gravitational-wave ringdown of binary black-hole mergers, Ma et al., [Phys. Rev. D 106, 084036 (2022)](https://doi.org/10.1103/PhysRevD.106.084036).

## Documentation

The documentation can be viewed at [here](https://sizheng-ma.github.io/qnm_filter/html/index.html).
