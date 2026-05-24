# QNM Rational Filter Demo: GWOSC vs Numerical Relativity Reconstruction

This demo script (`demo.py`) compares real GWOSC strain data against an NRSur7dq4 surrogate reconstruction for the original and a quasinormal-mode–removed (QNM-filtered) versions using `qnm_filter`. It automatically downloads required data files if they are not present locally and produces a figure (`plot.pdf`).

Please install the latest version of `qnm_filter` from the same repository that contains this README file. See installation guide below.

---

## 1) System requirements

### Operating systems
- Linux: Ubuntu 22.04 / 24.04 (recommended)
- macOS: 12 or later (Monterey, Ventura, Sonoma)
- Windows: WSL2 Ubuntu 22.04 recommended

### Python
- Python 3.10 – 3.11

### Required Python packages
- numpy = < 2.0
- scipy < 1.15.0
- matplotlib >= 3.6
- h5py (set by Python 3.11)
- gwpy (set by Python 3.11)
- gwsurrogate (set by Python 3.11)
- qnm_filter (install the latest version from this repository)

### Hardware
- Standard laptop/desktop CPU is sufficient
- Internet connection required on first run to download GWOSC data and posterior file

### Tested environments
- Python 3.11 on macOS 15.6.1
- Python 3.11 on Ubuntu 24.04

---

## 2) Installation guide

### Steps

1. Create / activate a conda environment and install dependencies
```bash
conda config --add channels conda-forge
conda create --name myenv python=3.11 pip gwpy gwsurrogate
conda activate myenv
```

2. Install qnm_filter
```bash
pip install -e .    # install qnm_filter from this repository
```

3. Demo project layout (example)
```
code_demo/GWevent/
├── data/        # created automatically if missing
└── demo.py
```

Typical install time: 5–20 minutes

---

## 3) Demo

### What happens in the demo
- Reads real strain data from GWOSC (GW250114 as an example)
- Loads posterior samples and selects the max-likelihood parameters
- Builds a surrogate waveform using NRSur7dq4
- Projects waveform to detector antenna patterns
- Applies QNM filtering to real data and the surrogate reconstruction
- Produces a comparison plot of original vs filtered waveforms (H1 only as an example)

### Auto-downloaded data (first run only)
Saved in `./data`:
- H-H1_GWOSC_O4b3Disc_16KHZ_R1-1420877824-4096.gwf
- L-L1_GWOSC_O4b3Disc_16KHZ_R1-1420877824-4096.gwf
- posterior_samples_NRSur7dq4.h5

### Run the demo
```bash
python demo.py
```

### Output
- File: `plot.pdf` in the project root

### Expected runtime
- First run (download included): 5–15 minutes depending on network
- Subsequent runs: 1–3 minutes

---

## 4) Instructions for use on your own data

### Modify GWOSC data files and channel
Edit at the top of `demo.py`:
```python
DATA_DIR = "./data"
GWOSC_GWF = {
    "H1": "your_H1_frame.gwf",
    "L1": "your_L1_frame.gwf",
}
GWOSC_URLS = {
    "H1": "http://path/to/your/H1_file.gwf",
    "L1": "http://path/to/your/L1_file.gwf",
}
GWOSC_CHANNEL = "{ifo}:GWOSC-16KHZ_R1_STRAIN"  # update if needed
```

### Use your own posterior file
```python
POSTERIOR_H5 = f"{DATA_DIR}/your_posterior.h5"
```

### Adjust analysis window durations
Edit inside `gwosc_setup()`:
```python
event_start, event_end = (ctx.t_geo - 2.0, ctx.t_geo + 2.0)
noise_start, noise_end = (ctx.t_geo + 1.0, ctx.t_geo + 1.0 + psd_seconds)
```

### Modify detector
You can switch detectors by editing a single line in `demo.py`:
```python
ifo = "H1"
```

### Modify QNM mode list
```python
model_list = [(2,2,0,"p"), (2,2,1,"p"), (2,2,2,"p")]
```

### Re-run
```bash
python demo.py
```
