# QNM-Filter Demo: GWOSC vs Surrogate Reconstruction

This demo script (`demo.py`) compares real GWOSC strain data against an NRSur7dq4 surrogate reconstruction for the original and a quasinormal-mode–removed (QNM-filtered) version using `qnm_filter`. It automatically downloads required data files if they are not present locally and produces a figure (`plot.pdf`).

Please install the latest version of `qnm_filter` from the same repository that contains this README file:
```bash
pip install -e .
```

---

## 1) System requirements

### Operating systems
- Linux: Ubuntu 20.04 / 22.04 (recommended)
- macOS: 12 or later (Monterey, Ventura, Sonoma)
- Windows: WSL2 Ubuntu 22.04 recommended

### Python
- Python 3.10 – 3.12

### Required Python packages
- numpy >= 1.23
- scipy >= 1.9
- matplotlib >= 3.6
- h5py >= 3.8
- gwpy >= 3.0
- gwsurrogate >= 1.0
- qnm_filter (install from this repository)

### Hardware
- Standard laptop/desktop CPU is sufficient
- Internet connection required on first run to download GWOSC data and posterior file

### Tested environments
- Python 3.11 on macOS 15.6.1
- Python 3.10 on Ubuntu 22.04

---

## 2) Installation guide

### Steps

1. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

2. Install dependencies
```bash
pip install --upgrade pip
pip install numpy scipy matplotlib h5py gwpy gwsurrogate
pip install -e .    # install qnm_filter from this repository
```

3. Project layout (example)
```
your-project/
├── data/        # created automatically if missing
├── qnm_filter/  # local repository installed via pip -e .
└── demo.py
```

Typical install time: 5–15 minutes

---

## 3) Demo

### What happens in the demo
- Loads posterior samples and selects the max-likelihood parameters
- Reads real strain data from GWOSC
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

### Modify QNM mode list
```python
model_list = [(2,2,0,"p"), (2,2,1,"p"), (2,2,2,"p")]
```

### Re-run
```bash
python demo.py
```
