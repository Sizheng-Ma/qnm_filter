
# QNM-Filter Demo: GW250114 vs NR Surrogate Reconstruction

This demo script (`demo.py`) compares real GWOSC strain data against an NRSur7dq4 surrogate reconstruction and a **quasinormal-mode–removed** (QNM-filtered) version using `qnm_filter`. It auto-fetches required data files if they are not present locally and produces a figure (`plot.pdf`) with original and filtered views.

---

## 1) System requirements

### Operating systems
- **Linux:** Ubuntu 20.04 / 22.04 (recommended)
- **macOS:** 12+ (Monterey) / 13 (Ventura) / 14 (Sonoma)
- **Windows:** WSL2 Ubuntu 22.04 recommended

> The code is standard scientific Python; any recent Unix-like environment is fine.

### Python
- **Python 3.10 – 3.12** (recommended)

### Software dependencies (Python packages)
- `numpy >= 1.23`
- `scipy >= 1.9`
- `matplotlib >= 3.6`
- `h5py >= 3.8`
- `gwpy >= 3.0`
- `gwsurrogate >= 1.0`
- `qnm_filter` (domain-specific package; install from your internal source or PyPI if available)

> Notes:
> - `gwsurrogate` may pull extra numerical dependencies; a working compiler toolchain is helpful on Linux.
> - `qnm_filter` is not a standard library. Install it from your organization’s repo or PyPI if published.

### Non-standard hardware
- **None required.** A typical laptop/desktop CPU is sufficient.
- **Internet connection** needed on first run for data auto-download (two `.gwf` files and one posterior `.h5`).

### Tested environments
- Python **3.11** on macOS 14 (Sonoma)
- Python **3.10** on Ubuntu 22.04 (jammy)

---

## 2) Installation guide

### Instructions

1) **Create and activate a virtual environment**
```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

2) **Install dependencies**
```bash
pip install --upgrade pip
pip install numpy scipy matplotlib h5py gwpy gwsurrogate
# Install qnm_filter from your source, e.g.:
# pip install qnm_filter
# or
# pip install git+https://<your-internal-git>/qnm_filter.git
```

3) **Project layout**
```
your-project/
├── data/                                # created automatically if missing
└── demo.py
```

> `demo.py` will create `./data` if needed and auto-download required files into it on first run.

### Typical install time
- **5–15 minutes** on a normal desktop (faster if you already have SciPy stack cached).  
  Installing `gwsurrogate` from scratch may add a few minutes.

---

## 3) Demo

### What the demo does
- Loads posterior samples and extracts the **max-likelihood** sample.  
- Reads real GWOSC strain around the event window for **H1** and **L1**.  
- Builds an **NRSur7dq4** surrogate waveform at the max-L parameters.  
- Projects the surrogate into each interferometer’s antenna response.  
- Uses `qnm_filter` to condition data and remove QNMs.  
- Plots whitened, band-passed **original data** vs **QNM-removed** data, and overlays the surrogate reconstruction.

### First-run auto-downloads
If the following files are missing in `./data`, the script fetches them automatically:
- `H-H1_GWOSC_O4b3Disc_16KHZ_R1-1420877824-4096.gwf`  
- `L-L1_GWOSC_O4b3Disc_16KHZ_R1-1420877824-4096.gwf`  
- `posterior_samples_NRSur7dq4.h5`

### How to run
```bash
python demo.py
```

### Expected output
- A figure file **`plot.pdf`** in the project root containing two panels:
  - **Top:** Original whitened Hanford data overlaid with surrogate reconstruction  
  - **Bottom:** QNM-removed (filtered) Hanford data overlaid with surrogate reconstruction

### Expected run time (normal desktop)
- **Data download (first run only):** depends on your network. Each `.gwf` frame can be large; expect **5–30+ minutes** total on a typical home connection.  
- **Computation & plotting (after data are present):** roughly **1–3 minutes**.

---

## 4) Instructions for use (your own data)

You can adapt the script to your event/data in a few small edits near the top of `demo.py`.

### a) Point to your own frame files
Edit these constants to your local files/URLs:
```python
DATA_DIR = "./data"

GWOSC_GWF = {
    "H1": f"{DATA_DIR}/<your-H1-file>.gwf",
    "L1": f"{DATA_DIR}/<your-L1-file>.gwf",
}

GWOSC_URLS = {
    "H1": "https://your.server/path/to/<your-H1-file>.gwf",
    "L1": "https://your.server/path/to/<your-L1-file>.gwf",
}
GWOSC_CHANNEL = "{ifo}:GWOSC-16KHZ_R1_STRAIN"  # change if your channel differs
```
If you keep `GWOSC_URLS` updated, the script will auto-download your frames if they’re missing.

### b) Point to your posterior samples
Replace the default posterior path (and optionally its auto-download URL) if you have your own PE outputs:
```python
POSTERIOR_H5 = f"{DATA_DIR}/posterior_samples_NRSur7dq4.h5"
# If you also want auto-download behavior for your posterior file, update the URL
# inside ensure_posterior_present() accordingly.
```

### c) Adjust event timing windows
The script derives the event geocenter time from the posterior. If you need different analysis/PSD windows, modify in `gwosc_setup()`:
```python
event_start, event_end = (ctx.t_geo - 2.0, ctx.t_geo + 2.0)
noise_start, noise_end = (ctx.t_geo + 1.0, ctx.t_geo + 1.0 + float(psd_seconds))
```

### d) Adjust QNM model list / sample rate / bandpass
- QNM modes are configured here:
```python
model_list = [(2, 2, 0, "p"), (2, 2, 1, "p"), (2, 2, 2, "p")]
```
- Change conditioning/plotting sample rates and segment lengths in the `plotting_cfg` / `nrsur_cfg` dictionaries.
- Change the visualization band-pass in `whiten_and_bandpass()`:
```python
fmin: float = 50.0
fmax: float = 2000.0
```

### e) Running
Once edited, run:
```bash
python demo.py
```
Look for:
- **Auto-download logs** (first run)  
- **plot.pdf** output

### f) Troubleshooting
- **Missing `qnm_filter`:** install from your internal source (or PyPI if available).  
- **HDF5 / posterior errors:** ensure your posterior file contains the expected datasets and keys (the script guards for `final_mass` vs `final_mass_non_evolved`, etc.).  
- **Memory:** processing ~few-second windows typically fits in <1 GB RAM; if you scale up, monitor usage.  
- **Slow first run:** large `.gwf` files can take time to download; if possible, pre-stage them locally.
