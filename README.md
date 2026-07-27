# PMF-MHD-recomb

Numerical pipeline for computing the imprint of primordial magnetic fields (PMFs) on the CMB via perturbations to the recombination history by coupling linearized magnetohydrodynamics (LMHD) to non-local perturbed radiative transfer recombination. Given a magnetic field strength B₀ and spectral index ε, the code propagates MHD perturbations through the tight-coupling and free-streaming regimes, computes the resulting second-order ionization perturbation δxe², and produces the visibility function prefactor used in CMB power spectrum calculations.

Companion code to arXiv:2506.16517 (see [Citation](#citation)).

---

## Physics overview

The pipeline models how a stochastic PMF with power spectrum P_B ∝ k^(ε-3) perturbs the photon-baryon fluid prior to and through recombination (z ~ 600–2000). It solves:

1. **MHD transfer functions** — Alfvén and magnetosonic mode ODEs in the tight-coupling regime (TCR) and free-streaming regime (FSR), for both Saha and 3-level-atom (TLA) ionization histories.
2. **Angular averaging** — Integrates over magnetic field orientations (θ ∈ [0, π/2]) to produce isotropically averaged transfer functions.
3. **Boltzmann solver** — Perturbed Lyman-alpha photon distribution incorporating radiative transfer.
4. **Correlation and source functions** — Builds the full two-point statistics of the fluid perturbations.
5. **Second-order ionization** — Solves the ODE for δxe²(z) driven by the PMF source terms.
6. **Visibility function** — Integrates the second-order optical depth and constructs visibility prefactor which modifies the homogeneous visiblity function.

---

## Pipeline stages

The computation proceeds in the following order; each stage reads outputs from the previous one. Every script takes plain positional arguments — a magnetic-field-strength index `bind` and either a wavenumber index `kind` or spectral-index `epsind` — and is designed to be run in parallel across the grid (e.g. via the `slurmloop_*.py` drivers described in `paper_data_plots/README.md`). Output paths are controlled by the `PMHD_OUTDIR` environment variable, read by every script (default: `data/outputs/` relative to the repo root; the paper's own figures are regenerated with `PMHD_OUTDIR=paper_data_plots/data`).

| Stage | Args | Scripts | Outputs (under `$PMHD_OUTDIR/`) |
|-------|------|---------|---------|
| **TFS** | `bind kind` | `TCR_Tfs.py`, `FSR_saha_Tfs.py`, `FSR_TLA_Tfs.py`, `TCR_Tfs_FM.py` | `Tfs/` |
| **ANGAVG** | `bind kind` | `angle_avging_saha.py`, `angle_avging_TLA.py`, `angle_avging_opt_depth.py` | `ang_avg/` |
| **CORRS** | `bind epsind` | `cross_corr_and_source_fncs.py`, `cont_source.py` | `cross_corr/`, `source_fncs/` |
| **XE2** | `bind epsind` | `xe2.py` | `xe2/` |
| **VISIB** | `bind` / `bind epsind` | `opt_depth.py`, `visib_integ.py` | `visib/` |

**Baryon heating (T_b) variant** — a parallel set of stages coupling a baryon-temperature perturbation into the ODEs, run alongside (not instead of) the ones above (they also depend on ANGAVG/CORRS outputs from the main pipeline):

| Stage | Args | Scripts | Outputs |
|-------|------|---------|---------|
| **TFS (Tb)** | `bind kind` | `FSR_TLA_Tfs_Tb.py` | `Tfs/` |
| **ANGAVG (Tb)** | `bind kind` | `angle_avging_TLA_Tb.py` | `ang_avg/Tb/` |
| **CORRS+XE2 (Tb)** | `bind epsind` | `cross_corr_and_source_funcs_Tb.py`, `xe2_Tb.py` | `cross_corr/`, `source_fncs/`, `xe2/` |
| **VISIB (Tb)** | `bind epsind` | `visib_integ_Tb.py` | `visib/` |

---

## Installation

**Requirements:** Python ≥ 3.9. [mamba](https://mamba.readthedocs.io) or [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) are strongly recommended over plain conda — environment creation is typically 5–10× faster.

### With mamba (recommended)

```bash
# Install mamba into your base conda environment (once)
conda install -n base -c conda-forge mamba

# Create the environment and activate it
mamba env create -f environment.yml
conda activate pmhd

# Install the package in editable mode
pip install -e .
```

### With micromamba (recommended for fresh installs)

```bash
# Install micromamba — see https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html
# Then:
micromamba env create -f environment.yml
micromamba activate pmhd

pip install -e .
```

### With plain conda (slower)

```bash
conda env create -f environment.yml
conda activate pmhd
pip install -e .
```

**Dependencies:** `numpy`, `scipy`, `astropy`, `matplotlib`

---

## Running the pipeline

Scripts can be run locally or submitted to a SLURM cluster. Each physics script is self-contained and accepts command-line arguments.

### Local (single parameter point)

Scripts read `$PMHD_OUTDIR` for where to write/read stage outputs (default `data/outputs/` at the repo root):

```bash
export PYTHONPATH=src
export PMHD_OUTDIR=data/outputs   # or e.g. paper_data_plots/data

# Tight-coupling transfer functions — bind 40, k-mode 30
python src/pmhd/physics/TCR_Tfs.py 40 30

# Free-streaming (TLA) — bind 40, k-mode 30
python src/pmhd/physics/FSR_TLA_Tfs.py 40 30

# Angular averaging (TLA) — bind 40, k-mode 30
python src/pmhd/physics/angle_avging_TLA.py 40 30

# Second-order ionization — bind 40, spectral index 9
python src/pmhd/physics/xe2.py 40 9

# Visibility — bind 40, spectral index 9
python src/pmhd/physics/opt_depth.py 40
python src/pmhd/physics/visib_integ.py 40 9
```

For regenerating the full paper dataset and figures across the whole (bind, kind, epsind) grid via SLURM, see `paper_data_plots/README.md`.

---

## Repository structure

```
PMF-MHD-recomb/
├── src/pmhd/
│   ├── cons.py                    # Physical and cosmological constants (Planck18)
│   ├── pars.py                    # Core physics: ionization histories, ODEs, profiles
│   ├── data/
│   │   ├── grids.py               # Grid generators (k, z, θ, B₀, ε)
│   │   └── pre_stored_data/       # Pre-computed Boltzmann outputs (f₂bars, Zenodo)
│   └── physics/
│       ├── TCR_Tfs.py             # Tight-coupling transfer functions
│       ├── FSR_saha_Tfs.py        # Free-streaming TFs (Saha ionization)
│       ├── FSR_TLA_Tfs.py         # Free-streaming TFs (3-level atom)
│       ├── TCR_Tfs_FM.py          # TCR with Faraday mixing
│       ├── angle_avging_saha.py   # Angular averaging (Saha)
│       ├── angle_avging_TLA.py    # Angular averaging (TLA)
│       ├── angle_avging_opt_depth.py
│       ├── inhomo_moments.py      # Perturbed Boltzmann moments
│       ├── firstOmoments_and_secondO_soln_PRD_fullk.py
│       ├── hompsd.py              # Homogeneous phase-space density
│       ├── cross_corr_and_source_fncs.py
│       ├── cont_source.py         # Continuum source terms
│       ├── xe2.py                 # Second-order ionization δxe²
│       ├── opt_depth.py           # Optical depth
│       ├── visib_integ.py         # Visibility function
│       └── *_Tb.py                # Baryon-heating (T_b) variants of the above stages
├── analysis/
│   ├── plot_transfer_and_damping.py
│   ├── plot_power_spectra_and_clumping.py
│   ├── plot_xe2_and_visibility.py
│   ├── plot_baryon_heating.py
│   └── plot_moments_and_lineshape.py
├── paper_data_plots/
│   ├── plots/                     # The paper's 19 figures (tracked in git)
│   ├── data/                      # Full-grid pipeline outputs for the paper (gitignored, ~150GB)
│   └── README.md                  # SLURM-driven regeneration workflow for the paper dataset/figures
├── data/
│   └── outputs/                   # Default stage-output location (gitignored; used when PMHD_OUTDIR is unset)
├── tests/
├── environment.yml
└── pyproject.toml
```

---

## Pre-stored data

The `src/pmhd/data/pre_stored_data/` directory contains pre-computed Boltzmann solver outputs (`f2bars` dictionaries) that are required by the correlation stage. These are already tracked in this repository (~46 MB).

To reproduce them from scratch instead, run `firstOmoments_and_secondO_soln_PRD_fullk.py` first.

---

## Citation

If you use this code, please cite:

```bibtex
@article{schiff2025primordialmagneticfieldsmodified,
      title={Primordial magnetic fields and modified recombination histories}, 
      author={Jonathan Schiff and Tejaswi Venumadhav},
      year={2025},
      eprint={2506.16517},
      archivePrefix={arXiv},
      primaryClass={astro-ph.CO},
      url={https://arxiv.org/abs/2506.16517}, 
}
```
