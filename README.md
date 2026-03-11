# PMF-MHD-recomb

Numerical pipeline for computing the imprint of primordial magnetic fields (PMFs) on the CMB via perturbations to the recombination history by coupling linearized magnetohydrodynamics (LMHD) to non-local perturbed radiative transfer recombination. Given a magnetic field strength B₀ and spectral index ε, the code propagates MHD perturbations through the tight-coupling and free-streaming regimes, computes the resulting second-order ionization perturbation δxe², and produces the visibility function prefactor used in CMB power spectrum calculations.

Companion code to: *[citation placeholder — arXiv:2506.16517]*

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

The computation proceeds in the following order. Each stage reads outputs from the previous stage.

| Stage | Scripts | Outputs |
|-------|---------|---------|
| **TFS** | `TCR_Tfs.py`, `FSR_saha_Tfs.py`, `FSR_TLA_Tfs.py`, `TCR_Tfs_FM.py` | `data/outputs/Tfs/` |
| **ANGAVG** | `angle_avging_saha.py`, `angle_avging_TLA.py`, `angle_avging_opt_depth.py` | `data/outputs/ang_avg/` |
| **CORRS** | `corrs_corr_and_source_fncs.py`, `cont_source.py` | `data/outputs/cross_corr/`, `data/outputs/source_fncs/` |
| **XE2** | `xe2.py` | `data/outputs/xe2/` |
| **VISIB** | `opt_depth.py`, `visib_integ.py` | `data/outputs/visib/` |

Each script takes `--bind` (magnetic field index) and `--kind` or `--epsind` (wavenumber / spectral index) arguments and is designed to be run in parallel across the grid.

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

**Dependencies:** `numpy`, `scipy`, `astropy`, `matplotlib`, `numdifftools`

---

## Running the pipeline

Scripts can be run locally or submitted to a SLURM cluster. Each physics script is self-contained and accepts command-line arguments.

### Local (single parameter point)

```bash
# Tight-coupling transfer functions — bind 40, k-mode 30
python src/pmhd/physics/TCR_Tfs.py --bind 40 --kind 30

# Free-streaming (TLA) — bind 40, k-mode 30
python src/pmhd/physics/FSR_TLA_Tfs.py --bind 40 --kind 30

# Angular averaging (TLA) — bind 40, k-mode 30
python src/pmhd/physics/angle_avging_TLA.py --bind 40 --kind 30

# Second-order ionization — bind 40, spectral index 9
python src/pmhd/physics/xe2.py --bind 40 --epsind 9

# Visibility — bind 40, spectral index 9
python src/pmhd/physics/visib_integ.py --bind 40 --epsind 9
```

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
│       ├── corrs_corr_and_source_fncs.py
│       ├── cont_source.py         # Continuum source terms
│       ├── xe2.py                 # Second-order ionization δxe²
│       ├── opt_depth.py           # Optical depth
│       └── visib_integ.py         # Visibility function
├── data/
│   └── outputs/                   # Stage outputs (gitignored)
├── analysis/
│   ├── baseline_tests.ipynb       # Baseline validation notebook
│   └── sample.ipynb               # Sample analysis
├── environment.yml
└── pyproject.toml
```

---

## Pre-stored data

The `src/pmhd/data/pre_stored_data/` directory contains pre-computed Boltzmann solver outputs (`f2bars` dictionaries) that are required by the correlation stage. These are large files archived on Zenodo at: *[Zenodo DOI placeholder]*.

To reproduce them from scratch, run `firstOmoments_and_secondO_soln_PRD_fullk.py` first.

---

## Citation

If you use this code, please cite:

```bibtex
@article{placeholder,
  author  = {Schiff, Jonathan},
  title   = {...},
  journal = {...},
  year    = {2025},
  eprint  = {2506.16517},
  archivePrefix = {arXiv}
}
```
