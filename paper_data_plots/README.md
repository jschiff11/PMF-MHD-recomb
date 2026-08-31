# Paper data & plots

Figures for the paper, produced by the pipeline with helium recombination
physics included in the transfer-function stage (the perturbed-ionization/
Saha response terms remain hydrogen-only throughout, since helium's own
perturbed ionization isn't tracked).

Layout:
- `plots/` — the 20 final PNGs (tracked in git).
- `data/` — pipeline intermediate outputs (`.npy`/`.pkl`, ~150 GB), gitignored.
  This replaced an external, out-of-repo data directory; the whole pipeline
  is now self-contained under this one repo.
- `theta_converged/` — theta-refined angular averages used to correct
  `05_magnetic_power_spectrum.png` (see below), tracked in git (~370 KB).

## Theta-grid convergence in `05_magnetic_power_spectrum.png`

The pipeline angle-averages on a uniform 33-node grid over `[0, pi]`
(`theta_gridfull()`), i.e. 32 intervals with `d_theta = pi/32 = 5.625 deg`, and
the perpendicular direction `theta = pi/2` falls on an interior node.

At `theta = pi/2` the Alfven restoring force (proportional to `cos^2 theta`)
vanishes, so that mode stays undamped while all oblique angles have damped
away. The surviving wedge has angular width `~1/(k v_A)`, which at high `k` is
far narrower than `d_theta`. The quadrature — an interpolating cubic spline
(`splrep`/`splint`) — then assigns the spike the width of a grid interval
instead of its true width, so `bxbxbar` saturates at exactly
`d_theta = 9.81748e-2`. In `Delta_B^2(k)` this appeared as a flat high-`k`
plateau on which all redshifts collapsed onto a single line.

This was checked by re-solving the transfer functions on a refined grid that
keeps the 17 production nodes and adds 14 geometric bisections toward `pi/2`,
reaching `6e-6 rad`. Restricted to the production nodes it reproduces the
shipped arrays bitwise; on the full grid the plateau disappears, the spectrum
continues to decline, and the redshift ordering is preserved. The previously
plotted values were too high by ~2.5x (z=1900) to ~10.7x (z=1000), and up to
~20x for individual modes.

`plot_power_spectra_and_clumping.py` now overrides `bxbxbar`/`bybybar` for
kinds 0-18 at `B0 = 315 pG` (the only field strength this figure plots) with
`theta_converged/ang_avg_TLA_B315pG_converged.npz`. Kinds 19-68 are unaffected
— kind 18 agrees to 0.1-0.7% at every plotted redshift — and are used as
shipped. No truncation in `k` is applied.

Scope: the artifact is confined to the **Alfven** branch. Magnetosonic-sourced
quantities (`deltamdeltambar`, `xexebar` and their cross-correlations) converge
by `k ~ 2e4 Mpc^-1`, because the magnetosonic driving term (proportional to
`sin theta cos theta`) also vanishes at perpendicular and so produces no sharp
endpoint feature. Propagating the correction changes the clumping factor by
-0.41% to -0.69% (both recombination models, z = 1000-1295) and leaves
`06_clumping_power_spectrum.png` byte-identical.

## Figure index

| File | Script that produced it | Contents |
|---|---|---|
| `01_damping_length_vs_theta.png`–`04_fm_amplitude_ratio.png` | `plot_transfer_and_damping.py` | Damping length vs anisotropy angle, Stokes-parameter evolution, transfer-function (z,k) scatter, free-motion/acoustic amplitude ratio |
| `05_magnetic_power_spectrum.png`, `06_clumping_power_spectrum.png` | `plot_power_spectra_and_clumping.py` | Magnetic and matter-clumping power spectra |
| `07_xe2_curves.png`–`11_clumping_and_visibility_side_by_side.png` | `plot_xe2_and_visibility.py` | Ionization-perturbation curves/heatmaps and visibility-function shift |
| `12_lmhd_coupling_kernel.png`–`17_pctdiff_xe2_Tb.png`, `20_photon_bath_ydistortion.png` | `plot_baryon_heating.py` | Baryon-heating (T_b) coupling kernel and percent-difference diagnostics; Compton-y check if drag-dissipated KE heated the photon bath instead of being neglected |
| `18_lyman_alpha_moments.png`, `19_hyrec_saha_3la_psd_lineshape.png` | `plot_moments_and_lineshape.py` | Lyman-alpha moments and HyRec/Saha/3LA lineshape comparison (helium-independent) |

## What would need to be run to regenerate these

The plotting scripts live in this repo under `analysis/`, and everything they
depend on, physics-wise, is under `src/pmhd/`. The pipeline-stage scripts
(`src/pmhd/physics/*.py`) no longer carry a `_He` suffix either — helium
recombination is simply the default now. Regenerating from scratch means
running the SLURM drivers in `src/pmhd/physics/` in dependency order (each
one submits one job per bind, looping `kind`/`epsind` internally):

1. `slurmloop_repo.py all` — `TCR_Tfs.py` + `FSR_saha_Tfs.py` +
   `angle_avging_saha.py`, all 61 binds.
2. `slurmloop_TCR_FM.py` (default 7-bind subset `[0,10,...,60]`) — needed
   only for figure 04. `slurmloop_FSR_z4500.py` (default 3-bind/7-k subset)
   — needed only for figure 02. Both independent of the rest.
3. `slurmloop_TLA.py all` — `FSR_TLA_Tfs.py` + `FSR_TLA_Tfs_Tb.py`, all 61
   binds (depends on step 1's Saha-FSR ICs).
4. `slurmloop_ang_avg_downstream.py all` — `angle_avging_TLA.py` +
   `angle_avging_TLA_Tb.py` + `angle_avging_opt_depth.py`, all 61 binds.
5. `slurmloop_xe2_visib.py --binds all --epsinds 0-24` — `opt_depth.py`
   (once per bind) + `cont_source.py`/`cross_corr_and_source_fncs.py`/
   `xe2.py`/`visib_integ.py` per epsind, all 61 binds × 25 epsind.
6. `slurmloop_tb_xe2_visib.py` (no args — runs its hardcoded 13
   `(bind, epsind)` pairs) — `cross_corr_and_source_funcs_Tb.py` +
   `xe2_Tb.py` + `visib_integ_Tb.py`.

Each driver hardcodes `OUTDIR = "/home/jonschiff/PMF-MHD-recomb/paper_data_plots/data"`
and exports it as `PMHD_OUTDIR` to its submitted jobs, so no manual env
setup is needed to run them — but on a different clone/machine, edit that
`OUTDIR` line in each driver to your own absolute repo path first.

**Plotting** (5 scripts in `analysis/`): `plot_transfer_and_damping.py`,
`plot_power_spectra_and_clumping.py`, `plot_xe2_and_visibility.py`,
`plot_baryon_heating.py`, `plot_moments_and_lineshape.py`. Each loads
the saved arrays for the relevant `(bind, kind, epsind)` combinations and
produces the corresponding figures above, e.g.:
```
PMHD_OUTDIR=paper_data_plots/data PMHD_PLOTDIR=paper_data_plots/plots \
  python analysis/plot_xe2_and_visibility.py
```
`plot_moments_and_lineshape.py` needs no pipeline data at all — it
computes everything on the fly.

## Grid sizes used

- 61 `B0` values, but stages 2–3 only needed a 7-value subset
  `bind = 0, 10, 20, ..., 60`; stage 4 (ang_avg/xe2/visib/cross_corr) was run
  for all 61.
- 69 `k` values (all of `karr`).
- 25 of the 100 `eps` values (`epsind = 0..24`), except the baryon-heating
  figures, which only needed 13 specific `(bind, epsind)` pairs.
- 17 `theta` values.

## Practical notes

- Run via SLURM, one job per `bind` looping `kind`/`epsind` internally
  (avoids overwhelming the shared job-count limit). Each driver requests
  `--cpus-per-task=1` explicitly.
- Some integrations are extremely stiff at strong/weak field extremes —
  `TCR_Tfs_FM.py` and `FSR_TLA_Tfs_Tb.py` in particular are markedly slower
  for the strongest-field binds (low `bind` index).
- On a busy shared cluster, wall-clock time is often dominated by queue
  congestion and node contention from other users' jobs rather than the
  computation itself — check `squeue`/`sacct` before assuming a stage is
  stuck; per-job compute time for most binds is on the order of minutes.
- Total intermediate-data footprint when this was run: ~150 GB, now stored
  in-repo (gitignored) under `paper_data_plots/data/`.
