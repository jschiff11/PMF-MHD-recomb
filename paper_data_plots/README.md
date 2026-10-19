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
- `doppler_nrbc/` — the 2026-09 Doppler/aberration + non-reflecting-closure update: pre-registered acceptance
  bars and addenda (`PREREG.md`), the verification record (`VERIFICATION.md`) and its comparison figures.

## 2026-09 update: Doppler/aberration terms and non-reflecting closure

The figures in `plots/` (except `20_photon_bath_ydistortion.png`, kept as published; see
`doppler_nrbc/VERIFICATION.md`) include:
- the Doppler factor multiplying the collision term, which gives an l=1 source in the first-order line-transfer
  bases (B, D+-);
- the rest-frame-monopole terms of the second-order source (`fdop`, `fdopbar`, `fv2` in `ftot`);
- a non-reflecting closure at l_max = 14 in all three moment hierarchies. Previously the m = 0 tower used the
  non-reflecting closure at l_max = 8 and the m = +-1 towers used hard truncation at l_max = 29.

The moment tables in `src/pmhd/data/pre_stored_data/` are the rebuilt ones. Effect on the figures: |Delta x_e|
changes by <= 0.13% of its peak, largest at 5000 pG; the visibility-peak shift by <= 2.2e-3 in z; clumping by <= 1e-5.
Also included:
- a step cap in the x_1s^(2) ODE solve (`solve_x1squad`, `xe2_Tb.py`), which removes an interpolation artifact;
- a fix to the Figure 13/15 percent-difference spline, which restores three curves of Figure 15 that were NaN.

Full record and known limitations: `doppler_nrbc/VERIFICATION.md`.

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
| `12_lmhd_coupling_kernel.png`–`17_pctdiff_xe2_Tb.png`, `20_photon_bath_ydistortion.png` | `plot_baryon_heating.py` (Figs 13–15 since 2026-10-16 from the transfer functions with component-scaled tolerances, `data_atol`, through `manuscript_sync/runs/7_atol_check/redraw_13_15.py`, the same plotting code; PREREG addenda 20–21) | Baryon-heating (T_b) coupling kernel and percent-difference diagnostics; Compton-y check if drag-dissipated KE heated the photon bath instead of being neglected |
| `18_lyman_alpha_moments.png`, `19_hyrec_saha_3la_psd_lineshape.png` | `plot_moments_and_lineshape.py` | Lyman-alpha moments and HyRec/Saha/3LA lineshape comparison (helium-independent) |

## Reproducing `18_lyman_alpha_moments.png` (Figure 8)

`plot_moments_and_lineshape.py` is self-contained (no pipeline data), but
several of its inputs appear only as array indices. Resolved values:

| Parameter | Value |
|---|---|
| Wavenumber | `karr[3]` = 3.149052e-20 m^-1 = **97169.60 Mpc^-1** |
| Redshift | `zind = 800` of `z_grid(1900,600,-1)` -> **z = 1100** (solver spans 800->802, i.e. z = 1100-1099; plotted row is z = 1100) |
| Multipole cutoff | `nm = 15` (`NM_PM1`), hierarchy closed at **l = 14** |
| Frequency grid | x in [-1000, +1000] Doppler widths, `steps = 100001`, **dx = 0.02**; plotted over x in [-200, 200] |
| Boundary conditions | **non-reflecting closure** at l = nm-1 (`fullzp1`/`fullzm1` default `closure=True`): the l_max diagonal gets -dx\*\|tvsh\|\*alpha_(L,1), alpha_(L,1) = sqrt((L^2-1)/(4L^2-1)), the same closure as the production moment tables |
| Doppler source | the D' bases include the first-order Doppler rate-factor l=1 source, +1/2 (m=+1) and +1/6 (m=-1) times (aH/ck) df/dx (default `doppler=True`); at this k, aH/ck ~ 5e-8, so it is invisible in the figure |
| Cosmology | astropy **Planck18**: H0 = 2.192711e-18 s^-1, Omega_m = 0.309660, Omega_Lambda = 0.688846, N_eff = 3.046, T0 = 2.7255 K, Y_p = 0.2454, Omega_b h^2 = 0.022418 |
| Plotted curves | l=1 is `p1moms[0, 1::15, 0]`, l=2 is `p1moms[0, 2::15, 0]`; the `::15` stride is the `nm` interleaving and the trailing `0` selects the **first of three basis solutions** |

The basis-solution index and the multipole normalization convention are the two
places an independent implementation is most likely to pick up an overall
amplitude offset while reproducing the curve shape correctly.

The published `18_lyman_alpha_moments.png` (made before 2026-09) used `nm = 30` with
simple truncation (`closure=False`) and no Doppler source (`doppler=False`), i.e. a
`::30` stride. The NRBC at l_max = 14 changes the curves by <= 0.8% of their peak at this k.

## 2026-10 update: manuscript sync (editorial proof of 2026-10-07) and two audits

The figures in `plots/` now come from `paper_data_plots/data_final4/` (gitignored), the regeneration after the third
repo audit on branch `manuscript-sync` (see `doppler_nrbc/VERIFICATION.md`, "2026-10-14"): tables, transfer functions and everything downstream in one pass. What changed, all recorded with
pre-registered bars in `manuscript_sync/PREREG.md` and summarized in `doppler_nrbc/VERIFICATION.md`
("2026-10 manuscript sync"):
- the manuscript's Eq. (63) convective term, Eq. (A40c) T_b form, photon-path optical depth (Eqs. 65–69), the
  Eq. (54) gravitational term, and the Eq. (B30) / (B32)+(B37) coefficients;
- δ_b = 3δ_γ in the tight-coupling era, helium electrons in z_fs, δα_B(δ_Tb) in δf_eq⁽¹⁾;
- the blue-side boundary of the homogeneous line solution at the line-centre blackbody value (tables rebuilt in
  `src/pmhd/data/pre_stored_data/`), the T_b chain's own continuum and photon-path inputs, Hprime by central
  difference, the θ = π/2 node of the local-3LA comparison, and the fast-mode handoff δ_b = 3δ_γ.
Effect: |Δx_e| deeper by ≤ 5% at its minimum and larger at z ≥ 1400; the photon-path optical depth lowers Δz_* by 7–10%
against the fixed-point integral; grid maximum of Δz_* 0.744 at 2685 pG, ε = −0.09 (the published data set gives 0.705 at
3083 pG, ε = −0.10 with the same peak extraction, see `manuscript_sync/PREREG.md` addendum 17); clumping unchanged to 0.12%; Figs 05, 12, 18, 19 unchanged. Every new term has an environment
switch that restores the previous behaviour (table below), and the transfer-function scripts run their angles in
parallel with `PMHD_NPROC` (bit-identical to the serial loop).

## What would need to be run to regenerate these

The plotting scripts live in this repo under `analysis/`, and everything they
depend on, physics-wise, is under `src/pmhd/`. The pipeline-stage scripts
(`src/pmhd/physics/*.py`) no longer carry a `_He` suffix either — helium
recombination is simply the default now. Regenerating from scratch means
running the SLURM drivers in `src/pmhd/physics/` in dependency order (each
one submits one job per bind, looping `kind`/`epsind` internally):

0. *(Only if the physics of the moment tables changes; the tables ship in `src/pmhd/data/pre_stored_data/`.)*
   `slurmloop_fbars.py all` — `firstOmoments_and_secondO_soln_PRD_fullk.py` for 130 chunks of 10 redshifts
   (about 150 core-h, 20 GB per job), writing under `PMHD_FBARS_OUTBASE`. Then build the masters and a complete
   table directory:
   `assemble_fbars_master.py --fbars-dir $OUTBASE/fbars/f1bars --out-dir <tables_dir> --f2bars-dir $OUTBASE/fbars/f2bars_dict`.
   Point the pipeline at a non-default table directory with `PMHD_TABLES_DIR`.
1. `slurmloop_repo.py all` — `TCR_Tfs.py` + `FSR_saha_Tfs.py` +
   `angle_avging_saha.py`, all 61 binds.
2. `slurmloop_TCR_FM.py` (default 7-bind subset `[0,10,...,60]`) — needed
   for figures 01–04 (the fast-mode continuation of Fig 01 and Fig 04). `slurmloop_FSR_z4500.py` (default 3-bind/7-k subset)
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

Each driver reads these from the environment (absolute paths; compute nodes cannot see `/tmp`) and exports them
to its submitted jobs:

| Variable | Meaning | Default |
|---|---|---|
| `PMHD_OUTDIR` | pipeline output directory | `/home/jonschiff/PMF-MHD-recomb/paper_data_plots/data` (edit on another clone) |
| `PMHD_TABLES_DIR` | moment tables | `src/pmhd/data/pre_stored_data` |
| `PMHD_DOPPLER_TERMS` | second-order Doppler terms in `ftot` (`0` = pre-2026-09 regression mode) | `1` |
| `PMHD_LOGDIR` | job logs | `$PMHD_OUTDIR/logs` |
| `PMHD_TB_BINDS` | subset of the Tb driver's `(bind, epsind)` pairs, by bind | all 13 pairs |
| `PMHD_KINDS`, `PMHD_KCHUNK`, `PMHD_MAX_QUEUED` | TLA driver: k subset, k per job, account-wide queue cap | all, 10, 190 |
| `PMHD_NPROC` | worker processes over θ in `TCR_Tfs.py`, `TCR_Tfs_FM.py`, `FSR_saha_Tfs.py`, `FSR_TLA_Tfs*.py` (set `--cpus-per-task` to match) | `1` |
| `PMHD_EQ63_TERM`, `PMHD_A40C_TERM`, `PMHD_GRAV11`, `PMHD_LOS_KERNEL`, `PMHD_DALPHA_FEQ`, `PMHD_ZFS_HE`, `PMHD_TCR_DELTA_B`, `PMHD_TB_INPUTS`, `PMHD_HOMPSD_BB_CENTER` | 2026-10 terms; `0` restores the pre-sync behaviour of each | `1` |
| `PMHD_B30_OLD`, `PMHD_ADVLENS_OLD`, `PMHD_DOP_SIGN_OLD`, `PMHD_HPRIME_OLD`, `PMHD_LOCAL_PI2_ZERO` | `1` restores the pre-sync coefficient / convention | `0` |
| `PMHD_XE_BG_OLD` | `1`: default-tolerance `odeint` background instead of the cached tight Radau solve (the tables are built on the tight one) | `0` |
| `PMHD_TB_COEFF_TG` | `1`: T_γ in the T_b variant's pressure and δα_B response terms (pre-audit behaviour) | `0` |
| `PMHD_ODE_METHOD`, `PMHD_TOL_SCALE` | transfer-function solver method and a tolerance multiplier (convergence tests, stall fallback) | `LSODA`, `1` |
| `PMHD_ATOL_SCALED`, `PMHD_ATOL_FRAC`, `PMHD_ATOL_PASSES`, `PMHD_ATOL_LSODA_BUDGET` | component-scaled absolute tolerances in the Saha, TLA and T_b transfer-function solves (`ode_atol.py`, PREREG addenda 20–22): atol_i = FRAC × rtol × max\|y_i\| of the previous pass, iterated until stable; the TLA and T_b Alfvén solves at rtol 1e-6; an LSODA pass over the budget (s) is redone with Radau. **On by default since 2026-10-16**; `0` reproduces the earlier outputs bit for bit (`data_final4` and everything before). Strong-field modes cost 2–3× more | `1`, `1e-3`, `4`, `2400` |

The plotting scripts read `PMHD_OUTDIR`, `PMHD_PLOTDIR`, and `PMHD_CACHEDIR` (where the Figure 07 and Figure 20
caches go; use a separate one per data set).

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
- The submit limit (QOS `normal`: 200 queued jobs) is per Slurm *account*, shared with everyone on it; a refused
  `sbatch` is not an error the drivers can see in their own queue. `slurmloop_TLA.py` throttles on the whole
  account's queue and retries refusals. After any stage, check that the expected number of output files exists.
- Some integrations are extremely stiff at strong/weak field extremes —
  `TCR_Tfs_FM.py` and `FSR_TLA_Tfs_Tb.py` in particular are markedly slower
  for the strongest-field binds (low `bind` index).
- Runtime of the first-order transfer functions with the default component-scaled tolerances (`PMHD_ATOL_SCALED=1`,
  the default since 2026-10-16; PREREG addenda 21–22). Measured per mode (Saha + TLA + T_b) on the gpu nodes
  (Intel Xeon Gold 6326), with the 17-angle pool (`PMHD_NPROC=17`, `--cpus-per-task=17`), in the addendum-21 run; mean
  (maximum): 34 s (108 s) at 5 pG, 37 s (106 s) at 20 pG, 2 min (19 min) at 315 pG, 5–6 min (1.2–1.7 h) at 2.5–2.7 nG,
  14 min (2.6 h) at 5000 pG. The slowest are the stiffest strong-field modes (k index 0–7). There LSODA stalls on the
  TLA and T_b stages, and `ode_atol.py` switches to Radau after `PMHD_ATOL_LSODA_BUDGET` (2400 s). Overall this is 2–3×
  the cost with `PMHD_ATOL_SCALED=0`. With `PMHD_NPROC=1` the 17 angles run serially, about 17× slower: 315 pG at
  k = 19.4 Mpc⁻¹ takes 43 s with the pool and roughly 12 min on one core. Size job time limits to the modes: up to 4 h
  for the stiff strong-field single-mode jobs, and short for the rest, so that the scheduler can backfill them.
- On a busy shared cluster, wall-clock time is often dominated by queue
  congestion and node contention from other users' jobs rather than the
  computation itself — check `squeue`/`sacct` before assuming a stage is
  stuck; per-job compute time for most binds is on the order of minutes.
- Total intermediate-data footprint when this was run: ~150 GB, now stored
  in-repo (gitignored) under `paper_data_plots/data/`.
