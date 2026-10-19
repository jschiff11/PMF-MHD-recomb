# Doppler/aberration corrections + non-reflecting closure: verification record

> In the public repository, the pre-registration files (`PREREG.md`), the cluster run records and
> `test_doppler_nrbc.py` (which needs reference data kept outside the repository) referred to below are not included;
> they are available from the authors on request.

Branch `doppler-nrbc` (stacked on `single-mode-source`, with `main` merged in at 993df41). The acceptance bars were
fixed in advance in [`PREREG.md`](PREREG.md). Addenda 1–5 there record every miss and decision, in order. This
file summarizes the outcome; the figures referred to are in [`figures/`](figures/).

## What changed

**Physics.**
- **Frame.** The PSD is parametrized by the matter-rest-frame frequency and the lab direction.
- **First order.** The collision term is multiplied by the Doppler factor 𝓓 = γ(1 − v·n̂/c). This adds an ℓ=1
  source to the line-transfer bases: −(aH/ck)∂f/∂x in 𝓑, and +½ and +⅙ (aH/ck)∂f/∂x in 𝓓⁺ and 𝓓⁻.
- **Second order.** The rest-frame monopole f₀₀ = f₀₀ᴺ + δf₀₀^Dop adds three source terms to `ftot`:
  - `fdop`: in- and out-scattering of δf^Dop (Eqs. 420 + 416);
  - `fdopbar`: the rate term (Eq. 421);
  - `fv2`: the Lorentz factor, ½⟨v²⟩/c² ∂f/∂ν (Eq. 417).
- **Closure.** The moment hierarchies are closed with the non-reflecting boundary condition at ℓ_max = 14 in all
  three towers. Previously the m=0 tower used the NRBC at ℓ_max = 8 and the m=±1 towers used hard truncation at 29.
- **Code.** `src/pmhd/physics/inhomo_moments.py`, `firstOmoments_and_secondO_soln_PRD_fullk.py` and
  `source_assembly.py`. The moment tables are rebuilt in `src/pmhd/data/pre_stored_data/`.

**Numerics, found and fixed along the way.**
- **x₁ₛ⁽²⁾ ODE step cap** (`max_step = 1` in `solve_x1squad`, and in the Radau solve of `xe2_Tb.py`), Addendum 5.
  - With unbounded steps, the dense-output interpolant produced a spurious bump of 3.8% of max|Δx_e| at 5 pG,
    ε = −0.01.
  - The cap by itself changes the published Δx_e by a median of 1.3e-6 of the peak, ≤ 4.8e-5 elsewhere, and
    1.7e-4 at 6 pG, ε = −0.01, where the old solve was itself off.
- **Figs 13/15.** The with/without-T_b percent-difference spline is now fitted through valid k only. The
  published Fig 15 was missing its z = 1200/1100/1000 curves: the TLA δx_e power is exactly 0 at k < 1 Mpc⁻¹
  there, and the 0/0 turned the whole spline into NaN.
- **Merged from `main`.** The `angle_avging_opt_depth` fix (FSR-era δτ carried into the TCR-era integral). It
  affects δτ and the visibility only.

## Checkpoint record

| CP | Result |
|---|---|
| CP1 code + unit tests | Regression mode bit-identical (A). New-term checks against independent research scripts pass (B1, B3–B5). **B2 missed**: 1e-9 superposition round-off; the production tables are bit-identical to one combined solve (B2′). Addendum 1. |
| CP2 one-chunk smoke test | **Marginal miss** on m=−1 non-derivative keys (1.0e-12). Accepted in Addendum 2. |
| CP3 full table rebuild | Complete, and bit-identical to the verified tables at the 4 test redshifts. **b̄ flag**: changes up to 1.35e-2 at z > 1500. Cause: the old ℓ_max = 8 tables are not converged before recombination. Addendum 3 sets the high-z bounds and a rebuild trigger. |
| CP4 pilot (20/315/2506 pG, ε = −0.1) | ⟨δx_e⁽²⁾⟩ **24/24 pass**; the peak location is within 1 of the predicted z. `ftot` **10/12**. The two 20 pG misses are prediction omissions: the prediction used bases without the Doppler source, and assumed ~1e-5 TF changes. **TF flag**: δx_e TF up to 3.7e-2 at low k, from the high-z tables. The **rebuild trigger fired**, via its 10%-of-a-tiny-prediction clause only; both bounds are ≤ 1.2e-5 in Δx_e. Decision: no ℓ_max = 39 rebuild. Addendum 4. |
| CP5/CP6 full grid + figures | All 61 fields; no failures; every output count complete. One solver artifact, fixed (Addendum 5). Figure comparison below. |

## Effect on the paper figures (with Doppler, compared with a no-Doppler baseline run through the same current code)

| Figure | Change |
|---|---|
| 01–05, 12, 19 | pixel-identical |
| 20 | Doppler-independent (baseline = Doppler, pixel-identical). The shipped PNG is kept as published; see "Open items". |
| 07 Δx_e(z) | \|Δx_e\| reduced by ≤ 0.10% of the peak at 5000 pG, 0.06% at 2506 pG, < 0.02% for ≤ 315 pG (ε = −0.1). Pointwise, up to 0.36% at 2506 pG near z ≈ 880. |
| 08 min Δx_e | ≤ 0.13%, largest at 5000 pG and ε → −0.25 |
| 09/11 visibility-peak shift Δz* | reduced by ≤ 2.2e-3 in z, against Δz* up to 0.70. As plotted (grid step 0.013), 79 of 1525 cells move one step. |
| 06, 10 clumping | ≤ 1e-5 relative |
| 14 T_b power | ≤ 8e-6 |
| 16, 17 | δT_b⁽²⁾ ≤ 0.08% and Δx_e (T_b-coupled) ≤ 0.10% of the peak |
| 13, 15 | **bar-D flags** (changed by > 1e-3 of the curve maximum). See below. |
| 18 𝓓± moments | closure change ≤ 0.8% at the plotted k. The Doppler source is invisible there (aH/ck ~ 5e-8); at k ~ 1 Mpc⁻¹ it adds an ℓ=1 bump of ~1% of the ℓ=2 amplitude. |

Side-by-side images: `figures/compare/cmp_*.png` (without | with). Difference plots: `figures/compare/diff_*.png`.
Masked Figs 13/15: `figures/compare/mask_*.png`. Pilot panels: `figures/cp4_pilot/`. These pilot panels predate
the ODE step cap; the Δx_e panel is unaffected at plotting precision.

**Bar-D flags, Figs 13 and 15.** Measured over the plotted k ≥ 7.7 Mpc⁻¹, as a fraction of each panel's curve maximum:

| Figure | 5000 pG | 315 pG | 5 pG |
|---|---|---|---|
| 13 | 7.6e-4 | 1.1e-3 | 1.9e-2 |
| 15 | 1.2e-2 | 1.7e-1 | 0.57 |

- **Where.** At k ≲ 100 Mpc⁻¹ in the weak-field panels, and in the early-z curves of Fig 15.
- **Cause.** The rebuilt high-z line-average tables changed the first-order δx_e (and T_b) transfer functions:
  the δx_e power by up to 13% (13 pG, k ≈ 8 Mpc⁻¹) and the T_b power by up to 33% (10 pG, k ≈ 2 Mpc⁻¹).
  - The hybrid run (new tables for z ≤ 1500, old above) reproduces the baseline at the flagged 315 pG point.
  - The attribution to the ℓ_max change rather than the first-order Doppler source is inferred: the source changes
    b̄ by ≤ 2e-5 for z ≤ 1500, and the ℓ_max = 8 vs 59 error accounts for the size of the change above 1500.
- **Status.** The flags stand. The new panels are more accurate than the published ones, but not fully converged
  in ℓ_max.

## 2026-10 manuscript sync (branch `manuscript-sync`)

Record of bringing the code into line with the corrected manuscript (editorial proof of 2026-10-07), with two external
audits. Pre-registration and every result: `paper_data_plots/manuscript_sync/PREREG.md` (addenda 1–11). Output data
set: `paper_data_plots/data_final/` (gitignored); figures: `paper_data_plots/plots_final/`, promoted into `plots/` on 2026-10-13; superseded on 2026-10-14 by the regeneration described in the next section.

**Physics brought to the manuscript.**
- Eq. (63): the convective term +(1/a)⟨Θ⁽¹⁾δx_e⁽¹⁾⟩ in the ⟨δx_e⁽²⁾⟩ equation (`solve_x1squad`, `xe2_Tb.py`);
  reduces |Δx_e| by a median 2.5e-2, up to 0.18 at 5 pG. Switch `PMHD_EQ63_TERM=0`.
- Eq. (A40c): the second-order T_b equation carries −(1/3)⟨δ̇_bδ_Tb⟩ only (`PMHD_A40C_TERM=0` restores the old form).
- Eqs. (65)–(69): the optical-depth variance and ⟨δτδτ̇⟩ are integrated along the photon path with the cell-integrated
  j₀(kcΔη) kernel (`angle_avging_opt_depth.py`; `PMHD_LOS_KERNEL=0` forces the kernel to 1, bit-identical to the old
  fixed-x integral). Lowers Δz_* by 7–10%.
- Eq. (54): the gravitational ∂_ν𝒜₁/𝓑₁/𝒞₁ term in `fred11` (`fgrav11`; `PMHD_GRAV11=0`); ≤ 1e-5 of `ftot`.
- Eq. (B30): `fred20` coefficient 8π (was 16π/3; `PMHD_B30_OLD=1`); Eqs. (B32)+(B37): advection + lensing coefficient
  −8π (was −8π/3; `PMHD_ADVLENS_OLD=1`); both ≤ 2.3e-6 of max|Δx_e|.
- Eq. (B20) sign: verified by direct angular integration of 2⟨(v·n̂/c)δf⁽¹⁾⟩ (addendum 1; `PMHD_DOP_SIGN_OLD`).
- Footnotes 2 and 4 (first-order Doppler dipole, γ⁽²⁾ term): kept in the code; dropping them changes ⟨δx_e⁽²⁾⟩ by
  ≤ 1.2e-6 of its maximum (addendum 6), so the manuscript's omission is valid.
- Reviewer items (addenda 7–8): δ_b = 3δ_γ in the tight-coupling era (`PMHD_TCR_DELTA_B=0` restores δ_b = δ_γ;
  ≤ 1.1e-6 on Δx_e); helium electrons in z_fs (`PMHD_ZFS_HE=0`; ≤ 7e-4); δα_B(δ_Tb) in δf_eq⁽¹⁾ and the
  n = 2 balance (`PMHD_DALPHA_FEQ=0`; ≤ 1.6e-4).
- Second audit (addendum 9): **blue-side boundary of the homogeneous line solution = blackbody at line centre**
  (`PMHD_HOMPSD_BB_CENTER=0` restores the Planck value at +1000 Doppler widths). The old boundary drove the line out
  of equilibrium at z ≥ 1300 (ā, b̄ tables too large by 1/0.955 at 1300, 1/0.25 at 1500, ~10⁴ at ≥ 1700). With the
  corrected boundary the first-order ⟨δx_eδ_b⟩/⟨δ_b²⟩ recovers the local Saha value where the line is in LTE
  (z ≥ 1400) and is unchanged at z ≤ 1300. Tables rebuilt and promoted to the default `src/pmhd/data/pre_stored_data/` (the 2026-09 tables stay in git
  history, 692f6f0 and earlier; the regression reference `pre_stored_data_prev` is unchanged). Also: the T_b chain uses its own continuum source and
  photon-path terms (`PMHD_TB_INPUTS=0`), θ = π/2 node kept in the local-3LA comparison (`PMHD_LOCAL_PI2_ZERO=1`),
  fast-mode handoff δ_b = 3δ_γ, He-inclusive Alfvén drag in the z > 1900 heating, Hprime from the astropy H by
  central difference (`PMHD_HPRIME_OLD=1`), δx_e sign in the 20 keV Saha diagnostic, odd parity of Θ′ in the mirrored
  angle averages.
- Numerics: per-angle parallel transfer-function integration (`theta_pool.py`, `PMHD_NPROC`; bit-identical to the
  serial loop); two free-streaming Saha modes (9 pG k12, 8 pG k13) that stall with LSODA use Radau (addendum 10).

**Effect on the figures** (`plots_final` vs the 2026-09 `plots/`; details in addendum 11). |Δx_e| is deeper by
≤ 5% at its minimum and larger by tens of percent at z ≥ 1400 (the line-centre boundary); the photon-path τ lowers
Δz_* by 7–10% against the fixed-point integral; grid maximum **0.744 at B₀ = 2685 pG, ε = −0.09** (published data set:
0.705 at 3083 pG, ε = −0.10; both with the peak extraction of 2026-10-15, addendum 17; the 0.737 quoted earlier
subtracted an unrefined no-field peak); clumping unchanged to 0.12%;
δT_b⁽²⁾ changes by ≤ 0.5% of its maximum. Figures 05, 12, 18, 19 unchanged; 14 and 20 change only at the
line-rendering level (lowest-k T_b power at 5 pG; He drag in y). Fig 11 colour limit raised to 0.75. Fig 20 regenerated (the Thomson term is ≤ 0.6% of the drag term for B₀ ≥ 315 pG).

**Regression.** `test_source_assembly.py` passes 7/7 with `PMHD_DOPPLER_TERMS=0` against the published outputs
(reference tables, see below); every new term is behind a switch whose "old" setting was shown bit-identical to the
previous output before the term was turned on (addenda 1, 3, 7–9).

## 2026-10-14: third audit and the final regeneration

Pre-registration and results: `paper_data_plots/manuscript_sync/PREREG.md`, addenda 12–16. Data set behind `plots/`:
`paper_data_plots/data_final4` (gitignored; links into `data_final3` and `data_final2` for the products that did not
change).
- **Background.** The homogeneous Peebles x_e is a tight Radau solve (cached `xe_peebs_tight.npy`; `PMHD_XE_BG_OLD=1`
  restores the default-tolerance `odeint`). The line departure (f_eq − f_bb)/f_bb, a near-cancellation at z ≥ 1750, was
  noise-dominated before (100× too large at z = 1900, wrong sign at 1800).
- **Tables.** All 1300 redshifts rebuilt on that background. A first attempt rebuilt only z ≥ 1551 and left the
  pipeline inconsistent: the processing term of the source subtracts large near-equal quantities and turned the ~1e-7
  background mismatch into percent-level single-redshift spikes (z = 1430 and ten others) and a spurious 3e-3 change of
  Δx_e. With consistent tables the change of Δx_e from the background fix is ≤ 2.5e-5 of its maximum (≤ 2.3e-6 for
  B₀ ≥ 300 pG) and Δz_* moves by ≤ 7e-6.
- **T_b variant (finding 4).** The pressure coefficient and the δα_B response terms use the homogeneous T_b; every
  other recombination term uses α_B(T_γ), the value the background is solved with (`PMHD_TB_COEFF_TG=1` restores T_γ
  everywhere). Using α_B(T_b) in the background-balance coefficient of x_1s⁽²⁾, whose terms cancel to 1 part in ~500
  at z ≈ 1400, produced a spurious field-independent +5e-4 change of the T_b-coupled Δx_e; that variant was discarded.
  Adopted approximation: the background is solved with α_B(T_γ); α_B(T_b)/α_B(T_γ) − 1 = 8e-6 (z = 1100), 6e-4 (800),
  4e-3 (600).
- **Convergence.** ℓ_max 14 → 22 and Δx 0.02 → 0.01 change the tables by ≤ 5.7e-4 in the band 10–10⁴ Mpc⁻¹ at
  z = 1100–1500 (no rebuild); the transfer-function solver tolerances are converged (100× tighter: Δx_e ≤ 2e-5,
  Δz_* ≤ 1.2e-6); the second-order solve to 2e-7. Reproducibility floor across CPU families (AMD vs Intel nodes):
  1e-6 in the T_b products, ~2e-6 in individual transfer functions.
- **Robustness.** Every per-mode transfer-function step runs under a timeout scaled to the granted cores with an
  automatic Radau fallback; the two modes that stall with LSODA (fields 56 k12, 57 k13) are solved that way.
- **Table builder.** The signed-m spline inputs are 1-D (`test_table_builder.py` runs the entry point for one redshift).

## 2026-10-15: fourth audit

PREREG addendum 17.
- **Visibility shift.** The no-field and perturbed visibility peaks are both the zero of the derivative of the same
  cubic spline (`analysis/plot_xe2_and_visibility.py`, `refined_peak`); a zero perturbation gives exactly zero. All
  absolute Δz_* quoted before 2026-10-15 were 0.0063 too low (an unrefined no-field peak). Grid maximum 0.7436 at
  B₀ = 2685 pG, ε = −0.09; no negative shifts (minimum +5e-5). Run-to-run Δz_* differences were unaffected.
- **Heatmaps.** Figs 08–11 interpolate the 61 × 25 computed cells only (no zero-padded tilt columns, no smoothing of
  Fig 09), axes ε ∈ [−0.25, −0.01], B₀ ∈ [5, 5000] pG. The colour map (`twilight`, cyclic) is unchanged; it renders 0 and
  the maximum alike.
- **Constants.** `cons.py` uses CODATA 2018 explicitly; `pre_stored_data/xe_peebs_tight.json` and
  `pre_stored_data/TABLES_PROVENANCE.json` record what the background and the tables were built with, and the package
  refuses a mismatch. Under an Astropy default of CODATA 2022 the results are bit-identical.
- **Resolution above z = 1551.** The addendum-17 propagation (tables refined to ℓ_max 38 and to Δx 0.01 at 41
  redshifts; Δx_e changes ≤ 1.4e-5 of its maximum) is a direct-source sensitivity estimate at fixed first-order transfer
  functions, not an upper bound on the complete calculation. The full-chain check that closes the item is in the next
  section.

## 2026-10-15: fifth audit

PREREG addendum 18. The audit confirms the fourth-audit corrections (peak location, heatmaps, constants) and identifies
no algebraic error in the production physics equations. Its two comments concern the table resolution above z = 1551:
1. *The first-order transfer functions were held fixed.* The tables also enter the first-order TLA evolution (escape
   probability, σ_A, σ_B), which the addendum-17 propagation did not rerun.
2. *A ±envelope history is not a bound on the visibility-peak shift*, which responds to the slope of the change; the
   addendum-17 ±b runs are stress tests only.

- **Full-chain check (closes both comments).** Each refinement separately (ℓ_max 38; Δx 0.01): refined tables at every
  redshift (exact builds at 57 and 121 redshifts, the change interpolated in z in between and validated against exact
  builds at six other redshifts), then every table-dependent stage rerun, including the first-order TLA and T_b
  transfer functions, the angle averages, the line-of-sight and optical-depth correlations and the visibility, for
  B₀ = 5000, 2685, 2506, 315 and 20 pG, all 25 tilts and the 10 T_b pairs; the signed refined histories are passed
  through the visibility. Δx_e changes by ≤ 6.1e-6 of its maximum and Δz_* by ≤ 3.3e-6 per refinement (combined
  estimate 4.6e-6; headline cell 2685 pG, ε = −0.09: −4.3e-8 and −3.0e-6), the T_b outputs by ≤ 5.9e-6; all at least
  16× below the pre-registered bars and within a few times the cross-CPU reproducibility floor. **Δz_* ≃ 0.743583
  stands and the production resolution is kept.**
- **High-redshift source.** The total source above z ≈ 1800 is numerically jagged in production (point-to-point second
  difference up to 2.7 × its maximum) and both refinements move it by O(1) locally at z > 1500, but Δx_e does not follow
  (≤ 6e-5 of its local value at z > 1500, ≤ 1e-5 at 900 ≤ z ≤ 1500): the quasi-equilibrium tail of addenda 13 and 17.
- **Interpolation gate.** The bar applies to the table entries the chain reads (the author's decision after the first
  check missed, recorded as such). Four unread m = −1 entries, numerically ~1e-35, miss at any sampling; the Δx 0.01
  sampling below z = 1400 had to be densified to every 10th redshift.

## 2026-10-18: sixth audit

Audit of `c3c14a33`. Both fifth-audit comments are closed as representative-case numerical checks, and no further
correction is needed. The auditor reproduced the extrema of `compare_res.json` and of the solver comparison, reran the
tests (6), the equation/source/LOS checks (178) and the operator checks (16), tested `ode_atol.py` analytically (scaled,
legacy and forced-fallback paths), and confirmed that the promoted Figs 13–15 are byte-identical to the recorded ones.
Two follow-ups were made: the addendum-21 conclusion no longer reads as if the second-order results were unchanged (they
change by ≤ 1.9e-5, Δz_* by ≤ 1.8e-6), and `paper_data_plots/README.md` gives the transfer-function runtimes with the
new default (the auditor's single-core T_b run of one 315 pG mode was stopped for being slow; with the 17-angle pool it
takes 43 s, about 12 min on one core).

## Known limitations / open items

1. **The small components of weak-field, low-k first-order transfer functions were not resolved in production; fixed
   behind a switch (PREREG addenda 19–21, 2026-10-16).**
   - Cause: one absolute tolerance for a state whose magnetic component is O(1) while δx_e (and at the lowest k also
     δ, Θ, Φ) can be 1e-8–1e-12, and the Alfvén solves at rtol 1e-3. Production LSODA output for those components was
     noise: at low k the δx_e transfer function was off by up to ~200× its true maximum at 20 and 5 pG, ~10× at 315 pG,
     ~0.2 at 2.5–2.7 nG; the 5 pG density by up to 2.9× at k = 0.03 Mpc⁻¹. The earlier ℓ_max attribution is withdrawn.
   - Fix: `PMHD_ATOL_SCALED=1` (`src/pmhd/physics/ode_atol.py`) gives each component an absolute tolerance scaled to its
     own size and the Alfvén solves rtol 1e-6, in the Saha, TLA and T_b stages. With it off the output is
     bit-identical to before. Converged against 1e-2 × tolerances and against Radau; LSODA stalls on the stiffest
     strong-field TLA/T_b modes are solved by the production Radau fallback. Strong-field modes cost 2–3× production.
   - Effect (six-field check, addendum 21): Δx_e ≤ 1.9e-5 of its maximum, Δz_* ≤ 1.8e-6 (headline cell −3.7e-7), T_b
     outputs ≤ 1.4e-5; `deltamrms` and the Saha-limit curve ≤ 7.3e-6. Figs 05–11, 16, 17 and Δz_* ≃ 0.743583 stand.
     **Figs 13 and 15 (and 14 marginally) changed visibly at low k**, where the published T_b-vs-TLA percent differences
     were solver noise; they were replaced on 2026-10-16 by the redrawn versions (addendum 22).
   - Since 2026-10-16 the scaled tolerances are the default (`PMHD_ATOL_SCALED=1`), with an in-code Radau fallback for
     LSODA passes over 2400 s; `PMHD_ATOL_SCALED=0` reproduces the earlier outputs (`data_final4`) bit for bit.
2. **Fig 20 is not regenerated in this update.**
   - Rerunning its script with the current code, from scratch rather than from the 2026-07-12 cache, changes 2.4% of
     pixels at z ≳ 2×10⁴. The curves come out slightly higher there.
   - The change is identical with and without the Doppler terms, and it has not been traced.
   - The regenerated version is in the local `paper_data_plots/plots_doppler/`.
3. **`paper_data_plots/data/` on disk is still the published (pre-2026-09) pipeline output.** The new outputs are in
   `paper_data_plots/data_doppler/`, and the no-Doppler baseline is in `paper_data_plots/data_baseline/`. All three
   are gitignored.

## Regression reference

The regression tests (`test_source_assembly.py`, `test_doppler_nrbc.py`) compare against the published outputs. They
therefore read the previous tables through `pmhd.data.tables.reference_tables_dir()`, which is `PMHD_REF_TABLES_DIR`
or, by default, `src/pmhd/data/pre_stored_data_prev/` (not tracked). To recreate that directory from git (the git tree
of the pre-2026-09 tables; it is the same object in the private and the public repository):
```
mkdir -p src/pmhd/data/pre_stored_data_prev
git archive bd70630f76f73faf4969f91704216416c4405b95 | tar -x -C src/pmhd/data/pre_stored_data_prev
```
Pre-2026-09 behaviour of the table builder: `PMHD_FBARS_NM=9 PMHD_FBARS_NMPM1=30 PMHD_FBARS_FLAGS="--no-closure --no-doppler"`.
The second-order Doppler terms in `ftot` are switched off with `PMHD_DOPPLER_TERMS=0`.
