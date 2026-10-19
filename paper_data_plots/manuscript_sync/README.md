# Manuscript sync: record, final results, provenance

> In the public repository, `PREREG.md` and `runs/` (the pre-registration and the cluster run records) are not
> included; they are available from the authors on request. `final_results.npz` is included.

State of 2026-10-15 (branch `manuscript-sync`). The figures in `paper_data_plots/plots/` and the moment tables in
`src/pmhd/data/pre_stored_data/` are the final ones after the third repository audit.

## Files

- `PREREG.md`: every pass/fail bar, written before the numbers existed, and every result (addenda 1–18), including the
  misses with their structure and the two diagnoses that withdrew earlier attributions (addenda 14–16).
- `final_results.npz`: the final numbers behind the figures and the text, from the data set `data_final4`.
  - `B0_pG` (61), `eps` (25, the computed tilts), `z` (1300): the grid.
  - `dzstar` (61×25): Δz_* per field and tilt, the peak of the visibility per conformal time, both the perturbed and the
    no-field peak taken as the zero of the derivative of the same cubic spline (PREREG addendum 17); maximum 0.7436 at
    B₀ = 2685 pG, ε = −0.09; `visibility_peak_z` the perturbed peaks, `no_field_peak_z`, `visibility_no_field`.
  - `fig08_xe2_min`, `fig10_deltamrms_1295` (61×25): the other arrays Figs 08–11 are drawn from.
  - `xe2_eps_m0p1` (61×1300): Δx_e(z) at ε = −0.1 for every field; `xe2_B2506pG_all_eps` (25×1300): every tilt at
    2506 pG (Fig 07 and Fig 17 use subsets of these).
  - `tb_pairs` (13×2, field index and tilt index), `xe2_tb_pairs`, `xe2_Tb_tb_pairs`, `dTb2_tb_pairs` (13×1300): the
    uncoupled and T_b-coupled Δx_e and ⟨δT_b⁽²⁾⟩ of the 13 T_b pairs (Figs 15–17).
- `../plots/fig20_y_pieces.npz`: the drag and Thomson pieces of the Compton y of Fig 20.
- `runs/`: the scripts that produced the final data set, as they were run on the cluster (absolute paths, Slurm).

## How the final data set was produced

`data_final4` is layered; each layer reuses the products of the previous one that its changes do not affect.

1. `runs/1_final2_tcr_saha_fm/` (2026-10-13): all stages for 58 fields with the corrected code; the tight-coupling and
   Saha transfer functions, the Saha angle averages, the fast-mode (Figs 01–04) and z = 4500 (Fig 02) transfer functions
   of this run are the ones in the final data set. Its table-dependent products were discarded (addendum 14: the tables
   below z = 1551 were still on the old background).
2. `runs/2_final3_tables_and_table_dependent/` (2026-10-14): tables rebuilt at every redshift on the tight background,
   checked (`tables_check15.py`), gated on field 0 (`gate15.py`), then the TLA and T_b transfer functions, angle
   averages, line of sight and stage 5 for all 61 fields (addendum 15).
3. `runs/3_final4_tb/` (2026-10-14): the T_b products with α_B(T_b) only in the pressure and δα_B response terms
   (addendum 16); the figures and the comparison against the 2026-10-13 run.

`runs/checks/` holds the background check of finding 1 and the comparison scripts of the pilots.
`runs/4_fourth_audit/` holds the fourth audit's checks (PREREG addendum 17): the constants tests, the item-4 measurement,
bound and joint propagation with their outputs (`X2.json`, `propagate.json`), and examples of the refined table jobs. The
propagation is a direct-source sensitivity estimate at fixed first-order transfer functions, not a bound on the complete
calculation. `runs/5_fifth_audit/` holds the full-chain check of the fifth audit (addendum 18): the refined-table
interpolation and its check (`interp_tables.py`, `interp_check_*.json`, the part-1 versions kept), the run driver and
the per-refinement run script (`chain5b.sh`, `final_run5_L.sh`, the D copy differing as in `final_run5_D.diff`), the
comparison with its self-test (`compare_res.py`, `compare_res.json`) and the diagnostics (`structure_res.py`,
`tf_res.py`, `f1base_check.py`, and the null rerun and solver-tolerance test of the two worst transfer-function modes,
`null_tf.sh`, `tol_tf.sh`, with their logs). It passes every bar: Δx_e changes ≤ 6.1e-6 of its maximum and Δz_* ≤ 3.3e-6 per
refinement, so Δz_* ≃ 0.743583 stands at the production resolution. `runs/6_radau_check/` holds the solver check of
addendum 19 (Radau TLA and T_b transfer functions): its pilot (5000 and 20 pG) passes the second-order bars, but the
convergence control misses because the small components of weak-field low-k transfer functions lie below the
absolute tolerance; the check stopped at CP1. `runs/7_atol_check/` holds the fix and its checks (addenda 20, 21):
the bit-identity and convergence gates, the six-field full-chain comparison (passes: Δx_e ≤ 1.9e-5, Δz_* ≤ 1.8e-6),
the figure check, and in `figures/` the redrawn Figs 13–15 proposed as replacements, with published-vs-redrawn
comparison sheets. The
comparison script `runs/3_final4_tb/diff_final.py` is kept as it was run; its absolute Δz_* subtract an unrefined no-field
peak (addendum 17), and the corrected values are those in `final_results.npz`.

The full grids (transfer functions, angle averages, sources, all 1525 Δx_e and visibility histories, several GB) are
on the cluster under `paper_data_plots/data_final4` and the directories it links to (`data_final3`, `data_final2`, and for
fields 10, 30, 50 the pilot directory `data_pilot12`, produced by the same code);
they are not in the repository.

## Reproducing

The pipeline drivers and their order are in `paper_data_plots/README.md`; the environment switches that define the
final configuration are those exported in `runs/3_final4_tb/env.sh` (all corrections on, `PMHD_XE_BG_OLD=0`,
`PMHD_TB_COEFF_TG=0`), plus `PMHD_ATOL_SCALED=0`, which was the behaviour then and has had to be set explicitly since 2026-10-16, when component-scaled tolerances became the default (PREREG addendum 22). Figs 13–15 come from the default-tolerance data set `data_atol` (`runs/7_atol_check/`). The runs read the tables from a working copy, `pre_stored_data_full`, which is byte-identical
to the promoted `src/pmhd/data/pre_stored_data`. The regression suite
(`src/pmhd/physics/test_source_assembly.py`) and the table-builder test (`test_table_builder.py`) pass.
