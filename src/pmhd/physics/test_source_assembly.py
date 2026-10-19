# pyright: ignore[reportMissingImports]
"""
Verification suite for ``pmhd.physics.source_assembly`` and the single-mode path.

Run:  PYTHONPATH=<repo>/src python test_source_assembly.py [--outdir DIR]

Checks (all against the committed paper data in ``paper_data_plots/data``):

1. **Abstraction check** -- drive ``assemble_at_z`` / ``assemble_cont_at_z`` DIRECTLY over the
   full 69-k grid with ``splint_reducer`` (i.e. the same entry points the single-mode path
   uses, not the script's ``main``) and require the power-spectrum ``ftot``, ``cross_corr``
   and ``xe2contsource`` to come out bit-identical.
2. **k-power rule** -- the single-mode reduction ``X(k0)*k0**(1-n)`` must reproduce the ratio
   the power-spectrum reduction gives between terms of different k-power when the spectrum's
   support collapses onto one k.  Expectation: ``n=2/n=1 -> 1/k0``, ``n=3/n=1 -> 1/k0**2``,
   independent of eps.
3. **End-to-end regression** -- run the three power-spectrum scripts for several
   (bind, epsind) and require bit-identical ``ftot`` / ``xe2contsource`` / ``cross_corr``, and ``xe2``
   within 2e-4 of its maximum (solver step cap, PREREG addendum 5).  Runs on ``reference_tables_dir()``.
   (Needs ``ang_avg`` inputs; reads them via a symlink so paper data is
   never written to.)

Note on reproducibility: ``xe2`` is only stable to ~2e-5 against perturbations far below
physical significance, because LSODA's adaptive stepping amplifies any input change up to its
own ``rtol=1e-6``.  Hence the reducers take a ``pre`` argument, so the floating-point
association of the original inline expressions is preserved exactly.
"""
import os
# This suite is a regression against the published (pre-2026-10) outputs: run with the manuscript-sync terms off
# (paper_data_plots/manuscript_sync/PREREG.md). Export them explicitly to test the new terms elsewhere.
for _k, _v in (("PMHD_EQ63_TERM", "0"), ("PMHD_GRAV11", "0"), ("PMHD_B30_OLD", "1"), ("PMHD_LOS_KERNEL", "0"), ("PMHD_A40C_TERM", "0"), ("PMHD_DOP_SIGN_OLD", "1"), ("PMHD_ADVLENS_OLD", "1"),
               ("PMHD_XE_BG_OLD", "1"), ("PMHD_TB_COEFF_TG", "1")):   # third audit (addendum 12): old background and T_gamma coefficients for the regression
    os.environ.setdefault(_k, _v)
import sys
import pickle
import argparse
import subprocess
from pathlib import Path

import numpy as np
from scipy.interpolate import splev

from pmhd import pars
from pmhd.data.tables import reference_tables_dir
from pmhd.data.grids import k_grid, z_grid, eps_grid, load_or_generate_B0arr
from pmhd.physics.source_assembly import (
    build_abg, unpack_f2bars, assemble_at_z, assemble_cont_at_z,
    splint_reducer, single_mode_reducer, CROSS_CORR_KEYS, BAR_KEYS,
)

REPO = Path(__file__).resolve().parents[3]
PRE_STORED = reference_tables_dir()   # pre-2026-09 moment tables, as the paper data; see pmhd.data.tables
PAPER = REPO / "paper_data_plots/data"


def check_abstraction(bind=38, epsind=0):
    """(1) shared assembly over 69 k == power-spectrum paper data, bit for bit."""
    karr, zarr, eps = k_grid(), z_grid(), eps_grid()[epsind]
    B0arr = load_or_generate_B0arr()
    xe_full = pars.xe_full
    am = pickle.load(open(PRE_STORED / 'abarinterpmaster.pkl', 'rb'))
    bm = pickle.load(open(PRE_STORED / 'bbarinterpmaster.pkl', 'rb'))
    cm = pickle.load(open(PRE_STORED / 'cbarinterpmaster.pkl', 'rb'))
    abg = build_abg(zarr, karr, lambda z, k: splev(z, am[k]),
                    lambda z, k: splev(z, bm[k]), lambda z, k: splev(z, cm[k]), xe_full)
    ang = PAPER / f"ang_avg/TLA/B_{round(1e12*B0arr[bind])}pG"
    M = {k: np.zeros((len(karr), len(zarr))) for k in BAR_KEYS}
    for ki in range(len(karr)):
        d = pickle.load(open(ang / f'ang_avg_k{ki}.pkl', 'rb'))
        for k in BAR_KEYS:
            M[k][ki] = d[k]
    taus, Dh, H = pars.taus(zarr, xe_full), pars.Deltah(zarr), pars.H(zarr)
    pab, xe = pars.pab(zarr), xe_full(zarr)
    red = splint_reducer(karr, eps)
    n = len(zarr)
    ft = np.zeros(n); cont = np.zeros(n); cc = {k: np.zeros(n) for k in CROSS_CORR_KEYS}
    for zi in range(n):
        m0 = pickle.load(open(PRE_STORED / f'f2bars_dict/f2bars_m0_dict_zch_{zi}.pkl', 'rb'))
        m1 = pickle.load(open(PRE_STORED / f'f2bars_dict/f2bars_m1_dict_zch_{zi}.pkl', 'rb'))
        p1 = pickle.load(open(PRE_STORED / f'f2bars_dict/f2bars_p1_dict_zch_{zi}.pkl', 'rb'))
        bars = {k: M[k][:, zi] for k in BAR_KEYS}
        bg = dict(z=zarr[zi], H=H[zi], taus=taus[zi], pab=pab[zi], xe=xe[zi], Deltah=Dh[zi])
        row, f11, fp, fa, f20, fg = assemble_at_z(bars, {k: abg[k][zi, :] for k in abg},
                                                  unpack_f2bars(m0, m1, p1), bg, red)
        ft[zi] = f11 + fp + fa + f20 + fg
        for k in CROSS_CORR_KEYS:
            cc[k][zi] = row[k]
        cont[zi] = assemble_cont_at_z(bars, karr, zarr[zi], xe_full, red)
    tag = f"B_{round(1e12*B0arr[bind])}pG_e{round(eps,3)}"
    ok = (np.array_equal(ft, np.load(PAPER / f"source_fncs/ftot_{tag}.npy"))
          and np.array_equal(cont, np.load(PAPER / f"source_fncs/xe2contsource_{tag}.npy")))
    cp = pickle.load(open(PAPER / f"cross_corr/cross_corr_{tag}.pkl", 'rb'))
    ok = ok and all(np.array_equal(cc[k], cp[k]) for k in cc)
    print(f"[1] abstraction over 69 k ({tag}): {'PASS (bit-identical)' if ok else 'FAIL'}")
    return ok


def check_kpower():
    """(2) single_mode_reducer's k0**(1-n) reproduces the PS ratio between k-powers."""
    karr = k_grid(); k0 = karr[8]
    allok = True
    for eps in (eps_grid()[0], eps_grid()[24], eps_grid()[99]):
        red_ps, red_sm = splint_reducer(karr, eps), single_mode_reducer(k0)
        spike = np.exp(-0.5 * ((np.log10(karr) - np.log10(k0)) / 0.01) ** 2)
        ps = {n: red_ps(spike, n) for n in (1, 2, 3)}
        sm = {n: red_sm(np.array([1.0]), n) for n in (1, 2, 3)}
        for n in (2, 3):
            allok &= np.isclose(ps[n] / ps[1], sm[n] / sm[1], rtol=1e-6)
    print(f"[2] single-mode k-power rule (n2/n1=1/k0, n3/n1=1/k0^2): "
          f"{'PASS' if allok else 'FAIL'}")
    return allok


def check_endtoend(outdir, cases=((0, 24), (21, 24), (38, 24), (55, 24), (38, 0))):
    """(3) the three power-spectrum scripts, several (bind, epsind), vs paper data."""
    B0arr, epsarr = load_or_generate_B0arr(), eps_grid()
    phys = REPO / "src/pmhd/physics"
    py = sys.executable   # the scripts run under the interpreter that runs this test (was <repo>/.conda/bin/python)
    allok = True
    for bind, epsind in cases:
        tag = f"B_{round(1e12*B0arr[bind])}pG_e{round(epsarr[epsind],3)}"
        d = Path(outdir) / f"reg_{tag}"
        for sub in ("cross_corr", "source_fncs", "xe2"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        link = d / "ang_avg"
        if not link.exists():
            link.symlink_to(PAPER / "ang_avg")
        # the stored paper data predate the Doppler/aberration terms: compare with them switched off
        env = dict(os.environ, PMHD_OUTDIR=str(d), PYTHONPATH=str(REPO / "src"), PMHD_DOPPLER_TERMS="0",
                   PMHD_TABLES_DIR=str(PRE_STORED))
        for cmd in ([py, "cross_corr_and_source_fncs.py", str(bind), str(epsind)],
                    [py, "cont_source.py", "--bind", str(bind), "--epsind", str(epsind)],
                    [py, "xe2.py", "--bind", str(bind), "--epsind", str(epsind)]):
            subprocess.run(cmd, cwd=phys, env=env, capture_output=True)
        ok = True
        for sub, name in (("source_fncs", "ftot"), ("source_fncs", "xe2contsource")):
            ok &= np.array_equal(np.load(d / sub / f"{name}_{tag}.npy"),
                                 np.load(PAPER / sub / f"{name}_{tag}.npy"))
        # xe2: the paper data predate the max_step = 1 cap in solve_x1squad (PREREG addendum 5); the cap alone
        # moves xe2 by <= 4.8e-5 of max|xe2| (1.66e-4 where the old unbounded-step solve was itself off)
        xa, xb = np.load(d / f"xe2/xe2_{tag}.npy"), np.load(PAPER / f"xe2/xe2_{tag}.npy")
        ok &= np.max(np.abs(xa - xb)) <= 2e-4*np.max(np.abs(xb))
        ca = pickle.load(open(d / f"cross_corr/cross_corr_{tag}.pkl", 'rb'))
        cb = pickle.load(open(PAPER / f"cross_corr/cross_corr_{tag}.pkl", 'rb'))
        ok &= all(np.array_equal(ca[k], cb[k]) for k in ca)
        print(f"[3] end-to-end {tag:22s}: {'PASS (ftot, cont, cross_corr bit-identical; xe2 within 2e-4 of max)' if ok else 'FAIL'}")
        allok &= ok
    return allok


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="/tmp/pmhd_source_assembly_tests")
    p.add_argument("--skip-endtoend", action="store_true")
    a = p.parse_args()
    results = [check_abstraction(), check_kpower()]
    if not a.skip_endtoend:
        results.append(check_endtoend(a.outdir))
    print("\nALL CHECKS PASS" if all(results) else "\nSOME CHECKS FAILED")
    raise SystemExit(0 if all(results) else 1)
