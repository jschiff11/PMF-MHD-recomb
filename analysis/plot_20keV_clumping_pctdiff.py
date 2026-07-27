"""
Referee B, Comment 4: percent difference in the clumping factor b(z) induced
by ignoring e+e- annihilation, i.e. re-initializing modes that cross the
horizon before T = 20 keV at z(20 keV) instead of zcross (kinds 0-22), using
the corrected pipeline (astropy Planck18 H(z), helium recombination included).

Clumping factor (Saha), per the standard definition:
    b(z) = |eps|/4 * (Lambda/2pi)^eps
           * \\int_{karr[-1]}^{karr[0]} k^(eps-1) * deltamdeltambar(k,z) dk

Inputs, read from $PMHD_OUTDIR (default: paper_data_plots/data):
  ang_avg/saha/B_{B}pG/ang_avg_k{kind}.pkl       (baseline, init @ zcross)
  ang_avg/saha_z20/B_{B}pG/ang_avg_k{kind}.pkl   (init @ z(20 keV), kind 0-22 only)

Run:  python analysis/plot_20keV_clumping_pctdiff.py
"""
import os, sys, pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splint

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pmhd import cons
from pmhd.data.grids import k_grid, eps_grid, z_grid, load_or_generate_B0arr

DATA = Path(os.environ.get("PMHD_OUTDIR", str(REPO_ROOT / "paper_data_plots/data")))
PLOTDIR = Path(os.environ.get("PMHD_PLOTDIR", str(REPO_ROOT / "paper_data_plots/plots")))
PLOTDIR.mkdir(parents=True, exist_ok=True)

karr = k_grid()
epsarr = eps_grid()
zarr = z_grid()
B0arr = load_or_generate_B0arr()

Lambda = 1e3 * cons.mpc  # 1 Gpc cutoff scale
epsind = 9               # eps = -0.1, matching the power-spectrum plots
eps = epsarr[epsind]

bindarr = [0, 30, 60]   # B0 = 5000, 315, 5 pG
KAFFECTED = 23          # modes with zcross > z(20 keV): kind 0-22
colors = [plt.cm.magma(i) for i in np.linspace(0.2, 0.85, len(bindarr))]


def pg(bind):
    return round(1e12 * B0arr[bind])


dd_saha = np.zeros((len(karr), len(zarr), len(B0arr)))
dd_z20 = np.zeros((len(karr), len(zarr), len(B0arr)))
for bind in bindarr:
    saha_dir = DATA / f"ang_avg/saha/B_{pg(bind)}pG"
    z20_dir = DATA / f"ang_avg/saha_z20/B_{pg(bind)}pG"
    for kind in range(len(karr)):
        with open(saha_dir / f"ang_avg_k{kind}.pkl", "rb") as f:
            base = pickle.load(f)["deltamdeltambar"]
        dd_saha[kind, :, bind] = base
        if kind < KAFFECTED:
            with open(z20_dir / f"ang_avg_k{kind}.pkl", "rb") as f:
                dd_z20[kind, :, bind] = pickle.load(f)["deltamdeltambar"]
        else:
            dd_z20[kind, :, bind] = base

prefac = abs(eps) / 4 * (Lambda / (2 * np.pi)) ** eps


def clumping(dd, bind):
    """b(z) for one bind: k-integral of k^(eps-1)*deltamdeltambar."""
    b = np.zeros(len(zarr))
    for zind in range(len(zarr)):
        b[zind] = prefac * splint(karr[-1], karr[0],
                                   splrep(np.flip(karr),
                                          np.flip(karr ** (eps - 1) * dd[:, zind, bind])))
    return b


fig, ax = plt.subplots(figsize=(8, 6))
for c, bind in enumerate(bindarr):
    b_base = clumping(dd_saha, bind)
    b_new = clumping(dd_z20, bind)
    pct = 100 * (b_new - b_base) / b_base
    ax.plot(zarr, pct, color=colors[c], lw=2, label=f'$B_0 = {pg(bind)}$ pG')
    print(f'B0={pg(bind):>4}pG: clumping-factor %diff  '
          f'min={np.nanmin(pct):+.3e}  max={np.nanmax(pct):+.3e}')

ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.set_xlabel('$z$', fontsize=16)
ax.set_ylabel('Percent difference in clumping factor\n'
              r'$b=\langle\delta_b^2\rangle$ (init @ 20 keV vs init @ zcross)',
              fontsize=14)
ax.set_title(f'$\\epsilon = {round(eps,3)}$', fontsize=16)
ax.tick_params(size=12, labelsize=12)
ax.legend(fontsize=13)
ax.set_xlim(zarr.max(), zarr.min())  # high z on the left

plt.tight_layout()
out = PLOTDIR / 'referee_B_comment4_clumping_pctdiff_20keV.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print('saved', out)
