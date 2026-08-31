"""
Referee B, Comment 4 (supplementary): dimensionless baryon density power
spectrum Delta^2_{delta_b}(k), comparing modes initialized at horizon
crossing (solid) vs re-initialized at z(20 keV) (dotted), for three field
strengths, using the corrected pipeline (astropy Planck18 H(z), He
recombination included). Companion to plot_20keV_clumping_pctdiff.py's
integrated-b(z) figure, shown here at the power-spectrum level.

Inputs, read from $PMHD_OUTDIR (default: paper_data_plots/data):
  ang_avg/saha/B_{B}pG/ang_avg_k{kind}.pkl       (baseline, init @ zcross)
  ang_avg/saha_z20/B_{B}pG/ang_avg_k{kind}.pkl   (init @ z(20 keV), kind 0-22 only)

Run:  python analysis/plot_20keV_power_spectrum.py
"""
import os, sys, pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import splrep, splev

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

Lambda = 1e3 * cons.mpc
epsind = 9
eps = epsarr[epsind]

bindarr = [0, 30, 60]
KAFFECTED = 23
colors = [plt.cm.magma(i) for i in np.linspace(0.2, 1, 8)]

lamarrfine = 10 ** np.arange(20, np.log10(2 * np.pi / karr[44]), .01)
karrfine = 2 * np.pi / lamarrfine


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


def delta2(dd, zind, bind):
    return ((abs(eps) / 4) * (Lambda * karrfine / (2 * np.pi)) ** eps
             * abs(splev(karrfine * cons.mpc, splrep(karr[::-1] * cons.mpc, dd[::-1, zind, bind]))))


fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs = axs.ravel()
zindarr = np.arange(np.argwhere(zarr == 1500).item(), np.argwhere(zarr == 1000).item() + 1, 100)

for bcount, bind in enumerate(bindarr):
    ax = axs[bcount]
    ax.tick_params(size=12, labelsize=12, labelbottom=True)
    for zi, zind in enumerate(zindarr):
        ax.semilogx(karrfine * cons.mpc, delta2(dd_saha, zind, bind), color=colors[zi], label=f'$z={zarr[zind]:.0f}$')
        ax.semilogx(karrfine * cons.mpc, delta2(dd_z20, zind, bind), ':', color=colors[zi])
    ax.set_xlabel(r'$k \; (Mpc^{-1})$', fontsize=16)
    ax.set_title(f'$B_0 = $ {pg(bind)}pG', fontsize=20)
    if bcount % 3 == 0:
        ax.set_ylabel(r'$\Delta^2_{\delta_b}(k)$', fontsize=16)
    if bcount == 0:
        legend1 = ax.legend(loc='upper right', fontsize=12)
        ax.add_artist(legend1)
        line_legend_elements = [
            Line2D([0], [0], linestyle='-', color='black', label='init @ zcross'),
            Line2D([0], [0], linestyle=':', color='black', label='init @ 20 keV'),
        ]
        ax.legend(handles=line_legend_elements, loc='upper left', fontsize=12)

plt.tight_layout()
out = PLOTDIR / 'referee_B_comment4_power_spectrum_20keV_new.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print('saved', out)
