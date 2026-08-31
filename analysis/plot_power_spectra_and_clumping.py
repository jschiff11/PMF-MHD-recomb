"""
Magnetic and baryon-clumping power-spectrum figures:
  - 05_magnetic_power_spectrum.png : Delta_B^2(k)/Delta_B^2(k_Mpc) at several
    redshifts, with the conservation scale k_cons marked
  - 06_clumping_power_spectrum.png : Delta^2_{delta_b}(k), 3LA-with-radiative-
    transfer vs Saha, for three field strengths

Inputs, read from the angle-averaged pipeline outputs under $PMHD_OUTDIR
(written by angle_avging_TLA.py and angle_avging_saha.py):
  ang_avg/TLA/B_{B}pG/ang_avg_k{kind}.pkl
  ang_avg/saha/B_{B}pG/ang_avg_k{kind}.pkl
for all 61 field strengths and all 69 wavenumbers.

Configuration (environment variables):
  PMHD_OUTDIR  : pipeline output directory to read data from
                 (default: <repo>/src/pmhd/data/outputs, the same default the
                 src/pmhd/physics scripts write to)
  PMHD_PLOTDIR : directory to write figures to (default: <repo>/analysis/plots)

Run:  python analysis/plot_power_spectra_and_clumping.py
"""
import os, sys, pickle, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.interpolate import splrep, splev

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pmhd import cons, pars
from pmhd.data.grids import k_grid, eps_grid, z_grid, load_or_generate_B0arr

DATA = Path(os.environ.get("PMHD_OUTDIR", str(REPO_ROOT / "src/pmhd/data/outputs")))
PLOTDIR = Path(os.environ.get("PMHD_PLOTDIR", str(REPO_ROOT / "analysis/plots")))
PLOTDIR.mkdir(parents=True, exist_ok=True)

epsarr = eps_grid()
B0arr = load_or_generate_B0arr()
karr = k_grid()
zarr = z_grid()


def pg(bind):
    return round(1e12 * B0arr[bind])


print("loading ang_avg TLA/saha for all binds ...", time.ctime())

bxbxbar = np.zeros((len(karr), len(zarr), len(B0arr)))
bybybar = np.zeros((len(karr), len(zarr), len(B0arr)))
deltamdeltambar = np.zeros((len(karr), len(zarr), len(B0arr)))
deltamdeltambar_saha = np.zeros((len(karr), len(zarr), len(B0arr)))

for bind in range(len(B0arr)):
    tla_dir = DATA / f"ang_avg/TLA/B_{pg(bind)}pG"
    saha_dir = DATA / f"ang_avg/saha/B_{pg(bind)}pG"
    for kind in range(len(karr)):
        with open(tla_dir / f"ang_avg_k{kind}.pkl", "rb") as f:
            d = pickle.load(f)
        bxbxbar[kind, :, bind] = d["bxbxbar"]
        bybybar[kind, :, bind] = d["bybybar"]
        deltamdeltambar[kind, :, bind] = d["deltamdeltambar"]
        with open(saha_dir / f"ang_avg_k{kind}.pkl", "rb") as f:
            dsaha = pickle.load(f)
        deltamdeltambar_saha[kind, :, bind] = dsaha["deltamdeltambar"]

print("done loading;", time.ctime())

xe_full = pars.xe_full

# ---------------------------------------------------------------------------
# Figure 05: magnetic power spectrum ratio vs k
# ---------------------------------------------------------------------------
bind = 30
lamarrfine = 10**np.arange(20, np.log10(2 * np.pi / karr[-1]), .01)
karrfine = 2 * np.pi / lamarrfine
epsind = 9

fig, ax = plt.subplots()
for zind in range(0, 1200, 300):
    line, = ax.loglog(karrfine * cons.mpc,
                      (1 / (karrfine[529] * cons.mpc)**epsarr[epsind]) * (karrfine * cons.mpc)**epsarr[epsind] *
                      (splev(karrfine * cons.mpc, splrep(karr[::-1] * cons.mpc, bxbxbar[:, zind, bind][::-1] / 4)) +
                       splev(karrfine * cons.mpc, splrep(karr[::-1] * cons.mpc, bybybar[:, zind, bind][::-1] / 4))),
                      label=f'z = {zarr[zind]}')
    color = line.get_color()
    kdyn = cons.mpc / np.sqrt(cons.c * pars.Btild(B0arr[bind])**2 * (1 + zarr[zind])**2 /
                               (pars.nh(zarr[zind]) * xe_full(zarr[zind]) * cons.sigmat * pars.H(zarr[zind])))
    ax.plot(kdyn, (1 / (karrfine[529] * cons.mpc)**epsarr[epsind]) * kdyn**epsarr[epsind],
           marker='*', color=color, markersize=12)

ax.loglog(karrfine * cons.mpc, (1 / (karrfine[529] * cons.mpc)**epsarr[epsind]) * (karrfine * cons.mpc)**epsarr[epsind],
          'm:', label=r'$k^{\epsilon}=k^{-0.1}$')
ax.legend(fontsize=12)
ax.tick_params(size=12, labelsize=12, labelbottom=True)
ax.set_xlabel(r'k (Mpc)$^{-1}$', fontsize=14)
ax.set_ylabel(r'$\frac{\Delta_B^2(k)}{\Delta_B^2(k_{Mpc})}$', fontsize=18)
ax.set_title(f'$B_0 = $ {pg(bind)} pG', fontsize=14)
plt.tight_layout()
plt.savefig(PLOTDIR / "05_magnetic_power_spectrum.png", dpi=150)
plt.close(fig)
print("saved 05_magnetic_power_spectrum.png")

# ---------------------------------------------------------------------------
# Figure 06: baryon clumping-factor power spectra, TLA vs Saha
# ---------------------------------------------------------------------------
numberofplotspec = 5
colors = [plt.cm.Reds(i) for i in np.linspace(0.2, 1, numberofplotspec)]
numberofplotspec2 = 8
colors2 = [plt.cm.magma(i) for i in np.linspace(0.2, 1, numberofplotspec2)]
bindarr2 = [0, 30, 60]
kmin_kind = np.argmin(np.abs(karr * cons.mpc - 1.0))  # kind whose k is closest to 1 Mpc^-1
lamarrfine = 10**np.arange(20, np.log10(2 * np.pi / karr[kmin_kind]), .01)
karrfine = 2 * np.pi / lamarrfine
Lambda = 1e3 * cons.mpc

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs = axs.ravel()
for bcount, bind in enumerate(bindarr2):
    ax = axs[bcount]
    ax.tick_params(size=12, labelsize=12, labelbottom=True)
    from cycler import cycler
    ax.set_prop_cycle(cycler('color', colors))
    zindarr = np.arange(np.argwhere(zarr == 1500).item(), np.argwhere(zarr == 1000).item() + 1, 100)

    for zi in range(len(zindarr)):
        zind = zindarr[zi]
        ax.semilogx(karrfine * cons.mpc,
                   (abs(epsarr[9]) / 4) * (Lambda * karrfine / (2 * np.pi))**epsarr[9] *
                   abs(splev(karrfine * cons.mpc, splrep(karr[::-1] * cons.mpc, deltamdeltambar[::-1, zind, bind]))),
                   label=f'$z={zarr[zind]:.0f}$', color=colors2[zi])
        ax.semilogx(karrfine * cons.mpc,
                   (abs(epsarr[9]) / 4) * (Lambda * karrfine / (2 * np.pi))**epsarr[9] *
                   abs(splev(karrfine * cons.mpc, splrep(karr[::-1] * cons.mpc, deltamdeltambar_saha[::-1, zind, bind]))),
                   ':', color=colors2[zi])
    ax.set_xlabel(r'$k \; (Mpc^{-1})$', fontsize=16)
    ax.set_title(f'$B_0 = $ {pg(bind)}pG', fontsize=20)
    if bcount % 3 == 0:
        ax.set_ylabel(r'$\Delta^2_{\delta_b}(k)$', fontsize=16)
    if bcount == 0:
        legend1 = ax.legend(loc='upper right', fontsize=12)
        ax.add_artist(legend1)
        from matplotlib.lines import Line2D
        line_legend_elements = [
            Line2D([0], [0], linestyle='-', color='black', label='3LA RT'),
            Line2D([0], [0], linestyle=':', color='black', label='Saha'),
        ]
        ax.legend(handles=line_legend_elements, loc='upper left', fontsize=12)

plt.tight_layout()
plt.savefig(PLOTDIR / "06_clumping_power_spectrum.png", dpi=150)
plt.close(fig)
print("saved 06_clumping_power_spectrum.png")

print("all power-spectrum/clumping plots done;", time.ctime())
