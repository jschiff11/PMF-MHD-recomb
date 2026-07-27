"""
Ionization-shift and visibility-function figures:
  - 07_xe2_curves.png : Delta x_e(z) for representative B0/eps: radiative-
    transfer 3LA (solid) vs local Saha (dotted) vs local perturbed 3LA (dashed)
  - 08_min_xe2_heatmap.png : peak |Delta x_e| over the (B0, eps) grid
  - 09_visibility_shift_heatmap.png : shift of the visibility-function peak
    Delta z_* over the (B0, eps) grid
  - 10_clumping_heatmap.png / 11_clumping_and_visibility_side_by_side.png :
    clumping factor b = <delta_b^2> at z = 1295 and side-by-side summary

Inputs, read from the pipeline outputs under $PMHD_OUTDIR:
  xe2/xe2_B_{B}pG_e{eps}.npy                    (xe2.py)
  visib/visib_B_{B}pG_e{eps}.npy                (visib_integ.py)
  cross_corr/cross_corr_B_{B}pG_e{eps}.pkl      (cross_corr_and_source_fncs.py)
  ang_avg/saha/B_{B}pG/ang_avg_k{kind}.pkl      (angle_avging_saha.py)
  Tfs/B_{B}pG/TCRmag_k{kind}.npy                (TCR_Tfs.py)
  Tfs/B_{B}pG/FSRTLAmag_k{kind}.npy             (FSR_TLA_Tfs.py)
for all 61 field strengths and the first 25 eps values (epsind 0-24); the
transfer functions only for kinds 0-43 and binds 0, 10, ..., 50 (used for the
local perturbed-3LA comparison curves).

Configuration (environment variables):
  PMHD_OUTDIR  : pipeline output directory to read data from
                 (default: <repo>/src/pmhd/data/outputs, the same default the
                 src/pmhd/physics scripts write to)
  PMHD_PLOTDIR : directory to write figures to (default: <repo>/analysis/plots)

Run:  python analysis/plot_xe2_and_visibility.py
"""
import os, sys, pickle, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev, splint
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

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
xesaha_full = pars.xesaha_full

NEPS = 25  # number of eps values used for these figures (epsind 0-24)
EPSINDMAX, BINDMAX = 25, 60


def pg(bind):
    return round(1e12 * B0arr[bind])


def epstag(epsind):
    return round(epsarr[epsind], 3)


print("loading xe2fullk (all binds x 25 epsind) ...", time.ctime())
xe2fullk = np.zeros((len(zarr), len(B0arr), len(epsarr)))
for bind in range(len(B0arr)):
    for epsind in range(NEPS):
        xe2fullk[:, bind, epsind] = np.load(DATA / f"xe2/xe2_B_{pg(bind)}pG_e{epstag(epsind)}.npy")

print("loading deltamdeltambar_saha (needed for xesaha2 comparison, all binds) ...", time.ctime())
deltamdeltambar_saha = np.zeros((len(karr), len(zarr), len(B0arr)))
for bind in range(0, len(B0arr), 10):
    saha_dir = DATA / f"ang_avg/saha/B_{pg(bind)}pG"
    for kind in range(len(karr)):
        with open(saha_dir / f"ang_avg_k{kind}.pkl", "rb") as f:
            dsaha = pickle.load(f)
        deltamdeltambar_saha[kind, :, bind] = dsaha["deltamdeltambar"]

Lambda = 1e3 * cons.mpc
deltamrmssaha = np.zeros((1300, len(B0arr), len(epsarr)))
for bind in range(0, len(B0arr), 10):
    for epsind in range(10):
        for zind in range(len(zarr)):
            deltamrmssaha[zind, bind, epsind] = (abs(epsarr[epsind]) / 4 * (Lambda / (2 * np.pi))**epsarr[epsind] *
                splint(karr[-1], karr[0], splrep(np.flip(karr), np.flip(karr**(epsarr[epsind] - 1) * deltamdeltambar_saha[:, zind, bind]))))

print("done loading;", time.ctime())

# ---------------------------------------------------------------------------
# Local perturbed 3LA (Sobolev) comparison for Figure 07 (dashed curves).
# Evolves the local perturbed-recombination ODE pars.RHSsobpert2 in patches
# modulated by +/- eps_mod times the local delta_b and Theta transfer
# functions, extracts the second-order mean shift
# [x_e(+eps) + x_e(-eps) - 2 x_e(0)] + eps*delta*[x_e(+eps) - x_e(-eps)],
# then angle-averages and integrates over the PMF spectrum.
# Uses the first 44 wavenumbers (modes already free-streaming by z = 1900,
# where the TLA transfer functions start) and field strengths
# bind = 0, 10, ..., 50, reading
#   Tfs/B_{B}pG/TCRmag_k{kind}.npy   (TCR_Tfs.py)
#   Tfs/B_{B}pG/FSRTLAmag_k{kind}.npy (FSR_TLA_Tfs.py)
# ---------------------------------------------------------------------------
from scipy.integrate import odeint
import multiprocessing as _mp
import os as _os

xe_full = pars.xe_full
eps_mod = 0.1  # modulation amplitude of the local patches
bindarr_loc = [0, 10, 20, 30, 40, 50]
NK_LOC = 44
NTHETA = 17
zgrid_loc = np.logspace(np.log10(1900), np.log10(600), 10**4)
thetaarr_full = np.linspace(0, np.pi, 33)

epstest = [0, 2, 4, 6, 8, 9]  # epsarr indices for Figure 07: [0,2,4,6,8] for the right panel, 9 (eps=-0.1) for the left
kLamb = 2 * np.pi / Lambda

FIG07_CACHE_DIR = REPO_ROOT / "analysis" / ".fig07_local3la_cache"
_fig07_cache_path = FIG07_CACHE_DIR / "fullavg.npz"

print("computing local perturbed-3LA curves for Figure 07 ...", time.ctime())
if _fig07_cache_path.exists():
    fullavg = np.load(_fig07_cache_path)["fullavg"]
    print("loaded from cache;", time.ctime())
else:
    # This block (spline-building + parallel ODE solve + angle/k-averaging) is
    # the expensive step for Figure 07 (~40s even with the ODE solve
    # parallelized across all cores), so its final result (fullavg) is cached
    # to disk -- delete FIG07_CACHE_DIR if the underlying pipeline data
    # changes and this needs to be recomputed.
    print("building delta/Theta transfer-function splines for the local-3LA curves ...", time.ctime())
    deltaspl = np.empty((NK_LOC, len(bindarr_loc), NTHETA), dtype=object)
    Thetaspl = np.empty((NK_LOC, len(bindarr_loc), NTHETA), dtype=object)
    Hgrid_loc = pars.H(zgrid_loc[::-1])
    for bcount, bind in enumerate(bindarr_loc):
        Bdir = DATA / f"Tfs/B_{pg(bind)}pG"
        for kind in range(NK_LOC):
            tcr_b_end = np.load(Bdir / f"TCRmag_k{kind}.npy")[:, 3, -1]
            tla = np.load(Bdir / f"FSRTLAmag_k{kind}.npy")
            for thetaind in range(NTHETA):
                deltaspl[kind, bcount, thetaind] = splrep(zgrid_loc[::-1], tcr_b_end[thetaind] * tla[thetaind, 0][::-1])
                Thetaspl[kind, bcount, thetaind] = splrep(zgrid_loc[::-1],
                                                          Hgrid_loc * tcr_b_end[thetaind] * tla[thetaind, 1][::-1])

    def deltafunc(z, kind, bcount, thetaind):
        return splev(z, deltaspl[kind, bcount, thetaind])

    def Thetafunc(z, kind, bcount, thetaind):
        return splev(z, Thetaspl[kind, bcount, thetaind])

    def _solve_local_pair(args):
        kind, bcount, thetaind = args
        xp = odeint(pars.RHSsobpert2, xe_full(1900), zarr,
                    args=(eps_mod, deltafunc, Thetafunc, kind, bcount, thetaind)).flatten()
        xm = odeint(pars.RHSsobpert2, xe_full(1900), zarr,
                    args=(-eps_mod, deltafunc, Thetafunc, kind, bcount, thetaind)).flatten()
        return kind, bcount, thetaind, xp, xm

    print("solving local perturbed-3LA ODEs (44 k x 6 B0 x 17 theta, +/-eps; parallel) ...", time.ctime())
    xeplus = np.zeros((NK_LOC, len(bindarr_loc), NTHETA, len(zarr)))
    xeminus = np.zeros((NK_LOC, len(bindarr_loc), NTHETA, len(zarr)))
    _tasks = [(k, b, t) for k in range(NK_LOC) for b in range(len(bindarr_loc)) for t in range(NTHETA)]
    with _mp.get_context("fork").Pool(processes=min(len(_tasks), _os.cpu_count() or 1)) as _pool:
        for kind, bcount, thetaind, xp, xm in _pool.imap_unordered(_solve_local_pair, _tasks, chunksize=8):
            xeplus[kind, bcount, thetaind] = xp
            xeminus[kind, bcount, thetaind] = xm
    print("local-3LA ODEs done;", time.ctime())

    # mirror theta in [0, pi/2] to [pi/2, pi] and take the second-order combination
    xeplusfull = np.zeros((NK_LOC, len(bindarr_loc), 33, len(zarr)))
    xeminusfull = np.zeros((NK_LOC, len(bindarr_loc), 33, len(zarr)))
    deltafull = np.zeros((NK_LOC, len(bindarr_loc), 33, len(zarr)))
    xeplusfull[:, :, :NTHETA, :] = xeplus
    xeminusfull[:, :, :NTHETA, :] = xeminus
    for kind in range(NK_LOC):
        for thetaind in range(NTHETA):
            for bcount in range(len(bindarr_loc)):
                deltafull[kind, bcount, thetaind, :] = deltafunc(zarr, kind, bcount, thetaind)
    for i in range(NTHETA):
        xeplusfull[:, :, 32 - i, :] = xeplusfull[:, :, i, :]
        xeminusfull[:, :, 32 - i, :] = xeminusfull[:, :, i, :]
        deltafull[:, :, 32 - i, :] = deltafull[:, :, i, :]

    xefullhold = ((1 + eps_mod * deltafull) * xeplusfull + (1 - eps_mod * deltafull) * xeminusfull
                  - 2 * np.broadcast_to(xe_full(zarr).reshape(1, 1, 1, len(zarr)),
                                        (NK_LOC, len(bindarr_loc), 33, len(zarr))))
    xefullhold[:, :, 16, :] = 0

    print("angle- and wavenumber-averaging the local-3LA shift ...", time.ctime())
    fullbar = np.zeros((NK_LOC, len(bindarr_loc), len(zarr)))
    for zind in range(len(zarr)):
        for kind in range(NK_LOC):
            for bcount in range(len(bindarr_loc)):
                fullbar[kind, bcount, zind] = splint(thetaarr_full[0], thetaarr_full[-1],
                                                     splrep(thetaarr_full, np.sin(thetaarr_full) * xefullhold[kind, bcount, :, zind]))

    fullavg = np.zeros((len(epstest), len(bindarr_loc), len(zarr)))
    for zind in range(len(zarr)):
        for bcount in range(len(bindarr_loc)):
            for i, ei in enumerate(epstest):
                fullhold = splrep(karr[:NK_LOC][::-1], (karr[:NK_LOC][::-1])**(epsarr[ei] - 1) * fullbar[::-1, bcount, zind])
                fullavg[i, bcount, zind] = abs(epsarr[ei]) / (4 * kLamb**epsarr[ei]) * splint(karr[NK_LOC], karr[0], fullhold)
    FIG07_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(_fig07_cache_path, fullavg=fullavg)
    print("local-3LA curves ready;", time.ctime())

# ---------------------------------------------------------------------------
# Figure 07: xe2(z) curves for representative B0/eps
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

epsind = 9
binds_07 = tuple(range(0, 41, 10))
blue_shades_07 = plt.cm.BuPu(np.linspace(0.95, 0.35, len(binds_07)))
for bcount, bind in enumerate(binds_07):
    color = blue_shades_07[bcount]
    xesaha2 = -xesaha_full(zarr) * deltamrmssaha[:, bind, epsind] * (1 - xesaha_full(zarr)) / (2 - xesaha_full(zarr))**3
    axs[0].plot(zarr, xe2fullk[:, bind, epsind], label=f'$B_0$={pg(bind)} pG', color=color)
    axs[0].plot(zarr, xesaha2, ':', color=color)
    axs[0].plot(zarr, fullavg[-1, bcount] / 2 / eps_mod**2, '--', color=color)
axs[0].set_title(f'$\\epsilon=${round(epsarr[epsind], 2)}')

bind = 10
epsinds_07 = tuple(range(0, 10, 2))
blue_shades_07b = plt.cm.BuPu(np.linspace(0.95, 0.35, len(epsinds_07)))
for epscount, epsind in enumerate(epsinds_07):
    color = blue_shades_07b[epscount]
    xesaha2 = -xesaha_full(zarr) * deltamrmssaha[:, bind, epsind] * (1 - xesaha_full(zarr)) / (2 - xesaha_full(zarr))**3
    axs[1].plot(zarr, xe2fullk[:, bind, epsind], label=f'$\\epsilon$={round(epsarr[epsind], 2)}', color=color)
    axs[1].plot(zarr, xesaha2, ':', color=color)
    axs[1].plot(zarr, fullavg[epscount, 1] / 2 / eps_mod**2, '--', color=color)

legend1 = axs[0].legend(loc='lower right', frameon=True)
axs[1].legend()
axs[1].set_title(f'$B_0=${pg(bind)} pG')
axs[0].set_xlabel('Redshift z', fontsize=14)
axs[1].set_xlabel('Redshift z', fontsize=14)
axs[0].set_ylabel(r'$\Delta x_e$', fontsize=14)
from matplotlib.lines import Line2D
custom_legend = [
    Line2D([0], [0], color='black', linestyle='-', label='RT 3LA'),
    Line2D([0], [0], color='black', linestyle='--', label='Local 3LA'),
    Line2D([0], [0], color='black', linestyle=':', label='Local Saha'),
]
legend2 = axs[0].legend(handles=custom_legend, loc='lower left', frameon=True)
axs[0].add_artist(legend1)
plt.tight_layout()
plt.savefig(PLOTDIR / "07_xe2_curves.png", dpi=150)
plt.close(fig)
print("saved 07_xe2_curves.png")

# ---------------------------------------------------------------------------
# Figure 08: min(xe2) heatmap vs (B0, eps)
# ---------------------------------------------------------------------------
xe2_min = np.min(xe2fullk, axis=0)
interp_func = RegularGridInterpolator((1e12 * B0arr, epsarr), xe2_min)
num_points = 1000
epsarr_dense = np.linspace(epsarr[0], epsarr[-1], num_points)
B0arr_dense = 1e12 * np.linspace(B0arr[0], B0arr[-1], num_points)
B0_grid, eps_grid = np.meshgrid(B0arr_dense, epsarr_dense, indexing='ij')
pts = np.array([B0_grid.flatten(), eps_grid.flatten()]).T
xe2_interp_2d = interp_func(pts).reshape(num_points, num_points)

fig, ax = plt.subplots(figsize=(8, 6))
cmap = ax.imshow(xe2_interp_2d, origin='lower', aspect='auto',
                 extent=[epsarr_dense[0], epsarr_dense[-1], B0arr_dense[0], B0arr_dense[-1]], cmap='twilight')
cbar = plt.colorbar(cmap, ax=ax)
cbar.set_label(r'min($\delta x_e^{(2)}$)')
ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel(r'$B_0 \, (\mathrm{pG})$')
ax.set_xlim([epsarr[0], epsarr[EPSINDMAX]])
ax.set_ylim([1e12 * B0arr[0], 1e12 * B0arr[BINDMAX - 1]])
plt.tight_layout()
plt.savefig(PLOTDIR / "08_min_xe2_heatmap.png", dpi=150)
plt.close(fig)
print("saved 08_min_xe2_heatmap.png")

# ---------------------------------------------------------------------------
# Figure 09: visibility function + peak-shift heatmap
# ---------------------------------------------------------------------------
from scipy.integrate import quad


def taudot(z):
    return -(pars.nh(z) * pars.xe_full(z) * cons.sigmat) / (1 + z)


def taudotinteg(z):
    return -cons.c * (pars.nh(z) * pars.xe_full(z) * cons.sigmat) / (1 + z) / pars.H(z)


def visib(z):
    return -taudot(z) * np.exp(-quad(taudotinteg, z, 0)[0])


def tau0(z):
    return -quad(taudotinteg, z, 0)[0]


print("computing background visibility (tau0/visibarr, parallel) ...", time.ctime())
# Each z is an independent nested-quad integral from z down to 0 -- embarrassingly
# parallel across the 1300 z-points. Fork context avoids re-import/pickling overhead.
import multiprocessing as _mp
import os as _os

_fork_ctx = _mp.get_context("fork")


def _tau0_visib_one(zind):
    z = zarr[zind]
    return zind, tau0(z), visib(z)


visibarr = np.zeros(len(zarr))
tau0arr = np.zeros(len(zarr))
_nworkers = min(len(zarr), _os.cpu_count() or 1)
with _fork_ctx.Pool(processes=_nworkers) as _pool:
    for zind, t0, v in _pool.imap_unordered(_tau0_visib_one, range(len(zarr)), chunksize=8):
        tau0arr[zind] = t0
        visibarr[zind] = v
print(f"background visibility done ({_nworkers} workers);", time.ctime())

print("loading visibprefactor (all binds x 25 epsind) ...", time.ctime())
visibprefactor = np.zeros((len(zarr), len(B0arr), len(epsarr)))
for bind in range(len(B0arr)):
    for epsind in range(NEPS):
        visibprefactor[:, bind, epsind] = np.load(DATA / f"visib/visib_B_{pg(bind)}pG_e{epstag(epsind)}.npy")

zstartind = 0
zarrfine = np.linspace(600, zarr[zstartind], 100000)
peakarr = np.zeros((len(B0arr), len(epsarr)))
relshift = np.zeros((len(B0arr), len(epsarr)))
for bind in range(len(B0arr)):
    for epsind in range(NEPS):
        peakarr[bind, epsind] = zarrfine[np.argmax(splev(zarrfine, splrep(zarr[:zstartind:-1],
                                          visibprefactor[:zstartind:-1, bind, epsind] * visibarr[:zstartind:-1])))]

no_field_peak_z = zarrfine[np.argmax(splev(zarrfine, splrep(zarr[::-1], visibarr[::-1])))]
relshift[:BINDMAX, :EPSINDMAX] = peakarr[:BINDMAX, :EPSINDMAX] - no_field_peak_z
relshift_smoothed = gaussian_filter(relshift, sigma=1.1)
interp_func2 = RegularGridInterpolator((1e12 * B0arr, epsarr), relshift_smoothed)
relshift_interp_2d = interp_func2(pts).reshape(num_points, num_points)

fig, ax = plt.subplots(figsize=(8, 6))
cmap = ax.imshow(relshift_interp_2d, origin='lower', aspect='auto', vmin=0, vmax=0.75,
                 extent=[epsarr_dense[0], epsarr_dense[-1], B0arr_dense[0], B0arr_dense[-1]], cmap='twilight')
cbar = plt.colorbar(cmap, ax=ax)
cbar.set_label(r'$\Delta z_*$')
ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel(r'$B_0 \, (\mathrm{pG})$')
plt.tight_layout()
plt.savefig(PLOTDIR / "09_visibility_shift_heatmap.png", dpi=150)
plt.close(fig)
print("saved 09_visibility_shift_heatmap.png")

# ---------------------------------------------------------------------------
# Figures 10-11: clumping factor heatmap at z=1295 + side-by-side with visibility shift
# ---------------------------------------------------------------------------
print("loading cross_corr pkl deltamrms (all binds x 25 epsind) ...", time.ctime())
deltamrms = np.zeros((len(zarr), len(B0arr), len(epsarr)))
for bind in range(len(B0arr)):
    for epsind in range(NEPS):
        with open(DATA / f"cross_corr/cross_corr_B_{pg(bind)}pG_e{epstag(epsind)}.pkl", "rb") as f:
            cc = pickle.load(f)
        deltamrms[:, bind, epsind] = abs(epsarr[epsind]) / 4 * (Lambda / (2 * np.pi))**epsarr[epsind] * cc["deltamrms"]

zind_1295 = int(np.argmin(np.abs(zarr - 1295)))
interp_func3 = RegularGridInterpolator((1e12 * B0arr, epsarr), deltamrms[zind_1295])
dmrms_interp_2d = interp_func3(pts).reshape(num_points, num_points)

fig, ax = plt.subplots(figsize=(8, 6))
cmap = ax.imshow(dmrms_interp_2d, origin='lower', aspect='auto',
                 extent=[epsarr_dense[0], epsarr_dense[-1], B0arr_dense[0], B0arr_dense[-1]], cmap='twilight')
cbar = plt.colorbar(cmap, ax=ax)
cbar.set_label(r'Clumping factor b at $z = 1295$')
ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel(r'$B_0 \, (\mathrm{pG})$')
ax.set_xlim([epsarr[0], epsarr[EPSINDMAX]])
ax.set_ylim([1e12 * B0arr[0], 1e12 * B0arr[BINDMAX]])
plt.tight_layout()
plt.savefig(PLOTDIR / "10_clumping_heatmap.png", dpi=150)
plt.close(fig)
print("saved 10_clumping_heatmap.png")

fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
im = axs[0].imshow(dmrms_interp_2d, origin='lower', aspect='auto',
                   extent=[epsarr_dense[0], epsarr_dense[-1], B0arr_dense[0], B0arr_dense[-1]], cmap='twilight')
im1 = axs[1].imshow(relshift_interp_2d, origin='lower', aspect='auto', vmin=0, vmax=0.72,
                    extent=[epsarr_dense[0], epsarr_dense[-1], B0arr_dense[0], B0arr_dense[-1]], cmap='twilight')
cbar = plt.colorbar(im, ax=axs[0])
cbar.set_label('Clumping factor b $= \\langle \\delta_b^2 \\rangle$ at $z=1295$', fontsize=12, rotation=270, labelpad=20)
cbar.ax.tick_params(size=12, labelsize=12)
cbar1 = plt.colorbar(im1, ax=axs[1])
cbar1.set_label(r'Shift to peak of visibility function $\Delta z_*$', fontsize=12, rotation=270, labelpad=20)
cbar.ax.tick_params(size=12, labelsize=12)

axs[0].set_xlabel(r'$\epsilon$', fontsize=16); axs[0].set_ylabel(r'$B_0 \, (\mathrm{pG})$', fontsize=16)
axs[1].set_xlabel(r'$\epsilon$', fontsize=16)
axs[0].set_xlim([epsarr[0], epsarr[EPSINDMAX - 1]]); axs[0].set_ylim([1e12 * B0arr[0], 1e12 * B0arr[BINDMAX - 1]])
axs[1].set_xlim([epsarr[0], epsarr[EPSINDMAX - 1]]); axs[1].set_ylim([1e12 * B0arr[0], 1e12 * B0arr[BINDMAX - 1]])
axs[0].tick_params(size=12, labelsize=12, labelbottom=True)
axs[1].tick_params(size=12, labelsize=12, labelbottom=True)

plt.tight_layout()
plt.savefig(PLOTDIR / "11_clumping_and_visibility_side_by_side.png", dpi=150)
plt.close(fig)
print("saved 11_clumping_and_visibility_side_by_side.png")

print("all xe2/visibility plots done;", time.ctime())
