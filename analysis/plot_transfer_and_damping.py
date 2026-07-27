"""
Transfer-function and damping figures:
  - 01_damping_length_vs_theta.png : damping wavelength vs redshift for
    FM/SM/Alfven modes at two field strengths, over a range of angles theta
  - 02_stokes_by_evolution_z4500.png : T_by / T_delta evolution in the FSR,
    initialized at z_FS vs at z = 4500
  - 03_transfer_scatter.png : transfer-function (z,k) scatter panels with
    horizon-crossing / free-streaming / conservation / balance overlays
  - 04_fm_amplitude_ratio.png : density-seeded vs field-seeded transfer-
    function ratio and relative primordial contributions to b_y

Inputs, read from the transfer-function pipeline outputs under $PMHD_OUTDIR:
  Tfs/B_{B}pG/TCR{mag,alf}_k{kind}.npy          (TCR_Tfs.py)
  Tfs/B_{B}pG/FSRsaha{mag,alf}_k{kind}.npy      (FSR_saha_Tfs.py)
  Tfs/B_{B}pG/FSRTLA{mag,alf}_k{kind}.npy       (FSR_TLA_Tfs.py)
  Tfs/FM/B_{B}pG/TCRmag_k{kind}.npy             (TCR_Tfs_FM.py)
  Tfs/z4500/B_{B}pG/FSRsaha{mag,alf}_k{kind}.npy (FSR_saha_Tfs_z4500.py)
for field strengths bind = 0, 10, ..., 60 and all 69 wavenumbers
(the z4500 variant only for binds 0/30/60 and kinds 3, 13, ..., 63).

Configuration (environment variables):
  PMHD_OUTDIR  : pipeline output directory to read data from
                 (default: <repo>/src/pmhd/data/outputs, the same default the
                 src/pmhd/physics scripts write to)
  PMHD_PLOTDIR : directory to write figures to (default: <repo>/analysis/plots)

Run:  python analysis/plot_transfer_and_damping.py
"""
import os, sys, math, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

from scipy.interpolate import splrep, splev

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pmhd import cons, pars
from pmhd.data.grids import (
    k_grid, eps_grid, z_grid, theta_gridfull,
    load_or_generate_B0arr, load_or_generate_z_arrays,
)

DATA = Path(os.environ.get("PMHD_OUTDIR", str(REPO_ROOT / "src/pmhd/data/outputs")))
PLOTDIR = Path(os.environ.get("PMHD_PLOTDIR", str(REPO_ROOT / "analysis/plots")))
PLOTDIR.mkdir(parents=True, exist_ok=True)

epsarr = eps_grid()
B0arr = load_or_generate_B0arr()
karr = k_grid()
zcrossarr, zfsarr = load_or_generate_z_arrays()
thetaarr = theta_gridfull()
zarr = z_grid()

bindarr = [0, 10, 20, 30, 40, 50, 60]
zsteps = 10**4


def pg(bind):
    return round(1e12 * B0arr[bind])


print("loading TCR/FSR/FSRTLA/FM transfer functions ...", time.ctime())

TCRmag = np.zeros((len(karr), len(bindarr), 17, 4, zsteps))
TCRalf = np.zeros((len(karr), len(bindarr), 17, 2, zsteps))
FSRsahamag = np.zeros((len(karr), len(bindarr), 17, 4, zsteps))
FSRsahaalf = np.zeros((len(karr), len(bindarr), 17, 2, zsteps))
FSRTLAmag = np.zeros((len(karr), len(bindarr), 17, 5, zsteps))
FSRTLAalf = np.zeros((len(karr), len(bindarr), 17, 2, zsteps))

for bcount, bind in enumerate(bindarr):
    Bdir = DATA / f"Tfs/B_{pg(bind)}pG"
    for kind in range(len(karr)):
        TCRmag[kind, bcount] = np.load(Bdir / f"TCRmag_k{kind}.npy")
        TCRalf[kind, bcount] = np.load(Bdir / f"TCRalf_k{kind}.npy")
        FSRsahamag[kind, bcount] = np.load(Bdir / f"FSRsahamag_k{kind}.npy")
        FSRsahaalf[kind, bcount] = np.load(Bdir / f"FSRsahaalf_k{kind}.npy")
        FSRTLAmag[kind, bcount] = np.load(Bdir / f"FSRTLAmag_k{kind}.npy")
        FSRTLAalf[kind, bcount] = np.load(Bdir / f"FSRTLAalf_k{kind}.npy")

TCRFM = np.zeros((len(karr), 7, 17, 4, zsteps))
for bcount, bind in enumerate(np.arange(0, 61, 10)):
    Bdir = DATA / f"Tfs/FM/B_{pg(bind)}pG"
    for kind in range(len(karr)):
        TCRFM[kind, bcount] = np.load(Bdir / f"TCRmag_k{kind}.npy")

zarraymaster = np.zeros((len(karr), zsteps))
for kind in range(len(karr)):
    zarraymaster[kind] = np.logspace(np.log10(zcrossarr[kind]), np.log10(zfsarr[kind]), num=zsteps)

zfsarraymaster = np.zeros((len(karr), zsteps))
for kind in range(len(karr)):
    zfsarraymaster[kind] = np.logspace(np.log10(zfsarr[kind]), np.log10(600), num=zsteps)

print("done loading;", time.ctime())

# ---------------------------------------------------------------------------
# Damping curves from the oscillation ENVELOPE: the record (running future-
# max) envelope of |T| along the stitched TCR -> FSR-saha evolution, crossed
# at e^-1 and log-log interpolated. The raw first |T| <= e^-1 crossing fires
# on the first acoustic downswing (a sound-horizon phase condition,
# eta-independent); the record envelope tracks the actual erosion of the
# oscillation amplitude, matches the Hu & White (1997) k_D to 1-2% on the
# FM/acoustic branch, and is equally well defined for overdamped
# (non-oscillatory) modes.
# ---------------------------------------------------------------------------
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

EINV = np.exp(-1)


def smooth_env_cross(s, zs):
    """Damping redshift: e^-1 crossing of the record envelope (the points
    where |T| equals its future maximum, s[i] = max(s[i:])). The record set
    contains the oscillation peak maxima AND every sample of a monotone
    tail, and its heights are monotone non-increasing in time, so it has a
    single, well-defined crossing (log-log interpolated). Returns None if
    the mode has not damped below e^-1 by the end of the series (tail
    guard, last 5%)."""
    if s[-int(0.05 * len(s)):].max() >= EINV:
        return None
    M = np.maximum.accumulate(s[::-1])[::-1]
    rec = np.where(s >= M)[0]
    h = np.maximum(s[rec], 1e-300)
    below = np.where(h <= EINV)[0]
    if len(below) == 0 or below[0] == 0:
        return None
    j = below[0]
    f = (np.log(EINV) - np.log(h[j - 1])) / (np.log(h[j]) - np.log(h[j - 1]))
    return np.exp(np.log(zs[rec[j - 1]]) + f * (np.log(zs[rec[j]]) - np.log(zs[rec[j - 1]])))


def stitched_env_cross(kind, T_tcr, T_fsr_scaled):
    s = np.concatenate([np.abs(T_tcr), np.abs(T_fsr_scaled)])
    zs = np.concatenate([zarraymaster[kind], zfsarraymaster[kind]])
    return smooth_env_cross(s, zs)


def hull(zlist):
    """Monotonized damped-region boundary, returned as (z, lambda) arrays.
    kind increases -> lambda increases, and 'everything smaller than
    lambda_D(z) is damped at z' requires z to decrease monotonically along
    the curve. Fold points (a recovery peak grazing e^-1 can make a
    slightly larger mode register as damped earlier than a smaller one) are
    DROPPED rather than clamped, then the kept points are PCHIP-resampled
    in log-log for a smooth C1 curve. Finally the boundary is extended flat
    to z=600: after the last mode crosses e^-1 the drag has collapsed
    through recombination and no larger scale ever damps, so the
    largest-damped-scale boundary freezes at its final value."""
    zmin = np.inf
    z_out, lam_out = [], []
    for i, z in enumerate(zlist):
        if z < zmin:
            z_out.append(z)
            lam_out.append(2 * np.pi / (karr[i] * cons.mpc))
            zmin = z
    z_out, lam_out = np.array(z_out), np.array(lam_out)
    if len(z_out) >= 4:
        logl, logz = np.log(lam_out), np.log(z_out)
        p = PchipInterpolator(logl, logz)
        dense = np.linspace(logl[0], logl[-1], 600)
        z_out, lam_out = np.exp(p(dense)), np.exp(dense)
    if len(z_out) and z_out[-1] > 600.0:
        z_out = np.append(z_out, 600.0)
        lam_out = np.append(lam_out, lam_out[-1])
    return z_out, lam_out


# FM FSR continuation: the production FSR runs are field-seeded ([0,0,0,1]),
# so the density-seeded FM mode needs its own free-streaming extension.
# FSRsahamag_hom evolves the same (delta, Theta, Phiy, b) state vector as
# TCmag, so each mode's FSR integration is seeded with its full FM TCR
# endpoint state (exact handoff). Without this, modes with lambda >~ 26 Mpc
# that dip below e^-1 during recombination but gravitationally regrow after
# decoupling would be miscounted as damped. Cached next to the FM data;
# delete the cache file to force recomputation.
_fm_fsr_cache = DATA / f"Tfs/FM/B_{pg(0)}pG/FSRdelta_fullstate.npy"
if _fm_fsr_cache.exists():
    FMfsr = np.load(_fm_fsr_cache)
else:
    print("computing FM FSR continuation (full-state handoff) ...", time.ctime())
    FMfsr = np.zeros((len(karr), zsteps))
    for kind in range(len(karr)):
        endpoint = TCRFM[kind, 0, 0, :, -1]

        def _rhs_logz(lz, v, _k=karr[kind]):
            z = np.exp(lz)
            dv = pars.FSRsahamag_hom(z, v, _k, thetaarr[0], B0arr[0],
                                     pars.xesaha_full, pars.xesaha_full_He)
            return [x * z for x in dv]

        lzgrid = np.linspace(np.log(zfsarr[kind]), np.log(600), zsteps)
        sol = solve_ivp(_rhs_logz, [lzgrid[0], lzgrid[-1]], list(endpoint),
                        method='LSODA', dense_output=True, atol=1e-9, rtol=1e-6)
        FMfsr[kind] = sol.sol(lzgrid)[0]
    np.save(_fm_fsr_cache, FMfsr)

FMenv = []
for kind in range(len(karr)):
    ze = smooth_env_cross(
        np.concatenate([np.abs(TCRFM[kind, 0, 0, 0, :]), np.abs(FMfsr[kind])]),
        np.concatenate([zarraymaster[kind], zfsarraymaster[kind]]))
    if ze is None:
        break
    FMenv.append(ze)
FMenv = hull(FMenv)

SMdamping_dict = {}
Adamping_dict = {}
thetaindarr_all = range(int((len(thetaarr) - 1) / 2) + 1)

for bcount, bind in enumerate(bindarr):
    for thetaind in thetaindarr_all:
        key = f"bind={bind},theta={thetaind}"
        SMdamping_dict[key] = []
        for kind in range(len(karr)):
            ze = stitched_env_cross(
                kind, TCRmag[kind, bcount, thetaind, 3, :],
                TCRmag[kind, bcount, thetaind, 3, -1] * FSRsahamag[kind, bcount, thetaind, 3, :])
            if ze is None:
                break
            SMdamping_dict[key].append(ze)
        SMdamping_dict[key] = hull(SMdamping_dict[key])

        Adamping_dict[key] = []
        for kind in range(len(karr)):
            ze = stitched_env_cross(
                kind, TCRalf[kind, bcount, thetaind, 1, :],
                TCRalf[kind, bcount, thetaind, 1, -1] * FSRsahaalf[kind, bcount, thetaind, 1, :])
            if ze is None:
                break
            Adamping_dict[key].append(ze)
        Adamping_dict[key] = hull(Adamping_dict[key])

print("damping curves built;", time.ctime())

# ---------------------------------------------------------------------------
# Figure 01: damping wavelength vs anisotropy angle, two field strengths overlaid
# ---------------------------------------------------------------------------
from scipy.interpolate import interp1d
import matplotlib.lines as mlines

thetalabalarr = np.array([r'$0$', r'$\pi/32$', r'$\pi/16$', r'$3\pi/32$', r'$\pi/8$',
                          r'$5\pi/32$', r'$3\pi/16$', r'$7\pi/32$', r'$\pi/4$', r'$9\pi/32$', r'$5\pi/16$',
                          r'$11\pi/32$', r'$3\pi/8$', r'$13\pi/32$', r'$7\pi/16$', r'$15\pi/32$', r'$\pi/2$'])

fig, ax = plt.subplots()
ax.tick_params(size=12, labelsize=12, labelbottom=True)

bind = bindarr[0]
bind2 = bindarr[5]
# At 20 pG the theta=pi/2 SM/Alfven modes never damp below e^-1 (the b-to-
# compressional coupling ~ B0^2 sin(theta) is too weak; verified min |T| =
# 0.50 over all k), so the 20 pG band plots/labels theta=15pi/32 -- the
# largest angle that damps at all -- as its darkest curve and shading edge.
red_thetas = [0, 12, 16]
green_thetas = [0, 12, 15]
red_colors = [plt.cm.Reds(i) for i in np.linspace(0.6, 1, len(red_thetas))]
green_colors = [plt.cm.Greens(i) for i in np.linspace(0.2, 1, len(green_thetas))]

fm_z, fm_lam = FMenv
ax.loglog(fm_z, fm_lam, c='k', linestyle='--')
ax.text(1.5e5, 0.12, 'FM/Acoustic', color='k', fontsize=12, rotation=35)


def draw_band(bind_, colors, alpha, fill_hi_theta, theta_list, fillcolor):
    for thetacount, thetaind in enumerate(theta_list):
        key = f"bind={bind_},theta={thetaind}"
        z_sm, lam_sm = SMdamping_dict[key]
        line, = ax.loglog(z_sm, lam_sm, c=colors[thetacount])
        z_a, lam_a = Adamping_dict[key]
        ax.loglog(z_a, lam_a, ':', color=line.get_color())
    z0, l0 = SMdamping_dict[f"bind={bind_},theta=0"]
    zH, lH = SMdamping_dict[f"bind={bind_},theta={fill_hi_theta}"]
    if len(z0) and len(zH):
        xvals = np.concatenate(([z0[0]], zH))
        y1 = np.concatenate(([l0[0]], lH))
        f2 = interp1d(z0[::-1], l0[::-1], bounds_error=False, fill_value="extrapolate")
        y2i = f2(xvals[::-1])[::-1]
        ax.fill_between(xvals, y1, y2i, where=(y1 <= y2i), alpha=alpha, color=fillcolor)


draw_band(bind, red_colors, 0.1, 16, red_thetas, 'red')
draw_band(bind2, green_colors, 0.3, 15, green_thetas, 'green')

ax.loglog(zfsarr, 2 * np.pi / (karr * cons.mpc), 'b--')
ax.text(950, 4e0, r'$\ell_{MFP}^{\gamma}$', fontsize=12, color='b')

ax.invert_xaxis()
ax.set_xlim([1e8, 1e2])
ax.set_xlabel(r'$z$', fontsize=14)
ax.set_ylabel(r'$\lambda_D$ (Mpc)', fontsize=14)
ax.set_ylim([1e-5, 3e2])

# consolidated B0 legend: legend fills columns first, so interleaving the
# 5000 pG / 20 pG handles gives row 1 = 5000 pG (reds), row 2 = 20 pG (greens)
hs = [mlines.Line2D([], [], color='white', label=f'$B_0 = {pg(bind)}$ pG'),
      mlines.Line2D([], [], color='white', label=f'$B_0 = {pg(bind2)}$ pG')]
for tc in range(len(red_thetas)):
    hs.append(mlines.Line2D([], [], color=red_colors[tc],
                            label=r'$\theta = $' + thetalabalarr[red_thetas[tc]]))
    hs.append(mlines.Line2D([], [], color=green_colors[tc],
                            label=r'$\theta = $' + thetalabalarr[green_thetas[tc]]))
legend2 = ax.legend(handles=hs, loc='upper left', bbox_to_anchor=(0, 1.0), fontsize=12,
                    frameon=True, framealpha=0.5, ncol=1 + len(red_thetas),
                    handletextpad=0.1, handlelength=1, columnspacing=0.5)
ax.add_artist(legend2)

ax.axvspan(2000, 600, color='lightgray', alpha=0.3)


def Tcmb_eV(z):
    return (8.617333262e-5 * 2.7255) * (1 + z)


def inv_Tcmb_eV(T):
    return T / (8.617333262e-5 * 2.7255) - 1


secax = ax.secondary_xaxis('top', functions=(Tcmb_eV, inv_Tcmb_eV))
secax.set_xlabel("T (eV)", fontsize=14)
secax.set_xscale('log')
secax.tick_params(axis='x', size=12, labelsize=12, direction='out')

ax.text(1095, 1.3e-3, 'H rec', color='gray', fontsize=11, ha='center', va='bottom')

plt.tight_layout()
plt.savefig(PLOTDIR / "01_damping_length_vs_theta.png", dpi=150)
plt.close(fig)
print("saved 01_damping_length_vs_theta.png")

# ---------------------------------------------------------------------------
# Figure 02: Stokes T_by/T_delta evolution, with z4500 diagnostic overlay
# ---------------------------------------------------------------------------
xe_full = pars.xe_full

FSRsahamag_z4500 = np.zeros((len(karr), len(bindarr), 17, 4, zsteps))
z4500_binds = bindarr[::3]  # [0,30,60]
z4500_kinds = [k for k in range(3, 73, 10) if k < len(karr)]
for bcount3, bind in enumerate(z4500_binds):
    bcount_full = bindarr.index(bind)
    Bdir = DATA / f"Tfs/z4500/B_{pg(bind)}pG"
    for kind in z4500_kinds:
        FSRsahamag_z4500[kind, bcount_full] = np.load(Bdir / f"FSRsahamag_k{kind}.npy")

fig, axs = plt.subplots(2, 3, figsize=(18, 5), sharex=True)
axs = axs.ravel()
cmap = plt.get_cmap('tab10')
arrind = 3
thetaind = 8

for bax, bcount in enumerate(np.arange(0, 7, 3)):
    bind = bindarr[bcount]
    ax = axs[bax]
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.tick_params(axis='y', size=12, labelsize=12)
    for i, kind in enumerate(range(3, 43, 10)):
        color = cmap(i % 10)
        ax.plot(zfsarraymaster[kind], FSRsahamag[kind, bcount, thetaind, arrind, :],
                label=r'$k \approx 10^{%.0f} \, \mathrm{Mpc}^{-1}$' % np.arange(5, 1, -1)[i], color=color)
        ax.plot(np.logspace(np.log10(4500), np.log10(600), num=zsteps),
                FSRsahamag_z4500[kind, bcount, thetaind, arrind, :], linestyle='--', color=color)
    ax.set_xlim([600, 5000])
    ax.set_title(r'$B_0=$' + f'{pg(bindarr[bcount])} pG ', fontsize=16)
    if bax % 3 == 0:
        ax.set_ylabel(r'$T_{b_y}$', fontsize=16)
    else:
        ax.set_ylabel('')
    if bax == 2:
        ax.legend(fontsize=12)

arrind = 0
for bax, bcount in enumerate(np.arange(0, 7, 3)):
    bind = bindarr[bcount]
    ax = axs[bax + 3]
    ax.tick_params(size=12, labelsize=12, labelbottom=True)
    for i, kind in enumerate(range(3, 43, 10)):
        color = cmap(i % 10)
        ax.plot(zfsarraymaster[kind], FSRsahamag[kind, bcount, thetaind, arrind, :], color=color)
        ax.plot(np.logspace(np.log10(4500), np.log10(600), num=zsteps),
                FSRsahamag_z4500[kind, bcount, thetaind, arrind, :], linestyle='--', color=color)
    ax.set_xlim([600, 5000])
    if bax % 3 == 0:
        ax.set_ylabel(r'$T_{\delta}$', fontsize=16)
    else:
        ax.set_ylabel('')

plt.tight_layout()
plt.savefig(PLOTDIR / "02_stokes_by_evolution_z4500.png", dpi=150)
plt.close(fig)
print("saved 02_stokes_by_evolution_z4500.png")

# ---------------------------------------------------------------------------
# Figure 03: transfer-function (z,k) scatter panels, B0=20pG, thetaind=2
# ---------------------------------------------------------------------------
from astropy.cosmology import Planck18 as cosmo
from matplotlib.patches import ConnectionPatch

xesaha_full = pars.xesaha_full
fhe = cons.yhe / (4.0 * (1.0 - cons.yhe))

bind = 20
thetaind = 2
bcount20 = bindarr.index(bind)

tfsbx = TCRalf[:, bcount20, thetaind, 1, :].copy()
tfsby = TCRmag[:, bcount20, thetaind, 3, :].copy()
tfsdelta = TCRmag[:, bcount20, thetaind, 0, :].copy()

tfsfsbx = np.zeros((len(karr), zsteps))
tfsfsby = np.zeros((len(karr), zsteps))
tfsfsdelta = np.zeros((len(karr), zsteps))
zbalance = np.zeros(len(karr))

zarrcons = np.logspace(np.log10(600), 5, zsteps)
zarrcons2 = np.logspace(5, 8, zsteps)


def kcons(z):
    return 1 / np.sqrt(
        cons.c * pars.Btild(B0arr[bind])**2 * (1 + z)**2 / (pars.nh(z) * xe_full(z) * cons.sigmat * pars.H(z))
    )


def kcons2(z):
    return 1 / np.sqrt(
        cons.c * pars.Btild(B0arr[bind])**2 * (1 + z)**2 / (pars.nh(z) * cons.sigmat * pars.H(z))
    )


xh = 1 - cons.yhe
rhob0 = cosmo.Ob0 * cosmo.critical_density0.value

for kind in range(len(karr)):
    tfsfsbx[kind] = tfsbx[kind, -1] * FSRsahaalf[kind, bcount20, thetaind, 1, :]

    hold = tfsby[kind, -1] * FSRsahamag[kind, bcount20, :, :, :]

    xiy = hold[thetaind, 3] / np.cos(thetaarr[thetaind]) - hold[thetaind, 0] * np.tan(thetaarr[thetaind])
    xiz = -hold[thetaind, 0]
    va = cons.c**2 * pars.Btild(B0arr[bind])**2 / pars.R(zfsarraymaster[kind])
    cs = (cons.kb * pars.Tcmb(zfsarraymaster[kind]) * xh * (2 + 2 * fhe - fhe * xesaha_full(zfsarraymaster[kind])) / cons.mh
          / (2 - xesaha_full(zfsarraymaster[kind]))
          - 4 * np.pi * cons.G * rhob0 * (1 + zfsarraymaster[kind]) / karr[kind]**2)

    tension = va * np.sin(thetaarr[thetaind]) * np.cos(thetaarr[thetaind]) * xiy
    pressure = (cs + va * np.sin(thetaarr[thetaind])**2) * xiz
    hits = np.argwhere((abs(pressure) >= abs(tension)) & (abs(xiz) >= 1e-4))
    if len(hits) != 0:
        zbalance[kind] = zfsarraymaster[kind][hits[0][0]]

    tfsfsby[kind] = hold[thetaind, 3]
    tfsfsdelta[kind] = hold[thetaind, 0]

ztotalarraymaster = np.zeros((len(karr), 2 * zsteps))
for kind in range(len(karr)):
    ztotalarraymaster[kind, :zsteps] = zarraymaster[kind]
    ztotalarraymaster[kind, zsteps:] = zfsarraymaster[kind]

tfsbx_total = np.zeros((len(karr), 2 * zsteps))
tfsby_total = np.zeros((len(karr), 2 * zsteps))
tfsdelta_total = np.zeros((len(karr), 2 * zsteps))
for kind in range(len(karr)):
    tfsbx_total[kind, :zsteps] = tfsbx[kind, :]
    tfsbx_total[kind, zsteps:] = tfsfsbx[kind, :]
    tfsby_total[kind, :zsteps] = tfsby[kind, :]
    tfsby_total[kind, zsteps:] = tfsfsby[kind, :]
    tfsdelta_total[kind, :zsteps] = tfsdelta[kind, :]
    tfsdelta_total[kind, zsteps:] = tfsfsdelta[kind, :]

fig, axs = plt.subplots(2, 2, figsize=(20, 10))
red_colors5 = [plt.cm.twilight(i) for i in np.linspace(0.2, 1, 5)]
fig.suptitle(f'$B_0 = {pg(bind)}$ pG, $\\theta = $ {thetalabalarr[thetaind]}', fontsize=16)

k_vals = np.repeat(karr[:, None] * cons.mpc, tfsby_total.shape[1], axis=1).flatten()
z_vals = ztotalarraymaster.flatten()
tf_vals = np.clip(tfsby_total.flatten(), -1, 1)

axs[0, 1].tick_params(size=12, labelsize=12)
axs[1, 1].tick_params(size=12, labelsize=12)
axs[0, 0].tick_params(size=12, labelsize=12)
axs[1, 0].tick_params(size=12, labelsize=12)

sc = axs[0, 0].scatter(z_vals, k_vals, c=tf_vals, cmap='coolwarm', vmin=-1, vmax=1, s=1, rasterized=True)
cbar = fig.colorbar(sc, ax=axs[0, 0])
cbar.set_label(r'$T_{b_y}(k,z)$', fontsize=16, labelpad=35)
cbar.ax.tick_params(size=12, labelsize=12)

axs[0, 0].set_xlabel('Redshift z', fontsize=16)
axs[0, 0].set_ylabel('k $(Mpc)^{-1}$', fontsize=16)
axs[0, 0].plot(zfsarr, karr * cons.mpc, 'k', label=r'$\ell_{MFP}^{\gamma}$')
axs[0, 0].plot(zcrossarr, karr * cons.mpc, 'g', label=r'$\ell_{hor}$')
axs[0, 0].plot(zarrcons, kcons(zarrcons) * cons.mpc, color='purple', label=r'$k_{cons}$')
axs[0, 0].set_xlim(np.min(ztotalarraymaster), np.max(ztotalarraymaster))
axs[0, 0].set_ylim(np.min(karr * cons.mpc), np.max(karr * cons.mpc))
axs[0, 0].set_yscale('log'); axs[0, 0].set_xscale('log'); axs[0, 0].invert_xaxis()

# NOTE: tf_vals is deliberately reassigned here (to the T_delta array) --
# both the delta scatter's vmin/vmax AND the axs[1,1] ylim below use this
# reassigned tf_vals, not the T_by one above.
tf_vals = np.clip(tfsdelta_total.flatten(), -1, 1)
sc = axs[1, 0].scatter(z_vals, k_vals, c=tf_vals,
                       cmap='coolwarm', vmin=max(-1, -1.01 * max(abs(tf_vals))), vmax=min(1, 1.01 * max(abs(tf_vals))),
                       s=1, rasterized=True)
cbar = fig.colorbar(sc, ax=axs[1, 0])
cbar.set_label(r'$T_{\delta}(k,z)$', fontsize=16, labelpad=35)
cbar.ax.tick_params(size=12, labelsize=12)

axs[1, 0].set_xlabel('Redshift z', fontsize=16)
axs[1, 0].set_ylabel('k $(Mpc)^{-1}$', fontsize=16)
axs[1, 0].plot(zfsarr, karr * cons.mpc, 'k', label=r'$(\ell_{MFP}^{\gamma})^{-1}$')
axs[1, 0].plot(zcrossarr, karr * cons.mpc, 'g', label=r'$k_{hor}$')
axs[1, 0].plot(zarrcons, kcons(zarrcons) * cons.mpc, color='purple', label=r'$k_{cons}$')
axs[1, 0].plot(zbalance, karr * cons.mpc, color='orange', label=r'$k_{balance}$')
axs[1, 0].legend(fontsize=14)
axs[1, 0].set_xlim(np.min(ztotalarraymaster), np.max(ztotalarraymaster))
axs[1, 0].set_ylim(np.min(karr * cons.mpc), np.max(karr * cons.mpc))
axs[1, 0].set_yscale('log'); axs[1, 0].set_xscale('log'); axs[1, 0].invert_xaxis()

for cind, kind in enumerate(range(3, 53, 10)):
    diff = TCRmag[kind, bcount20]
    fs = FSRsahamag[kind, bcount20]

    axs[0, 1].semilogx(zarraymaster[kind], diff[thetaind, 3, :], c=red_colors5[cind],
                       label=r'$k \approx 10^{%.0f} \, \mathrm{Mpc}^{-1}$' % np.arange(5, 0, -1)[cind])
    axs[0, 1].semilogx(zfsarraymaster[kind], diff[thetaind, 3, -1] * fs[thetaind, 3, :], c=red_colors5[cind])
    axs[1, 1].semilogx(zarraymaster[kind], diff[thetaind, 0, :], c=red_colors5[cind],
                       label=r'$k \approx 10^{%.0f} \, \mathrm{Mpc}^{-1}$' % np.arange(5, 0, -1)[cind])
    axs[1, 1].semilogx(zfsarraymaster[kind], diff[thetaind, 3, -1] * fs[thetaind, 0, :], c=red_colors5[cind])
    axs[0, 1].plot(zfsarr[kind], diff[thetaind, 3, -1], marker='D', color=red_colors5[cind], markersize=12)
    axs[1, 1].plot(zfsarr[kind], diff[thetaind, 3, -1] * fs[thetaind, 0, 0], marker='D', color=red_colors5[cind], markersize=12)
    if zbalance[kind] != 0:
        axs[1, 1].plot(zbalance[kind],
                       diff[thetaind, 3, -1] * fs[thetaind, 0, np.argwhere(zfsarraymaster[kind] == zbalance[kind])],
                       marker='s', color=red_colors5[cind], markersize=12)
    if zarrcons2[np.argmin(abs(kcons2(zarrcons2) - karr[kind]))] != zarrcons2[0]:
        if zarrcons2[np.argmin(abs(kcons2(zarrcons2) - karr[kind]))] > zfsarr[kind]:
            axs[0, 1].plot(zarrcons2[np.argmin(abs(kcons2(zarrcons2) - karr[kind]))],
                           splev(zarrcons2[np.argmin(abs(kcons2(zarrcons2) - karr[kind]))],
                                splrep(zarraymaster[kind][::-1], diff[thetaind, 3, ::-1])),
                           marker='*', color=red_colors5[cind], markersize=12)
            axs[1, 1].plot(zarrcons2[np.argmin(abs(kcons2(zarrcons2) - karr[kind]))],
                           splev(zarrcons2[np.argmin(abs(kcons2(zarrcons2) - karr[kind]))],
                                splrep(zarraymaster[kind][::-1], diff[thetaind, 0, ::-1])),
                           marker='*', color=red_colors5[cind], markersize=12)
        else:
            axs[0, 1].plot(zarrcons2[np.argmin(abs(kcons2(zarrcons2) - karr[kind]))],
                           splev(zarrcons2[np.argmin(abs(kcons2(zarrcons2) - karr[kind]))],
                                splrep(zfsarraymaster[kind][::-1], diff[thetaind, 3, -1] * fs[thetaind, 3, ::-1])),
                           marker='*', color=red_colors5[cind], markersize=12)
            axs[1, 1].plot(zarrcons2[np.argmin(abs(kcons2(zarrcons2) - karr[kind]))],
                           splev(zarrcons2[np.argmin(abs(kcons2(zarrcons2) - karr[kind]))],
                                splrep(zfsarraymaster[kind][::-1], diff[thetaind, 3, -1] * fs[thetaind, 0, ::-1])),
                           marker='*', color=red_colors5[cind], markersize=12)
    else:
        if zarrcons[np.argmin(abs(kcons(zarrcons) - karr[kind]))] > zfsarr[kind]:
            axs[0, 1].plot(zarrcons[np.argmin(abs(kcons(zarrcons) - karr[kind]))],
                           splev(zarrcons[np.argmin(abs(kcons(zarrcons) - karr[kind]))],
                                splrep(zarraymaster[kind][::-1], diff[thetaind, 3, ::-1])),
                           marker='*', color=red_colors5[cind], markersize=12)
            axs[1, 1].plot(zarrcons[np.argmin(abs(kcons(zarrcons) - karr[kind]))],
                           splev(zarrcons[np.argmin(abs(kcons(zarrcons) - karr[kind]))],
                                splrep(zarraymaster[kind][::-1], diff[thetaind, 0, ::-1])),
                           marker='*', color=red_colors5[cind], markersize=12)
        else:
            axs[0, 1].plot(zarrcons[np.argmin(abs(kcons(zarrcons) - karr[kind]))],
                           splev(zarrcons[np.argmin(abs(kcons(zarrcons) - karr[kind]))],
                                splrep(zfsarraymaster[kind][::-1], diff[thetaind, 3, -1] * fs[thetaind, 3, ::-1])),
                           marker='*', color=red_colors5[cind], markersize=12)
            axs[1, 1].plot(zarrcons[np.argmin(abs(kcons(zarrcons) - karr[kind]))],
                           splev(zarrcons[np.argmin(abs(kcons(zarrcons) - karr[kind]))],
                                splrep(zfsarraymaster[kind][::-1], diff[thetaind, 3, -1] * fs[thetaind, 0, ::-1])),
                           marker='*', color=red_colors5[cind], markersize=12)

axs[0, 1].set_xlim([600, 1e8])
axs[1, 1].set_xlim([600, zfsarr[3] + 100])
axs[1, 1].set_ylim([max(-1, -1.01 * max(abs(tf_vals))), min(1, 1.01 * max(abs(tf_vals)))])
axs[0, 1].set_xlabel('Redshift z', fontsize=16)
axs[1, 1].set_xlabel('Redshift z', fontsize=16)

x1, y1_top = 0, 1
x1, y1_bot = 0, 0
x2_bot, y2_bot = 600, karr[43] * cons.mpc
x2_top, y2_top = 600, karr[3] * cons.mpc

axs[1, 0].hlines(karr[43] * cons.mpc, 600, zfsarr[43], color='black', linestyle='--', linewidth=2.5)
axs[1, 0].hlines(karr[3] * cons.mpc, 600, zfsarr[3], color='black', linestyle='--', linewidth=2.5)

con1 = ConnectionPatch(xyA=(x1, y1_top), coordsA=axs[1, 1].transAxes,
                       xyB=(x2_top, y2_top), coordsB=axs[1, 0].transData,
                       color='black', linestyle='--', linewidth=2.5)
con2 = ConnectionPatch(xyA=(x1, y1_bot), coordsA=axs[1, 1].transAxes,
                       xyB=(x2_bot, y2_bot), coordsB=axs[1, 0].transData,
                       color='black', linestyle='--', linewidth=2.5)

legend_elements = [
    Line2D([0], [0], marker='D', color='w', label=r'$z_{FS}$', markerfacecolor='black', markersize=10),
    Line2D([0], [0], marker='*', color='w', label=r'$z_{cons}$', markerfacecolor='black', markersize=14),
    Line2D([0], [0], marker='s', color='w', label=r'$z_{balance}$', markerfacecolor='black', markersize=10),
]
axs[1, 1].legend(handles=legend_elements, loc='best', fontsize=14)

fig.add_artist(con1)
fig.add_artist(con2)

axs[0, 1].legend(fontsize=14)

plt.savefig(PLOTDIR / "03_transfer_scatter.png", dpi=150)
plt.close(fig)
print("saved 03_transfer_scatter.png")

# ---------------------------------------------------------------------------
# Figure 04: FM/TCRmag amplitude-ratio plot
# ---------------------------------------------------------------------------
def relinitamp(k, eps, Lambda):
    # 3*B0*sin(theta)*sqrt(Delta^2_deltagamma/Delta^2_B) with delta_gamma the radiation
    # temperature perturbation (delta_gamma = -zeta/3, so Delta^2_deltagamma = A_s/9)
    # and Delta^2_B = (|eps|*B0^2/2)*(k/k_Lambda)^eps: net prefactor is 1
    A_s = cons.Ak0
    ns = cons.ns
    pivot = cons.k0
    k_Lamb = 2 * np.pi / Lambda
    return np.sin(np.pi / 2) * np.sqrt((2 * A_s / abs(eps)) * k**(ns - 1 - eps) / (pivot**(ns - 1) * k_Lamb**(-eps)))

relamparr = np.zeros((len(karr), len(epsarr)))
for ecount, eps in enumerate(epsarr):
    relamparr[:, ecount] = relinitamp(karr * cons.mpc, eps, 1e3)

thetaind71 = 8
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
binds_04 = np.arange(0, 61, 10)
blue_shades_04 = plt.cm.BuPu(np.linspace(0.95, 0.35, len(binds_04)))
for bcount, bind in enumerate(binds_04):
    color = blue_shades_04[bcount]
    axs[0].loglog(karr * cons.mpc,
                 B0arr[bind] * abs(TCRFM[:, bcount, thetaind71, 3, -1] / TCRmag[:, bcount, thetaind71, 3, -1]),
                 label=f'$B_0 = {pg(bind)}$ pG', color=color)
    axs[0].hlines(3 * B0arr[bind] * np.sin(thetaarr[thetaind71]), karr[0] * cons.mpc, karr[-1] * cons.mpc,
                 color=color, linestyle='--', linewidth=3)
axs[0].set_xlabel(r'$k\; (Mpc^{-1})$', fontsize=20)
axs[0].set_ylabel(r'$\left|\frac{T_{\tilde{b}_y,\delta_\gamma}(k,z_{FS})}{T_{\tilde{b}_y,\tilde{b}_y}(k,z_{FS})}\right|$', fontsize=20, rotation=90)
axs[0].legend(loc='upper right', fontsize=14)
axs[0].tick_params(axis='both', which='major', labelsize=16)

cmap = axs[1].imshow(relamparr[:-30, :].T, aspect='auto', origin='lower',
                     extent=[karr[0] * cons.mpc, karr[-30] * cons.mpc, -epsarr[0], -epsarr[-1]],
                     cmap='twilight', norm=LogNorm())
cbar = plt.colorbar(cmap, ax=axs[1])
cbar.set_label(r'$\left|\frac{T_{\tilde{b}_y,\delta_\gamma}(k,z_{FS})}{T_{\tilde{b}_y,\tilde{b}_y}(k,z_{FS})}\right| \sqrt{\frac{\Delta^2_{\delta_\gamma}(k)}{\Delta^2_{\tilde{B}}(k)}}$', rotation=270, labelpad=65, fontsize=20)
cbar.ax.tick_params(labelsize=16)
axs[1].tick_params(axis='both', which='major', labelsize=16)
axs[1].set_xlabel(r'$k (Mpc^{-1})$', fontsize=20)
axs[1].set_ylabel(r'$\vert\epsilon \vert$', fontsize=20)
axs[1].set_xscale('log')

plt.tight_layout()
plt.savefig(PLOTDIR / "04_fm_amplitude_ratio.png", dpi=150)
plt.close(fig)
print("saved 04_fm_amplitude_ratio.png")

print("all transfer/damping plots done;", time.ctime())
