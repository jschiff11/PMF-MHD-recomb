"""
LMHD coupling-kernel and baryon-heating figures:
  - 12_lmhd_coupling_kernel.png : Lorentz-force variance integrand over
    (q/k, theta) for several spectral indices (self-contained, no data needed)
  - 13_pctdiff_clumping_Tb.png : percent difference in Delta^2_{delta_b}
    with/without baryon heating
  - 14_TbTb_power_spectrum.png : baryon-temperature perturbation power spectrum
  - 15_pctdiff_ionization_Tb.png : percent difference in Delta^2_{delta x_e}
    with/without baryon heating
  - 16_dTb2_amplitude.png : lowest-order shift to the background baryon
    temperature <delta_Tb^(2)>
  - 17_pctdiff_xe2_Tb.png : percent difference in Delta x_e with/without
    baryon heating
  - 20_photon_bath_ydistortion.png : cumulative Compton-y parameter from
    photon-heating channels otherwise neglected in the ideal-LMHD limit:
    the drag-dissipated kinetic energy (2*alpha*rho_K), plus the full
    second-order perturbed Thomson heating rate <dq^(2)/dt|_T> (baryon-
    heating appendix), compared to the FIRAS |y| bound

Inputs, read from the pipeline outputs under $PMHD_OUTDIR:
  ang_avg/TLA/B_{B}pG/ang_avg_k{kind}.pkl       (angle_avging_TLA.py)
  ang_avg/Tb/B_{B}pG/ang_avg_k{kind}_Tb.pkl     (angle_avging_TLA_Tb.py)
  xe2/xe2_B_{B}pG_e{eps}.npy                    (xe2.py)
  xe2/xe2_Tb_B_{B}pG_e{eps}.npy                 (xe2_Tb.py)
  xe2/dTb2_B_{B}pG_e{eps}.npy                   (xe2_Tb.py)
  cross_corr/cross_corr_Tb_B_{B}pG_e{eps}.pkl   (cross_corr_and_source_funcs_Tb.py)
The Tb-variant data is needed for the (bind, epsind) pairs
{0,10,20,30,40,50,60}@epsind=9 and bind=10@epsind={0,4,8,12,16,20}.

Configuration (environment variables):
  PMHD_OUTDIR  : pipeline output directory to read data from
                 (default: <repo>/src/pmhd/data/outputs, the same default the
                 src/pmhd/physics scripts write to)
  PMHD_PLOTDIR : directory to write figures to (default: <repo>/analysis/plots)

Run:  python analysis/plot_baryon_heating.py
"""
import os, sys, pickle, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import LogLocator
from matplotlib.lines import Line2D
from scipy.interpolate import splrep, splev, splint

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pmhd import cons, pars
from pmhd.data.grids import (
    k_grid, eps_grid, z_grid, theta_grid, load_or_generate_B0arr, load_or_generate_z_arrays,
)

DATA = Path(os.environ.get("PMHD_OUTDIR", str(REPO_ROOT / "src/pmhd/data/outputs")))
PLOTDIR = Path(os.environ.get("PMHD_PLOTDIR", str(REPO_ROOT / "analysis/plots")))
PLOTDIR.mkdir(parents=True, exist_ok=True)

epsarr = eps_grid()
B0arr = load_or_generate_B0arr()
karr = k_grid()
zarr = z_grid()


def pg(bind):
    return round(1e12 * B0arr[bind])


def add_log_minor_ticks(ax):
    """Force minor tick marks onto any log-scaled axis. matplotlib's default
    LogLocator sometimes auto-suppresses minor ticks when an axis spans many
    decades (seen here on the y-axis of the loglog figures), even though the
    semilogy figures show them fine -- so set them explicitly for consistency."""
    if ax.get_xscale() == "log":
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    if ax.get_yscale() == "log":
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))


def epstag(epsind):
    return round(epsarr[epsind], 3)


# ---------------------------------------------------------------------------
# Figure 12: LMHD coupling-kernel integrand -- self-contained, no data dependency
# ---------------------------------------------------------------------------
def f(krat, theta, nb):
    # converting the d^3q variance integral of the Lorentz force to dln q
    # gives an integrand prefactor (q/k)^(nb+4)
    return abs(krat**(nb + 4) * np.sin(theta) * (
        (krat * np.sin(theta)**2) / (1 + krat**2 - 2 * krat * np.cos(theta)) + 2 * np.cos(theta)
    ) * (1 + krat**2 - 2 * krat * np.cos(theta))**(nb / 2))


krat_values = np.linspace(1e-2, 100, 1500)
krat_values = np.sort(np.append(krat_values, 1 - 1e-20))
theta_values = np.linspace(0 + 1e-16, np.pi, 2000)
n_values = [-1, -2, -2.5, -3, -3.5, -4]
K, X = np.meshgrid(krat_values, theta_values)

fig, axs = plt.subplots(3, 2, figsize=(20, 12), sharex=True, sharey=True)
axs = axs.flatten()
for i, n in enumerate(n_values):
    Z = f(K, X, n)
    im = axs[i].imshow(Z, extent=[krat_values.min(), krat_values.max(), theta_values.min(), theta_values.max()],
                       origin='lower', cmap='twilight', aspect='auto', norm=mcolors.LogNorm(vmin=1e-5))
    axs[i].set_title(f'$n_B$ = {n}', fontsize=24)
    axs[i].set_xscale('log')
    cbar = fig.colorbar(im, ax=axs[i])
    cbar.ax.tick_params(labelsize=14)
    axs[i].xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))

axs[4].set_xlabel(r'$q/k$', fontsize=20)
axs[5].set_xlabel(r'$q/k$', fontsize=20)
axs[0].set_ylabel(r'$\theta$', fontsize=20)
axs[2].set_ylabel(r'$\theta$', fontsize=20)
axs[4].set_ylabel(r'$\theta$', fontsize=20)

for i in range(4):
    axs[i].tick_params(size=12, labelsize=14, labelbottom=False)
axs[4].tick_params(size=14, labelsize=14, labelbottom=True)
axs[5].tick_params(size=14, labelsize=14, labelbottom=True)

for j in range(len(n_values), len(axs)):
    axs[j].axis('off')

fig.tight_layout(pad=2.0)
plt.subplots_adjust(bottom=0.09, top=0.95)
plt.savefig(PLOTDIR / "12_lmhd_coupling_kernel.png", dpi=150)
plt.close(fig)
print("saved 12_lmhd_coupling_kernel.png")

# ---------------------------------------------------------------------------
# Load ang_avg TLA/Tb dicts (all 61 binds) for the baryon-heating comparisons
# ---------------------------------------------------------------------------
print("loading ang_avg TLA/Tb (all binds) ...", time.ctime())
deltamdeltambar = np.zeros((len(karr), len(zarr), len(B0arr)))
deltamdeltambar_Tb = np.zeros((len(karr), len(zarr), len(B0arr)))
xexebar = np.zeros((len(karr), len(zarr), len(B0arr)))
xexebar_Tb = np.zeros((len(karr), len(zarr), len(B0arr)))
TbTbbar = np.zeros((len(karr), len(zarr), len(B0arr)))

for bind in range(0, len(B0arr), 10):
    tla_dir = DATA / f"ang_avg/TLA/B_{pg(bind)}pG"
    tb_dir = DATA / f"ang_avg/Tb/B_{pg(bind)}pG"
    for kind in range(len(karr)):
        with open(tla_dir / f"ang_avg_k{kind}.pkl", "rb") as f_:
            d = pickle.load(f_)
        deltamdeltambar[kind, :, bind] = d["deltamdeltambar"]
        xexebar[kind, :, bind] = d["xexebar"]
        with open(tb_dir / f"ang_avg_k{kind}_Tb.pkl", "rb") as f_:
            dtb = pickle.load(f_)
        deltamdeltambar_Tb[kind, :, bind] = dtb["deltamdeltambar"]
        xexebar_Tb[kind, :, bind] = dtb["xexebar"]
        TbTbbar[kind, :, bind] = dtb["TbTbbar"]
print("done loading;", time.ctime())

bindarr2 = [0, 30, 60]
lamarrfine = 10**np.arange(20, np.log10(2 * np.pi / karr[44]), .01)
karrfine = 2 * np.pi / lamarrfine
Lambda = 1e3 * cons.mpc
numberofplotspec = 8
colors2 = [plt.cm.magma(i) for i in np.linspace(0.2, 1, numberofplotspec)]
epsind = 9

# ---------------------------------------------------------------------------
# Figure 13: percent diff in clumping Delta^2_delta_b with/without Tb
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs = axs.ravel()
for bcount, bind in enumerate(bindarr2):
    ax = axs[bcount]
    ax.tick_params(size=12, labelsize=12, labelbottom=True)
    zindarr = np.arange(np.argwhere(zarr == 1201).item(), np.argwhere(zarr == 601).item() + 1, 100)
    for zi in range(len(zindarr)):
        zind = zindarr[zi]
        ax.loglog(karrfine * cons.mpc,
                 100 * abs(splev(karrfine * cons.mpc, splrep(karr[::-1] * cons.mpc,
                            ((deltamdeltambar - deltamdeltambar_Tb) / deltamdeltambar)[::-1, zind, bind]))),
                 label=f'$z={zarr[zind]-1:.0f}$', color=colors2[zi])
    ax.set_xlabel(r'$k \; (Mpc^{-1})$', fontsize=16)
    ax.set_title(f'$B_0 = $ {pg(bind)}pG', fontsize=20)
    if bcount % 3 == 0:
        ax.set_ylabel('Percent difference in $\\Delta^{2}_{\\delta_b}(k)$\nwith and without baryon heating', fontsize=13)
    if bcount == 0:
        ax.legend(fontsize=10)
    add_log_minor_ticks(ax)
plt.tight_layout()
plt.savefig(PLOTDIR / "13_pctdiff_clumping_Tb.png", dpi=150)
plt.close(fig)
print("saved 13_pctdiff_clumping_Tb.png")

# ---------------------------------------------------------------------------
# Figure 14: baryon-temperature perturbation (TbTbbar) power spectrum
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs = axs.ravel()
for bcount, bind in enumerate(bindarr2):
    ax = axs[bcount]
    ax.tick_params(size=12, labelsize=12, labelbottom=True)
    zindarr = np.arange(np.argwhere(zarr == 1101).item(), np.argwhere(zarr == 601).item() + 1, 100)
    for zi in range(len(zindarr)):
        zind = zindarr[zi]
        ax.loglog(karrfine * cons.mpc,
                 -epsarr[epsind] * (Lambda / 2 * np.pi)**epsarr[epsind] * (karrfine * cons.mpc)**epsarr[epsind] *
                 abs(splev(karrfine * cons.mpc, splrep(karr[::-1] * cons.mpc, TbTbbar[::-1, zind, bind]))),
                 label=f'$z={zarr[zind]-1:.0f}$', color=colors2[zi])
    ax.set_xlabel(r'$k \; (Mpc^{-1})$', fontsize=16)
    ax.set_title(f'$B_0 = $ {pg(bind)}pG', fontsize=20)
    if bcount % 3 == 0:
        ax.set_ylabel(r'$\Delta^2_{T_b}(k)$', fontsize=16)
    if bcount == 0:
        ax.legend(fontsize=10)
    add_log_minor_ticks(ax)
plt.tight_layout()
plt.savefig(PLOTDIR / "14_TbTb_power_spectrum.png", dpi=150)
plt.close(fig)
print("saved 14_TbTb_power_spectrum.png")

# ---------------------------------------------------------------------------
# Figure 15: percent diff in ionization Delta^2_delta_xe with/without Tb
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs = axs.ravel()
for bcount, bind in enumerate(bindarr2):
    ax = axs[bcount]
    ax.tick_params(size=12, labelsize=12, labelbottom=True)
    zindarr = np.arange(np.argwhere(zarr == 1201).item(), np.argwhere(zarr == 601).item() + 1, 100)
    for zi in range(len(zindarr)):
        zind = zindarr[zi]
        ax.loglog(karrfine * cons.mpc,
                 100 * abs(splev(karrfine * cons.mpc, splrep(karr[::-1] * cons.mpc,
                            ((xexebar - xexebar_Tb) / xexebar)[::-1, zind, bind]))),
                 label=f'$z={zarr[zind]-1:.0f}$', color=colors2[zi])
    ax.set_xlabel(r'$k \; (Mpc^{-1})$', fontsize=16)
    ax.set_title(f'$B_0 = $ {pg(bind)}pG', fontsize=20)
    if bcount % 3 == 0:
        ax.set_ylabel('Percent difference in $\\Delta^{2}_{\\delta x_e}(k)$\nwith and without baryon heating', fontsize=13)
    if bcount == 0:
        ax.legend(fontsize=10)
    add_log_minor_ticks(ax)
plt.tight_layout()
plt.savefig(PLOTDIR / "15_pctdiff_ionization_Tb.png", dpi=150)
plt.close(fig)
print("saved 15_pctdiff_ionization_Tb.png")

# ---------------------------------------------------------------------------
# Figures 16-17: dTb2 amplitude + xe2_Tb-vs-xe2 percent difference
# 13 (bind,epsind) pairs: {0,10,20,30,40,50,60}@9 + bind=10@{0,4,8,12,16,20}
# ---------------------------------------------------------------------------
print("loading xe2_Tb/dTb2 for the requested pairs ...", time.ctime())
xe2_Tb = np.zeros((len(B0arr), len(epsarr), len(zarr)))
dTb2 = np.zeros((len(B0arr), len(epsarr), len(zarr)))
xe2fullk_ref = np.zeros((len(zarr), len(B0arr), len(epsarr)))

pairs = sorted(set([(b, 9) for b in (0, 10, 20, 30, 40, 50, 60)] + [(10, e) for e in (0, 4, 8, 12, 16, 20)]))
for bind, epsind in pairs:
    xe2_Tb[bind, epsind] = np.load(DATA / f"xe2/xe2_Tb_B_{pg(bind)}pG_e{epstag(epsind)}.npy")
    dTb2[bind, epsind] = np.load(DATA / f"xe2/dTb2_B_{pg(bind)}pG_e{epstag(epsind)}.npy")
    xe2fullk_ref[:, bind, epsind] = np.load(DATA / f"xe2/xe2_B_{pg(bind)}pG_e{epstag(epsind)}.npy")
print("done loading;", time.ctime())

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
epsind = 9
binds_16 = (0, 10, 20, 30, 40)
blue_shades_16 = plt.cm.BuPu(np.linspace(0.95, 0.35, len(binds_16)))
for i, bind in enumerate(binds_16):
    axs[0].semilogy(zarr, abs(dTb2[bind, epsind]), label=f'$B_0$={pg(bind)} pG', color=blue_shades_16[i])
axs[0].set_title(f'$\\epsilon=${round(epsarr[epsind], 2)}')
bind = 10
epsinds_16 = (0, 4, 8, 12, 16)
blue_shades_16b = plt.cm.BuPu(np.linspace(0.95, 0.35, len(epsinds_16)))
for i, epsind in enumerate(epsinds_16):
    axs[1].semilogy(zarr, abs(dTb2[bind, epsind]), label=f'$\\epsilon$={round(epsarr[epsind], 2)}', color=blue_shades_16b[i])
legend1 = axs[0].legend(loc='lower left', frameon=True)
axs[1].legend()
axs[1].set_title(f'$B_0=${pg(bind)} pG')
axs[0].set_xlabel('Redshift z', fontsize=14); axs[1].set_xlabel('Redshift z', fontsize=14)
axs[0].set_ylabel(r'$|\langle \delta_{T_b}^{(2)} \rangle|$', fontsize=14)
axs[0].add_artist(legend1)
for ax in axs:
    add_log_minor_ticks(ax)
ymax = max(
    np.max(np.abs(dTb2[bind, 9])) for bind in (0, 10, 20, 30, 40)
)
ymax = max(ymax, max(np.max(np.abs(dTb2[10, epsind])) for epsind in (0, 4, 8, 12, 16)))
axs[0].set_ylim([1e-9, 10**np.ceil(np.log10(ymax))])
plt.tight_layout()
plt.savefig(PLOTDIR / "16_dTb2_amplitude.png", dpi=150)
plt.close(fig)
print("saved 16_dTb2_amplitude.png")

from scipy.signal import savgol_filter


def pctdiff_xe2_smoothed(bind, epsind):
    """Percent difference in Delta x_e with/without baryon heating, smoothed
    with a Savitzky-Golay filter. The raw curve is dominated by ODE-solver
    precision noise at early (high-z) times: xe2fullk_ref itself varies
    smoothly, but right after baryon-heating perturbations turn on (z=1900)
    the true difference xe2_Tb-xe2fullk_ref is right at the solver's
    numerical floor (checked directly -- adjacent-z points jump between
    ~1e-9% and ~1e-3% with no smooth trend), before the real signal has
    grown large enough to dominate. Smoothing removes that floor-level
    noise; it does not suppress genuine physical structure, which is already
    smooth in the raw data at lower z once the signal exceeds the floor."""
    pctdiff = 1e2 * np.abs((xe2_Tb[bind, epsind] - xe2fullk_ref[:, bind, epsind]) / np.abs(xe2fullk_ref[:, bind, epsind]))
    log_pctdiff = np.log(np.maximum(pctdiff, 1e-300))
    smoothed = savgol_filter(log_pctdiff, window_length=151, polyorder=3)
    return np.exp(smoothed)


fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
epsind = 9
binds_17 = (0, 10, 20, 30, 40)
blue_shades_17 = plt.cm.BuPu(np.linspace(0.95, 0.35, len(binds_17)))
for i, bind in enumerate(binds_17):
    axs[0].semilogy(zarr, pctdiff_xe2_smoothed(bind, epsind), label=f'$B_0$={pg(bind)} pG', color=blue_shades_17[i])
axs[0].set_title(f'$\\epsilon=${round(epsarr[epsind], 2)}')
bind = 10
epsinds_17 = (0, 4, 8, 12, 16)
blue_shades_17b = plt.cm.BuPu(np.linspace(0.95, 0.35, len(epsinds_17)))
for i, epsind in enumerate(epsinds_17):
    axs[1].semilogy(zarr, pctdiff_xe2_smoothed(bind, epsind), label=f'$\\epsilon$={round(epsarr[epsind], 2)}', color=blue_shades_17b[i])
axs[0].legend(fontsize=10); axs[1].legend(fontsize=10)
axs[1].set_title(f'$B_0=${pg(bind)} pG')
axs[0].set_xlabel('Redshift z', fontsize=14); axs[1].set_xlabel('Redshift z', fontsize=14)
axs[0].set_ylabel("Percent difference in $\\Delta x_e$", fontsize=12)
axs[0].set_ylim(bottom=1e-5)
for ax in axs:
    add_log_minor_ticks(ax)
plt.tight_layout()
plt.savefig(PLOTDIR / "17_pctdiff_xe2_Tb.png", dpi=150)
plt.close(fig)
print("saved 17_pctdiff_xe2_Tb.png")

# ---------------------------------------------------------------------------
# Figure 20: cumulative Compton-y parameter if all drag-dissipated kinetic
# energy (2*alpha*rho_K, the same quantity underlying the now-removed
# Gamma_diss baryon-heating term) were instead deposited into the photon
# bath. y = Delta_rho_gamma^comoving / (4 rho_gamma^comoving) is the correct
# leading-order conversion throughout the Compton-y era (z << z_mu-y ~ 5e4;
# no mu/thermalization branching-ratio machinery needed). Compared to the
# FIRAS bound |y| <~ 1.5e-5.
#
# The curve spans the full y-era window used by Wagstaff & Banerjee (2015)
# (z=5e4 down to z=1090), combining two pieces:
#   z=1900->600  : FSR-TLA pipeline data (unchanged from the original figure)
#   z=5e4->1900  : FSR-saha data (Saha ionization -- valid since xe~1 here
#                  regardless of Saha vs 3LA), integrated from each k-mode's
#                  own true zfs rather than the FSR-TLA pipeline's artificial
#                  z=1900 "attractor restart" cutoff (no real FSR-TLA data
#                  exists above z=1900 for any mode).
# This second piece deliberately EXCLUDES the tight-coupling (TCR) contribution
# for any k-mode still tightly coupled above z=1900: in TCR, baryons and
# photons move as a single fluid, so the two-fluid Thomson-drag-transfer
# picture used here (rate x KE, draining one species' kinetic energy into the
# other) does not apply -- there is no second reservoir for the energy to
# drain into. The physically correct TCR channel (Silk-diffusion "acoustic
# reheating"/blackbody mixing, Chluba, Khatri & Sunyaev 2012) is a distinct
# calculation not implemented here. Since the TCR damping rate is independently
# verified to be << alpha (the FSR rate) and << H throughout tight coupling,
# omitting it makes this an intentional lower bound on the true extended y,
# not an approximation error.
#
# *** IMPORTANT for anyone modifying or reproducing this calculation ***
# The TCR exclusion above is NOT just a z>1900 concern -- it must be enforced
# via a PER-MODE, PER-REDSHIFT check (zfs(k) >= z) at EVERY z in the curve,
# including inside the "published" z<=1900 window. This is because each
# k-mode has its own free-streaming redshift zfs(k) (loaded via
# load_or_generate_z_arrays()), and for long-wavelength (small-k) modes
# zfs(k) can be WELL BELOW 1900 -- e.g. the smallest k in the grid typically
# has zfs ~ 900. Such a mode is therefore *still tightly coupled* for a good
# stretch of the nominal z<1900 "FSR-TLA" range, even though the pipeline
# happily returns ang_avg_Tb data for it there (angle_avging_TLA_Tb.py fills
# that stretch in directly from the TCR solution -- see its own
# `if zstartarr[input_kind] != 1900` branch). Because the k-integral here is
# weighted as k^(eps-3), which favors small k steeply, this contamination is
# NOT a small correction: in an earlier version of this calculation that used
# the pipeline's precomputed cross_corr_Tb pkl (vxrms/vyrms/vzrms, summed over
# the FULL k range with no zfs(k)-vs-z check), the smallest-k mode dominated
# the entire k-integral by 3-4 orders of magnitude while still tightly
# coupled, inflating the marginal-case (B0=5nG) y by a factor of ~500
# (from ~3e-8 to ~1.6e-5, i.e. from safely below to "marginally exceeding"
# the FIRAS bound). The fix (implemented in y_parameter() and
# _y_extension_from_raw() below) is to restrict the k-integral at EVERY z to
# kind=0..kmax_idx where kmax_idx is the largest index with zfsarr[kmax_idx]
# >= z (karr and zfsarr are both sorted with zfsarr descending in kind, so
# this is a simple prefix). Do NOT shortcut this by reading cc['vxrms'] etc.
# directly from a cross_corr_Tb pkl for this figure -- that data has no such
# restriction applied and will silently reintroduce the bug.
# ---------------------------------------------------------------------------
print("computing photon-bath heating (Compton-y) check ...", time.ctime())
from scipy.integrate import cumulative_trapezoid, odeint
from astropy.cosmology import Planck18 as cosmo

rhob0 = cosmo.Ob0 * cosmo.critical_density0.value
rho_gamma_comoving0 = cons.arad * cons.T0 ** 4
Lambda = 1e3 * cons.mpc
alpha_z = pars.f_lambda(zarr) * (1 - cons.yhe) * pars.xe_full_He(zarr)
H_z = pars.H(zarr)

# Background quantities for the full second-order perturbed Thomson heating
# rate <dq^(2)/dt|_T> sourcing Delta_rho_dot_gamma (baryon-heating appendix):
#   <dq^(2)/dt|_T> = (3/2) n_b k_B Gamma_T <
#       [delta_b^(1) delta_xe^(1) + delta_xe^(2)]/xe (T_gamma - T_b)
#       - [delta_Tb^(2) + delta_b^(1) delta_Tb^(1)
#          + delta_xe^(1) delta_Tb^(1)/xe] T_b >.
# Besides the already-tracked drag channel 2*alpha*rho_K, this is an
# additional, physically distinct source of photon heating (the recombination-
# rate-driven terms instead source cosmological recombination radiation, not
# a smooth y-distortion, and are not included here). Verified numerically
# against the existing 2*alpha*rho_K term to be a <=few% correction to y at
# z=1090 for all field strengths considered (negligible against the FIRAS
# bound); included here for completeness. See y_parameter() below for how
# each bracket term maps onto the pipeline's existing saved arrays.
_fhe_bh = cons.yhe / (4.0 * (1.0 - cons.yhe))
_zhold_Tb = np.arange(1900, 600, -0.01)
_solTb = odeint(pars.RHSTbhom, pars.Tcmb(1900), _zhold_Tb, args=(pars.xe_full_He,))
_Tbhomspl = splrep(_zhold_Tb[::-1], _solTb.flatten()[::-1])


def Tbhom(z):
    return splev(z, _Tbhomspl)


def Gammac(z):
    return 8 * cons.arad * pars.Tcmb(z) ** 4 * pars.xe_full_He(z) * cons.sigmat / (
        3 * cons.me * (1 + pars.xe_full_He(z) + _fhe_bh)) / cons.c


_Tgamma_z = pars.Tcmb(zarr)
_Tb_z = Tbhom(zarr)
_Gammac_z = Gammac(zarr)
_xeHe_z = pars.xe_full_He(zarr)
_nb_z = pars.nh(zarr) * (1 + _fhe_bh + _xeHe_z)
_rhob_z = rhob0 * (1 + zarr) ** 3
_cs2_z = cons.kb * _Tb_z * _nb_z / _rhob_z


def _load_all_ang_avg_Tb(bind):
    """Per-k PhixPhixbar/PhiyPhiybar/ThetaThetabar/Tbdeltambar/Tbxebar for
    all k, shape (nk, len(zarr))."""
    nk = len(karr)
    Pxx = np.zeros((nk, len(zarr)))
    Pyy = np.zeros((nk, len(zarr)))
    Tt = np.zeros((nk, len(zarr)))
    Tbdeltambar = np.zeros((nk, len(zarr)))
    Tbxebar = np.zeros((nk, len(zarr)))
    for kind in range(nk):
        with open(DATA / f"ang_avg/Tb/B_{pg(bind)}pG/ang_avg_k{kind}_Tb.pkl", "rb") as f:
            d = pickle.load(f)
        Pxx[kind] = d["PhixPhixbar"]
        Pyy[kind] = d["PhiyPhiybar"]
        Tt[kind] = d["ThetaThetabar"]
        Tbdeltambar[kind] = d["Tbdeltambar"]
        Tbxebar[kind] = d["Tbxebar"]
    return Pxx, Pyy, Tt, Tbdeltambar, Tbxebar


def _kint_upto(kweight, vals, kmax_idx):
    # Integrate only over k-modes already free-streaming at this z (kind=0..kmax_idx,
    # since karr/zfsarr are both sorted with zfsarr descending -- see module docstring
    # note on the TCR k-truncation below). Fewer than 2 points -> no FSR modes yet.
    if kmax_idx < 1:
        return 0.0
    k_sub = karr[:kmax_idx + 1]
    integrand_sub = (kweight * vals)[:kmax_idx + 1]
    return splint(k_sub[-1], k_sub[0], splrep(np.flip(k_sub), np.flip(integrand_sub)))


def y_parameter(bind, epsind, ang_avg_cache):
    """Published z=1900->600 piece, FSR-TLA, with the k-integral truncated at
    each z to exclude modes still tightly coupled there (zfs(k) < z). Without
    this, the k^(eps-3) weighting lets the smallest-k (longest-wavelength,
    latest-zfs) mode -- still in TCR at z=1900 for many spectra -- dominate the
    entire k-integral by orders of magnitude, via the same physically invalid
    two-fluid-drag-on-a-single-fluid application flagged for the z>1900
    extension. Confirmed this dominance directly: excluding it drops the
    marginal-case (B0=5nG) published y by a factor of ~500.

    In addition to the drag-dissipation channel (2*alpha*rho_K), this
    includes the full second-order perturbed Thomson heating rate
    <dq^(2)/dt|_T> (baryon-heating appendix):
        <dq^(2)/dt|_T> = (3/2) n_b k_B Gamma_T <
            [delta_b^(1) delta_xe^(1) + delta_xe^(2)]/xe (T_gamma - T_b)
            - [delta_Tb^(2) + delta_b^(1) delta_Tb^(1)
               + delta_xe^(1) delta_Tb^(1)/xe] T_b >,
    which sources the photon energy density with a MINUS sign (energy
    flowing into the baryons is energy lost by the photons). The bracket
    [delta_b^(1) delta_xe^(1) + delta_xe^(2)] equals the pipeline's own
    saved xe2_Tb array exactly: xe2_Tb.py builds Delta_xe as
    norm*(-x1s^(2) + deltamxecross), where deltamxecross IS the k-integrated
    <delta_b delta_xe> cross term, so -x1s^(2) is delta_xe^(2) in the
    appendix's (uncorrected) internal convention and the saved array already
    equals the full bracket sum -- no separate <delta_b delta_xe> term is
    added here. delta_Tb^(2) is the pipeline's saved dTb2 array directly (no
    such correction needed there: the Tb ODE tracks the physical Tb
    perturbation, not a level-population moment). The two remaining cross
    terms, <delta_b delta_Tb> and <delta_xe delta_Tb>, use the same per-z
    k-truncation and k^(eps-1) spectral weighting as every other <.. delta_b>-
    type cross-correlation computed elsewhere in this pipeline."""
    if bind not in ang_avg_cache:
        ang_avg_cache[bind] = _load_all_ang_avg_Tb(bind)
    Pxx, Pyy, Tt, Tbdeltambar, Tbxebar = ang_avg_cache[bind]
    kweight_v = karr ** (epsarr[epsind] - 3)
    kweight_c = karr ** (epsarr[epsind] - 1)
    nz = len(zarr)
    vxrms = np.zeros(nz)
    vyrms = np.zeros(nz)
    vzrms = np.zeros(nz)
    Tbdelta_raw = np.zeros(nz)
    Tbxe_raw = np.zeros(nz)
    for zind in range(nz):
        # THE critical line -- see the big comment above Figure 20's header for
        # why: this must be evaluated fresh at every z, not just once for the
        # whole z<1900 window, since zfs(k) varies by mode and can be <1900.
        kmax_idx = np.sum(zfsarr >= zarr[zind]) - 1
        vxrms[zind] = _kint_upto(kweight_v, Pxx[:, zind], kmax_idx)
        vyrms[zind] = _kint_upto(kweight_v, Pyy[:, zind], kmax_idx)
        vzrms[zind] = _kint_upto(kweight_v, Tt[:, zind], kmax_idx)
        Tbdelta_raw[zind] = _kint_upto(kweight_c, Tbdeltambar[:, zind], kmax_idx)
        Tbxe_raw[zind] = _kint_upto(kweight_c, Tbxebar[:, zind], kmax_idx)
    v2 = vxrms + vyrms + vzrms
    rho_K_raw = 0.5 * rhob0 * (1 + zarr) ** 3 * v2
    norm = (abs(epsarr[epsind]) / 4) * (Lambda / (2 * np.pi)) ** epsarr[epsind]
    Tbdelta = norm * Tbdelta_raw
    Tbxe = norm * Tbxe_raw
    Gamma_drag = 2 * alpha_z * norm * rho_K_raw
    # (3/2) n_b k_B Gamma_T = rho_b cs2 Gamma_T / Tb, so the explicit Tb
    # factors on each bracket term cancel/combine as shown.
    term_ionization = 1.5 * _rhob_z * _cs2_z * _Gammac_z * (
        xe2_Tb[bind, epsind] / _xeHe_z) * (_Tgamma_z / _Tb_z - 1)
    term_temperature = 1.5 * _rhob_z * _cs2_z * _Gammac_z * (
        dTb2[bind, epsind] + Tbdelta + Tbxe / _xeHe_z)
    dq2dt_T = term_ionization - term_temperature
    Gamma = Gamma_drag - dq2dt_T
    integrand = Gamma / ((1 + zarr) ** 5 * H_z)
    delta_rho_gamma_comoving = -cumulative_trapezoid(integrand, zarr, initial=0.0)
    return delta_rho_gamma_comoving / (4 * rho_gamma_comoving0)


# --- z=5e4->1900 FSR-saha extension (conservative, TCR excluded) -----------
thetaarr = theta_grid()
zcrossarr, zfsarr = load_or_generate_z_arrays()
ZTARGET = np.logspace(np.log10(5e4), np.log10(1900), 300)  # descending, 5e4 -> 1900


def _angavg(arr_theta_z):
    """arr_theta_z: shape (ntheta, nz), theta over [0,pi/2] only. Returns
    int_0^{pi} sin(theta) arr^2 dtheta, shape (nz,), using the fact that Phix/
    Phiy/Theta are all even under theta -> pi-theta (matching the reflection
    convention in angle_avging_TLA_Tb.py), so the full-range [0,pi] integral
    is exactly twice the half-range [0,pi/2] integral computed here."""
    ntheta, nz = arr_theta_z.shape
    out = np.zeros(nz)
    for iz in range(nz):
        sq = arr_theta_z[:, iz] ** 2 * np.sin(thetaarr)
        out[iz] = splint(thetaarr[0], thetaarr[-1], splrep(thetaarr, sq))
    return 2 * out


def _load_and_angavg_FSRsaha(bind, kind):
    # FSRsahaalf/FSRsahamag are integrated with a "unit" IC (bx=1 / by=1, see
    # FSR_saha_Tfs.py), not the true physical amplitude -- must be rescaled by
    # the true bx/by value at z=zfs (the last point of the TCR solution),
    # exactly as angle_avging_TLA_Tb.py does via its "diffalf_Tf_b"/
    # "diffmag_Tf_b" factors, before this can be compared with any physically
    # normalized quantity (verified to reproduce the published ang_avg_Tb
    # PhixPhixbar/PhiyPhiybar/ThetaThetabar at z=1900 to <0.1%).
    alf = np.load(DATA / f"Tfs/B_{pg(bind)}pG/FSRsahaalf_k{kind}.npy")  # (ntheta, 2, 1e4): [Phix, bx]
    mag = np.load(DATA / f"Tfs/B_{pg(bind)}pG/FSRsahamag_k{kind}.npy")  # (ntheta, 4, 1e4): [delta, Theta, Phiy, by]
    tcralf = np.load(DATA / f"Tfs/B_{pg(bind)}pG/TCRalf_k{kind}.npy")
    tcrmag = np.load(DATA / f"Tfs/B_{pg(bind)}pG/TCRmag_k{kind}.npy")
    diffalf = tcralf[:, 1, -1].reshape(-1, 1)   # true physical bx at z=zfs, per theta
    diffmag = tcrmag[:, 3, -1].reshape(-1, 1)   # true physical by at z=zfs, per theta
    zgrid = np.logspace(np.log10(zfsarr[kind]), np.log10(600), num=10**4)
    Hz = pars.H(zgrid)
    Phix_phys = Hz[None, :] * diffalf * alf[:, 0, :]
    Theta_phys = Hz[None, :] * diffmag * mag[:, 1, :]
    Phiy_phys = Hz[None, :] * diffmag * mag[:, 2, :]
    return zgrid, _angavg(Phix_phys), _angavg(Phiy_phys), _angavg(Theta_phys)


def _interp_onto_target(zgrid, arr, ztarget_subset):
    order = np.argsort(zgrid)
    return splev(ztarget_subset, splrep(zgrid[order], arr[order]))


FIG20_CACHE_DIR = REPO_ROOT / "analysis" / ".fig20_raw_ext_cache"


def _load_raw_FSR_extension(bind):
    """bind-dependent-only raw velocity arrays on ZTARGET (independent of epsind).
    This is the expensive step in Figure 20 (per k-mode angle-averaging via
    _angavg, which does a spline fit/integral at each of 10^4 z-grid points,
    repeated for ~45 k-modes) so its result is cached to disk per bind --
    delete FIG20_CACHE_DIR (or the one bind's .npz) if the underlying pipeline
    data changes and this needs to be recomputed.

    NOTE: this z>1900 extension is built from the Saha-only FSRsaha/TCR
    transfer functions, which have no Tb-tracking variant, so it does NOT
    include the Compton-relaxation source terms added to y_parameter() below.
    Given those terms were verified to be a <=few% correction to y in the
    z<1900 published window (where the equivalent data does exist), and the
    extension already dominates the total y budget, omitting them here is a
    minor, uncharacterized (but expected to be similarly small) gap rather
    than a captured approximation."""
    cache_path = FIG20_CACHE_DIR / f"bind{bind}.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        return d["vx_FSR"], d["vy_FSR"], d["vz_FSR"]
    nk = len(karr)
    nz = len(ZTARGET)
    vx_FSR = np.zeros((nk, nz))
    vy_FSR = np.zeros((nk, nz))
    vz_FSR = np.zeros((nk, nz))
    for kind in range(nk):
        zf = zfsarr[kind]
        mask_fsr = (ZTARGET <= zf) & (zf > 1900)
        if mask_fsr.any():
            zgrid, pxx, pyy, tt = _load_and_angavg_FSRsaha(bind, kind)
            zt = np.clip(ZTARGET[mask_fsr], zgrid.min(), zgrid.max())
            vx_FSR[kind, mask_fsr] = _interp_onto_target(zgrid, pxx, zt)
            vy_FSR[kind, mask_fsr] = _interp_onto_target(zgrid, pyy, zt)
            vz_FSR[kind, mask_fsr] = _interp_onto_target(zgrid, tt, zt)
    FIG20_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, vx_FSR=vx_FSR, vy_FSR=vy_FSR, vz_FSR=vz_FSR)
    return vx_FSR, vy_FSR, vz_FSR


def _y_extension_from_raw(vx_FSR, vy_FSR, vz_FSR, epsind):
    """z=5e4->1900 extension y-curve from bind-dependent raw arrays. Uses the
    same per-z k-domain TRUNCATION as y_parameter (_kint_upto), rather than
    zero-padding excluded (still-TCR) k's and integrating over the full range
    -- a cubic spline fit through a hard zero-tail has non-local support and
    was found to systematically bias the integral high by ~20% versus proper
    truncation (checked directly on a synthetic smooth profile)."""
    nz = len(ZTARGET)
    kweight = karr ** (epsarr[epsind] - 3)
    vxrms = np.zeros(nz)
    vyrms = np.zeros(nz)
    vzrms = np.zeros(nz)
    for iz in range(nz):
        # Same per-z, per-mode zfs(k)>=z restriction as y_parameter() below --
        # see the big comment above Figure 20's header for why this can't be
        # computed once and reused across z.
        kmax_idx = np.sum(zfsarr >= ZTARGET[iz]) - 1
        vxrms[iz] = _kint_upto(kweight, vx_FSR[:, iz], kmax_idx)
        vyrms[iz] = _kint_upto(kweight, vy_FSR[:, iz], kmax_idx)
        vzrms[iz] = _kint_upto(kweight, vz_FSR[:, iz], kmax_idx)

    z = ZTARGET
    rhob_z = rhob0 * (1 + z) ** 3
    alpha_x = pars.f_lambda(z) * (1 - cons.yhe) * pars.xesaha_full(z)
    alpha_yz = pars.f_lambda(z) * (1 - cons.yhe) * pars.xesaha_full_He(z)
    Gamma_raw = 2 * alpha_x * 0.5 * rhob_z * vxrms + 2 * alpha_yz * 0.5 * rhob_z * (vyrms + vzrms)
    norm = (abs(epsarr[epsind]) / 4) * (Lambda / (2 * np.pi)) ** epsarr[epsind]
    integrand = norm * Gamma_raw / ((1 + z) ** 5 * pars.H(z))
    cum = -cumulative_trapezoid(integrand, z, initial=0.0)
    return cum / (4 * rho_gamma_comoving0)


_idx_zdec = np.argwhere(zarr == 1090)[0, 0]


def y_parameter_full(bind, epsind, raw_cache, ang_avg_cache):
    """Combined z=5e4->1090 curve: extension (5e4->1900) + published (1900->1090).
    Truncated at z_dec=1090 (rather than continuing to z=600) since the
    y=Delta_rho_gamma/(4 rho_gamma) relation is only calibrated for
    z_dec <~ z <~ z_mu-y -- matching the window used in Wagstaff & Banerjee
    (2015) exactly, rather than showing the formally-out-of-window tail."""
    if bind not in raw_cache:
        raw_cache[bind] = _load_raw_FSR_extension(bind)
    y_ext = _y_extension_from_raw(*raw_cache[bind], epsind)
    y_pub = y_parameter(bind, epsind, ang_avg_cache)
    zarr_trunc = zarr[:_idx_zdec + 1]
    y_pub_trunc = y_pub[:_idx_zdec + 1]
    z_full = np.concatenate([ZTARGET, zarr_trunc[1:]])
    y_full = np.concatenate([y_ext, y_ext[-1] + y_pub_trunc[1:]])
    return z_full, y_full


print("computing z=5e4->1900 FSR-saha extension for Figure 20 ...", time.ctime())
_raw_cache = {}
_ang_avg_cache = {}
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
epsind = 9
binds_20 = (0, 10, 20, 30, 40, 50, 60)
blue_shades_20 = plt.cm.BuPu(np.linspace(0.95, 0.35, len(binds_20)))
for i, bind in enumerate(binds_20):
    z_full, y_full = y_parameter_full(bind, epsind, _raw_cache, _ang_avg_cache)
    axs[0].loglog(z_full, y_full, label=f'$B_0$={pg(bind)} pG', color=blue_shades_20[i])
axs[0].set_title(f'$\\epsilon=${round(epsarr[epsind], 2)}')
bind = 10
epsinds_20 = (0, 8, 16)
blue_shades_20b = plt.cm.BuPu(np.linspace(0.95, 0.35, len(epsinds_20)))
for i, epsind in enumerate(epsinds_20):
    z_full, y_full = y_parameter_full(bind, epsind, _raw_cache, _ang_avg_cache)
    axs[1].loglog(z_full, y_full, label=f'$\\epsilon$={round(epsarr[epsind], 2)}', color=blue_shades_20b[i])
axs[1].set_title(f'$B_0=${pg(bind)} pG')
for ax in axs:
    ax.axhline(1.5e-5, color='k', linestyle='--', linewidth=1)
    ax.axvline(1090, color='gray', linestyle=':', linewidth=1)
    ax.text(0.5, 1.5e-5, 'FIRAS $|y|$ limit', transform=ax.get_yaxis_transform(),
            ha='center', va='bottom', fontsize=9)
    ax.text(1090, 0.5, '$z_{\\rm dec}=1090$', transform=ax.get_xaxis_transform(),
            ha='right', va='center', fontsize=9, rotation=90)
    ax.set_xlabel('Redshift z', fontsize=14)
    ax.set_xlim(850, 3e4)
    ax.set_ylim(bottom=1e-16)
    add_log_minor_ticks(ax)
axs[0].legend(fontsize=8); axs[1].legend(fontsize=8)
axs[0].set_ylabel(r'$|y|$', fontsize=14)
plt.tight_layout()
plt.savefig(PLOTDIR / "20_photon_bath_ydistortion.png", dpi=150)
plt.close(fig)
print("saved 20_photon_bath_ydistortion.png")

print("all baryon-heating plots done;", time.ctime())
