"""
Lyman-alpha moment and line-shape figures:
  - 18_lyman_alpha_moments.png : D_l^+/D_l^- basis solutions of the m = +/-1
    perturbed Boltzmann hierarchies vs frequency detuning
  - 19_hyrec_saha_3la_psd_lineshape.png : Saha/3LA/HyRec ionization histories,
    Fokker-Planck vs Sobolev recombination-rate comparison, and the PSD
    spectral distortion across the Lyman-alpha line

These figures are helium-independent: the photon-moment machinery (hompsd,
inhomo_moments) and pars.taus/feq depend only on the hydrogen ionization
fraction. No pipeline data is required -- the homogeneous phase-space density
is computed on the fly with hompsd.psdHR (as in the src/pmhd/physics
pipeline), and the HyRec comparison table is read from the repo's
pre-stored data.

FIGURE 18 PARAMETERS (for anyone reproducing it)
------------------------------------------------
Several inputs appear below only as array indices; the resolved values are:

  wavenumber        karr[3] = 3.149052e-20 m^-1 = 97169.60 Mpc^-1
                    (k_grid() = 2*pi/logspace(20, 26.9, 69), SI units)
  redshift          zind 800 of z_grid(1900, 600, -1)  ->  z = 1100
                    (the solver spans zind 800->802, i.e. z = 1100-1099; the
                    plotted row is moments[0, ...] = z = 1100)
  multipole cutoff  nm = 30, hierarchy truncated at l = nm - 1 = 29
  frequency grid    x = (nu - nu_Lya)/(nu_Lya * Delta_H) in [-1000, +1000]
                    Doppler widths, steps = 100001, dx = 0.02;
                    plotted over x in [-200, 200]
  boundary cond.    simple truncation (cutoff B.C. on the advection term at
                    j = nm-1). The m = +/-1 hierarchies do NOT use the
                    non-reflecting condition of the m = 0 solver fullz().
  cosmology         astropy Planck18, via pmhd.cons / pars.H(z):
                    H0 = 2.192711e-18 s^-1, Omega_m = 0.309660,
                    Omega_Lambda = 0.688846, N_eff = 3.046, T0 = 2.7255 K,
                    Y_p = 0.2454, Omega_b h^2 = 0.022418
  plotted curves    l=1 is p1moms[0, 1::30, 0], l=2 is p1moms[0, 2::30, 0].
                    The ::30 stride is the nm interleaving of the moment
                    hierarchy; the trailing index 0 selects the FIRST of the
                    three basis solutions returned by fullzp1/fullzm1.

The basis-solution index and the multipole normalization convention are the
two places where an independent implementation is most likely to pick up an
overall amplitude offset while reproducing the curve shape correctly.

Configuration (environment variables):
  PMHD_PLOTDIR : directory to write figures to (default: <repo>/analysis/plots)

Run:  python analysis/plot_moments_and_lineshape.py
"""
import os, sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import quad, odeint
from scipy.interpolate import splrep, splev, splint

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pmhd import cons, pars
from pmhd.data.grids import k_grid, eps_grid, z_grid
from pmhd.physics import hompsd
from pmhd.physics import inhomo_moments as inhomomom

PLOTDIR = Path(os.environ.get("PMHD_PLOTDIR", str(REPO_ROOT / "analysis/plots")))
PLOTDIR.mkdir(parents=True, exist_ok=True)

epsarr = eps_grid()
karr = k_grid()
zarr = z_grid()
xe_full = pars.xe_full
xesaha_full = pars.xesaha_full

# Homogeneous background quantities used by the PSD solver
taus_total = pars.taus(zarr, xe_full)
pab_total = pars.pab(zarr)
feq_total = pars.feq(zarr, xe_full)

# ---------------------------------------------------------------------------
# Figure 18: Lyman-alpha moments D_l^+ / D_l^- vs frequency detuning
# ---------------------------------------------------------------------------
print("computing homogeneous PSD rows for the moment solver ...", time.ctime())
steps = 100001
output_xs, dx = np.linspace(-1000.0, 1000.0, num=steps, retstep=True)

# The moment solvers index the psd list by redshift index and evaluate each
# entry as a spline (see inhomo_moments.fullzp1); only the redshift rows
# actually used below (zind 800-801, i.e. z = 1100-1099) are computed.
psd = [None] * len(zarr)
psdhomo = {}
for zind in (800, 801):
    hold = hompsd.psdHR(output_xs, zarr, zind, taus_total, pab_total, feq_total)
    psd[zind] = splrep(output_xs, hold[2])
    psdhomo[zind] = hold[2]

print("computing Lyman-alpha moments ...", time.ctime())

p1moms = inhomomom.fullzp1(zarr, 800, 802, karr[3], xe_full, 30, steps, psd)
m1moms = inhomomom.fullzm1(zarr, 800, 802, karr[3], xe_full, 30, steps, psd)

fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True, sharex=True)
axs[0].plot(output_xs, p1moms[0, 1::30, 0], 'k-', label=r'$\ell=1$')
axs[0].plot(output_xs, p1moms[0, 2::30, 0], 'k:', label=r'$\ell=2$')
axs[1].plot(output_xs, m1moms[0, 1::30, 0], 'k-', label=r'$\ell=1$')
axs[1].plot(output_xs, m1moms[0, 2::30, 0], 'k:', label=r'$\ell=2$')
axs[0].legend(fontsize=16)
axs[0].set_xlim([-200, 200])
axs[0].set_ylabel(r'$\mathcal{D}_{\ell}^+$', fontsize=16)
axs[0].set_xlabel(r'$x=(\nu - \nu_{\mathrm{Ly}\alpha})/(\nu_{\mathrm{Ly}\alpha} \Delta_{\mathrm{H}})$', fontsize=14)
axs[1].set_xlabel(r'$x=(\nu - \nu_{\mathrm{Ly}\alpha})/(\nu_{\mathrm{Ly}\alpha} \Delta_{\mathrm{H}})$', fontsize=14)
axs[0].tick_params(size=12, labelsize=12, labelbottom=True)
axs[1].tick_params(size=12, labelsize=12, labelbottom=True)
plt.tight_layout()
plt.savefig(PLOTDIR / "18_lyman_alpha_moments.png", dpi=150)
plt.close(fig)
print("saved 18_lyman_alpha_moments.png;", time.ctime())

# ---------------------------------------------------------------------------
# Figure 19: redistribution function, HyRec/Saha/3LA xe(z), PSD lineshape
# ---------------------------------------------------------------------------
print("computing line-averaged PSD over all redshifts ...", time.ctime())
xarrfine = np.linspace(-1000, 1000, 100001)

xibar = np.array([hompsd.psdHR(xarrfine, zarr, zind, taus_total, pab_total, feq_total)[1] for zind in range(len(zarr))])

xibarint = splrep(zarr[::-1], xibar)


def xibarfunc(z):
    return splev(z, xibarint)


def modCredist(z, xibarf):
    return (3 * cons.Alya * (pars.pab(z) * (1 - xibarf(z))) / (1 - (1 - pars.pab(z)) * (1 - xibarf(z))) + cons.L2s1s) / (
        3 * cons.Alya * (pars.pab(z) * (1 - xibarf(z))) / (1 - (1 - pars.pab(z)) * (1 - xibarf(z))) + cons.L2s1s + 4 * pars.betab(z))


def RHSwredist(xe, z):
    return (modCredist(z, xibarfunc) / ((1 + z) * pars.H(z))) * (
        pars.nh(z) * xe**2 * pars.alphab(z) - 4 * (1 - xe) * pars.betab(z) *
        np.exp(-(cons.en2 - cons.en1) / (cons.kb * pars.Tcmb(z))))


print("solving redistribution xe ODE ...", time.ctime())
xeredist = odeint(RHSwredist, xe_full(zarr[0]), zarr)

print("loading HyRec table ...", time.ctime())
data = np.loadtxt(REPO_ROOT / "src/pmhd/data/pre_stored_data/output_xe.dat")
hyrecz = data[:, 0]
hyrecxe = data[:, 1]

xarr = np.arange(-1000, 1000.2, 0.2)
taushold = pars.taus(1100, xe_full)
integhold = np.zeros(len(xarr))
for xind, x in enumerate(xarr):
    integhold[xind] = quad(pars.voigt, 1000, x, args=1100)[0]

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[2, 1])
axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[1, 0]), plt.subplot(gs[:, 1])]

axs[0].semilogy(zarr, xesaha_full(zarr), 'r', label='Saha')
axs[0].semilogy(zarr, xe_full(zarr), 'b', label='3LA')
axs[0].semilogy(hyrecz, hyrecxe, 'g', label='Hyrec')
axs[0].legend(fontsize=14)
axs[0].set_xlim([600, 1800])
axs[0].set_ylim([5e-4, 2])
axs[0].set_ylabel(r'$x_e$', fontsize=16)

axs[1].semilogy(zarr, abs(xeredist[:, 0] - xe_full(zarr)) / xe_full(zarr), 'b')
axs[1].set_xlim([600, 1800])
axs[1].set_ylabel(r'$\frac{|x_e^{FP} - x_e^{3LA}|}{x_e^{3LA}}$', fontsize=14)
axs[1].set_xlabel('Redshift z', fontsize=14)

axs[2].plot(xarr, (8 * np.pi * cons.nuly**3) / (cons.c**3 * pars.nh(1100)) * (
    pars.feq(1100, xe_full) + np.exp(-(cons.en2 - cons.en1) / (cons.kb * pars.Tcmb(1100))) +
    (np.exp(-(cons.en2 - cons.en1) / (cons.kb * pars.Tcmb(1100))) - pars.feq(1100, xe_full)) *
    np.exp(taushold * integhold)
), 'k--', label='Sobolev')
axs[2].plot(output_xs, (8 * np.pi * cons.nuly**3) / (cons.c**3 * pars.nh(1100)) * psdhomo[800], 'k', label='Fokker-Planck')
axs[2].semilogy(xarr, pars.voigt(xarr, 1100), 'b', label=r'$\phi(x)$')
axs[2].set_xlabel(r'$x=(\nu - \nu_{\mathrm{Ly}\alpha})/(\nu_{\mathrm{Ly}\alpha} \Delta_{\mathrm{H}})$', fontsize=14)
axs[2].set_xlim([-400, 400])
axs[2].legend(fontsize=12)

plt.tight_layout()
plt.savefig(PLOTDIR / "19_hyrec_saha_3la_psd_lineshape.png", dpi=150)
plt.close(fig)
print("saved 19_hyrec_saha_3la_psd_lineshape.png;", time.ctime())

print("all moments/lineshape plots done;", time.ctime())
