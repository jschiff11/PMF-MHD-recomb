# pyright: ignore[reportMissingImports]
"""
Second-order ionization shift for a SINGLE (k, theta) mode -- e.g. a sound-wave standing
mode in an MHD simulation box -- built entirely from repo code.

The stochastic power-spectrum pipeline
(``FSR_TLA_Tfs`` -> ``angle_avging_TLA`` -> ``cross_corr_and_source_fncs`` -> ``xe2``)
averages over field/wavevector angle and integrates over k with the weight
``k**(eps-n)``, closing with the amplitude ``(|eps|/4)(Lambda/2pi)**eps``.  A single mode
is that same chain with three restrictions:

  1. no theta average -- evaluate the bilinear products at one theta (no sin(theta) measure);
  2. no k integral    -- evaluate at one k, keeping the RELATIVE physical k-powers
                         (``pmhd.physics.source_assembly.single_mode_reducer``);
  3. no ``(|eps|/4)(Lambda/2pi)**eps`` -- that prefactor normalises a CONTINUOUS spectrum
     (it is tied to Delta^2_B and the finite ``karr`` range, and is NOT unit-normalised);
     for a delta-function spectrum it is simply the wrong object.

Normalisation (this is the convention, and it is verified)
----------------------------------------------------------
For a single plane wave  dB_i(x)/B0 = (dB/B0) cos(k0.x), i.e. a spectrum
``P_B(k,theta) = A delta(k-k0) delta(theta-theta0)``, the zero-lag correlator works out to

    b(t) = <delta delta> = (1/2) (dB/B0)**2 |T_delta(k0, theta0, t)|**2

(the 2/(2pi)^3 in that derivation being the trace of the transverse projector P_ij).
So the ``*bar`` inputs here are the PLAIN products of the transfer functions (|T|**2 etc.,
NOT carrying the power-spectrum path's unnormalised ``int_0^pi sin(theta) dtheta = 2``), and
the physical correlator is ``(1/2)(dB/B0)**2`` times them.  Every bar entering the assembly is
the same kind of zero-lag bilinear under the same measure, so that prefactor multiplies all of
them uniformly; since the assembly is linear in the bars, applying it once at the end is exact.

VERIFIED: feeding ``deltamrms`` through this convention reproduces the independently validated
clumping transfer function (``sound_wave_simple/tla_tf.py``) to 1.0000 at every z, and the
simulation's measured ``cor_db_db`` to 0.992-1.003.  The alternative (``bar = 2*|T|**2``) would
have been off by exactly 2 against a first-order result the simulation independently confirms.

Everything else (alpha/beta/gamma, cross_corr, fred11/fproc/fadveclens/fred20, the kappa
continuum source and the ``x1squad`` ODE) is the SAME code the power-spectrum path uses,
via ``pmhd.physics.source_assembly``.

Conventions kept identical to ``angle_avging_TLA.py``: ``Theta``, ``Phix``, ``Phiy`` are
multiplied by ``pars.H(z)`` (lines 87/91/95 there) while ``delta`` and ``dxe`` stay raw.

Deliberate differences from the power-spectrum stage-1 (documented, and appropriate when
comparing against a simulation that initialises its own wave):
  * the transfer functions are solved directly at the exact requested k (the repo k-grid does
    not land on an arbitrary k), with abar/bbar/cbar interpolated cubically in log-k;
  * no TCR stitching factor (``diffmag_Tf_b``): the wave starts from ``b=1`` at ``zstart``.

Usage
-----
    python single_mode_source.py --k 1.0803e-22 --theta 1.5707963 --B0 1e-9 --outdir DIR

or import :func:`single_mode_xe2` / :func:`bars_from_tf` to drive it with your own fields
(e.g. amplitudes measured from a simulation).
"""
import os
import sys
import time
import pickle
import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import splev, splrep, interp1d

from pmhd import cons, pars
from pmhd.data.tables import tables_dir
from pmhd.data.grids import k_grid, z_grid
from pmhd.physics.source_assembly import (
    build_abg, unpack_f2bars, assemble_at_z, assemble_cont_at_z,
    single_mode_reducer, solve_x1squad, CROSS_CORR_KEYS,
    assemble_dop_at_z, doppler_terms_enabled, grav11_enabled,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRE_STORED = tables_dir()   # moment tables only; see pmhd.data.tables


# ---------------------------------------------------------------------------
#  cubic-in-log-k interpolation of the pre-stored tables onto an arbitrary k
# ---------------------------------------------------------------------------
def kinterp_master(master, karr, k0, zgrid):
    """Interpolate a 69-entry spline master (abar/bbar/cbar) onto ``k0``.

    Cubic in log-k at each z of ``zgrid``, then re-splined in z.  Returns a tck.
    """
    order = np.argsort(karr)
    logk, logk0 = np.log10(karr[order]), np.log10(k0)
    tab = np.array([splev(zgrid, master[i]) for i in range(len(karr))]).T   # (nz, nk)
    vals = np.array([interp1d(logk, tab[iz, order], kind='cubic')(logk0) for iz in range(len(zgrid))])
    return splrep(zgrid[::-1], vals[::-1])


def kinterp_f2bar_moments(k0, zarr):
    """f2bar moment columns interpolated (cubic in log-k) onto ``k0`` and re-splined onto ``zarr``.

    IMPORTANT: the ``f2bars_*_dict_zch_{zind}.pkl`` files are indexed by the REPO's
    ``z_grid()`` index (1900 -> 601, step -1), not by position in an arbitrary ``zarr``.
    So the tables are first reduced to ``k0`` over the repo z-grid and then splined in z to
    the requested ``zarr`` -- the same two-step used by the power-spectrum path's consumers.

    Returns a list of ``moms`` dicts (one per entry of ``zarr``) in the layout of
    :func:`pmhd.physics.source_assembly.unpack_f2bars`, each per-k value a length-1 array.
    """
    karr = k_grid()
    zgrid = z_grid()
    order = np.argsort(karr)
    logk, logk0 = np.log10(karr[order]), np.log10(k0)

    # --- step 1: reduce to k0 on the repo z-grid ---
    on_zgrid = None
    for zind in range(len(zgrid)):
        with open(PRE_STORED / f'f2bars_dict/f2bars_m0_dict_zch_{zind}.pkl', 'rb') as f:
            m0 = pickle.load(f)
        with open(PRE_STORED / f'f2bars_dict/f2bars_m1_dict_zch_{zind}.pkl', 'rb') as f:
            m1 = pickle.load(f)
        with open(PRE_STORED / f'f2bars_dict/f2bars_p1_dict_zch_{zind}.pkl', 'rb') as f:
            p1 = pickle.load(f)
        full = unpack_f2bars(m0, m1, p1)
        if on_zgrid is None:
            on_zgrid = {key: np.zeros(len(zgrid)) for key in full}
        for key, val in full.items():
            arr = np.asarray(val)
            if arr.ndim == 0 or arr.size == 1:          # taus_phi, dfdnu: k-independent
                on_zgrid[key][zind] = float(arr)
            else:
                on_zgrid[key][zind] = float(interp1d(logk, arr[order], kind='cubic')(logk0))

    # --- step 2: spline in z onto the requested grid ---
    on_zarr = {key: splev(zarr, splrep(zgrid[::-1], vals[::-1]))
               for key, vals in on_zgrid.items()}
    return [{key: (float(vals[i]) if key in ('taus_phi', 'dfdnu') else np.array([vals[i]]))
             for key, vals in on_zarr.items()} for i in range(len(zarr))]


# ---------------------------------------------------------------------------
#  stage 1: first-order transfer functions at the exact (k, theta, B0)
# ---------------------------------------------------------------------------
def solve_first_order(k0, theta, B0, zarr, zstart=2500.0, zhandoff=1550.0, zend=100.0,
                      abar_tck=None, bbar_tck=None, cbar_tck=None):
    """Saha -> TLA transfer functions at one mode, BOTH branches.

    Returns ``(delta, Theta_raw, dxe, Phiy, Phix)`` sampled on ``zarr``.  ``Theta_raw``, ``Phiy``
    and ``Phix`` are the repo's non-dimensional ``Theta = ikv_z/H``, ``Phiy = ikv_y/H``,
    ``Phix = ikv_x/H``; the ``*H`` convention is applied in :func:`bars_from_tf`.

    Two independent systems are integrated, matching the repo's own pipeline:

    * **magnetosonic** -- ``FSRsahamag_hom`` -> ``FSRTLAmag``, state ``[delta, Theta, Phiy, b]``
      (+ ``dxe`` after the handoff), IC ``[0,0,0,1]`` at ``zstart`` (zero fluid, unit dB/B0);
    * **Alfven** -- ``FSRsahaalf`` -> ``FSRTLAalf``, state ``[Phix, bx]``, IC ``[0, 1]`` at
      ``zstart`` (the same normalisation; cf. ``FSR_saha_Tfs.py``'s ``fssahaalfinteg(..., 0, 1)``
      and ``FSR_TLA_Tfs.py:119``).

    The Alfven branch is what populates ``PhixPhixbar``, the other half of ``fred11``'s vorticity
    term ``(PhixPhixbar + PhiyPhiybar)*(dx_D2_p + dx_D2_m)`` (source_assembly.py:228-230).  It was
    previously not solved at all, so ``PhixPhixbar`` was identically zero and theta != pi/2 was
    untrustworthy.

    At theta = pi/2 the addition is inert: ``FSRsahaalf``/``FSRTLAalf`` couple Phix and bx only
    through ``cos(theta)`` (``Phix2bx``, ``bx2Phix``), so with ``Phix(zstart) = 0`` the branch is
    analytically zero for all z.  Numerically it integrates to ``max|Phix| ~ 1e-13`` (LSODA noise
    against the ``bx = 1`` scale), i.e. ~1e-26 in ``PhixPhixbar`` relative to ``ThetaThetabar`` --
    the theta = pi/2 ``xe2`` is unchanged to every printed digit.

    Off theta = pi/2 the Alfven branch is not a correction but the DOMINANT half of the vorticity
    term: ``PhixPhixbar/PhiyPhiybar`` = 1.1e3 / 1.5e2 / 15 at theta = 60 / 45 / 30 degrees.
    """
    xe_full, xesaha_full = pars.xe_full, pars.xesaha_full

    def Acon(z):  return (1 - xe_full(z)) * pars.nh(z) * pars.sigmaa(z)
    def B1con(z): return xe_full(z)**2 * pars.nh(z) * (pars.alpha1s(z) - pars.alphab(z))
    def B2con(z): return xe_full(z) * pars.nh(z) * (pars.alpha1s(z) - pars.alphab(z)) * (2 - xe_full(z)) / (1 - xe_full(z))

    sol_s = solve_ivp(pars.FSRsahamag_hom, [zstart, zend], [0, 0, 0, 1],
                      args=(k0, theta, B0, xesaha_full, xesaha_full),
                      method='LSODA', dense_output=True, atol=1e-9, rtol=1e-6)
    di, Ti, Pi, bi = sol_s.sol(zhandoff)
    xei = -xesaha_full(zhandoff) * ((1 - xesaha_full(zhandoff)) / (2 - xesaha_full(zhandoff))) * di

    af = (lambda z, kind: splev(z, abar_tck))
    bf = (lambda z, kind: splev(z, bbar_tck))
    cf = (lambda z, kind: splev(z, cbar_tck))
    sol_t = solve_ivp(pars.FSRTLAmag, [zhandoff, zend], np.array([di, Ti, Pi, bi, xei]),
                      args=(k0, 0, theta, B0, xe_full, xe_full, af, bf, cf, Acon, B1con, B2con),
                      method='LSODA', dense_output=True, atol=1e-9, rtol=1e-6)
    st = sol_t.sol(zarr)

    # ---- Alfven branch: [Phix, bx], IC [0, 1] at zstart, same handoff redshift ----
    sol_sa = solve_ivp(pars.FSRsahaalf, [zstart, zend], [0.0, 1.0],
                       args=(k0, theta, B0, xesaha_full),
                       method='LSODA', dense_output=True, atol=1e-9, rtol=1e-6)
    Pxi, bxi = sol_sa.sol(zhandoff)
    sol_ta = solve_ivp(pars.FSRTLAalf, [zhandoff, zend], np.array([Pxi, bxi]),
                       args=(k0, theta, B0, xe_full),
                       method='LSODA', dense_output=True, atol=1e-9, rtol=1e-6)
    Phix = sol_ta.sol(zarr)[0]

    # state = [delta, Theta, Phiy, b, dxe]; Phiy and Phix are both returned so the fred11
    # vorticity term (PhixPhixbar + PhiyPhiybar) is complete away from theta = pi/2.
    return st[0], st[1], st[4], st[2], Phix


# ---------------------------------------------------------------------------
#  stage 2: single-mode bilinear products (no theta average)
# ---------------------------------------------------------------------------
def bars_from_tf(delta, Theta_raw, dxe, zarr, Phix=None, Phiy=None):
    """Single-mode analogues of ``angle_avging_TLA``'s ``*bar`` products.

    ``Theta``/``Phix``/``Phiy`` are multiplied by ``pars.H(z)`` and ``delta``/``dxe`` are raw,
    exactly as in ``angle_avging_TLA.py`` (lines 87/91/93/95/97).  No ``sin(theta)`` measure
    and no stochastic parity relations: this is one mode at one angle.

    Each value is a per-z list of length-1 arrays, ready for
    :func:`pmhd.physics.source_assembly.assemble_at_z` with a single-mode reducer.
    """
    H = pars.H(zarr)
    Th = H * Theta_raw
    Px = H * Phix if Phix is not None else np.zeros_like(delta)
    Py = H * Phiy if Phiy is not None else np.zeros_like(delta)
    return {
        'deltamdeltambar': delta * delta,
        'xexebar':         dxe * dxe,
        'xedeltambar':     dxe * delta,
        'deltamThetabar':  delta * Th,
        'xeThetabar':      dxe * Th,
        'ThetaThetabar':   Th * Th,
        'PhixPhixbar':     Px * Px,
        'PhiyPhiybar':     Py * Py,
        'bxbxbar':         np.zeros_like(delta),
        'bybybar':         np.zeros_like(delta),
        'PhiyThetabar':    Py * Th,
        'PhixPhixprimebar':   np.zeros_like(delta),
        'PhiyPhiyprimebar':   np.zeros_like(delta),
        'ThetaThetaprimebar': np.zeros_like(delta),
    }


# ---------------------------------------------------------------------------
#  stages 3-4: assembly + the x1squad ODE
# ---------------------------------------------------------------------------
def single_mode_xe2(k0, zarr, bars, moms_per_z=None, verbose=True):
    """Assemble ``cross_corr``/``ftot``/``conterm`` for one mode and solve ``x1squad``.

    Returns a dict with the cross-correlations, the four ftot terms, the continuum source,
    the ODE solution ``x1s2`` and ``xe2 = -x1s2 + <delta_m dxe>`` (no spectrum prefactor).
    """
    xe_full = pars.xe_full
    nz = len(zarr)
    karr1 = np.array([k0])
    reduce_1 = single_mode_reducer(k0)

    # alpha/beta/gamma at the single k, from the repo construction
    with open(PRE_STORED / 'abarinterpmaster.pkl', 'rb') as f: abarm = pickle.load(f)
    with open(PRE_STORED / 'bbarinterpmaster.pkl', 'rb') as f: bbarm = pickle.load(f)
    with open(PRE_STORED / 'cbarinterpmaster.pkl', 'rb') as f: cbarm = pickle.load(f)
    karr = k_grid()
    a_tck = kinterp_master(abarm, karr, k0, zarr)
    b_tck = kinterp_master(bbarm, karr, k0, zarr)
    c_tck = kinterp_master(cbarm, karr, k0, zarr)
    abg_all = build_abg(zarr, karr1,
                        lambda z, kind: splev(z, a_tck),
                        lambda z, kind: splev(z, b_tck),
                        lambda z, kind: splev(z, c_tck), xe_full)

    if moms_per_z is None:
        moms_per_z = kinterp_f2bar_moments(k0, zarr)

    tausall = pars.taus(zarr, xe_full)
    Deltah  = pars.Deltah(zarr)
    H       = pars.H(zarr)
    paball  = pars.pab(zarr)
    xearr   = xe_full(zarr)

    cc = {key: np.zeros(nz) for key in CROSS_CORR_KEYS}
    fred11 = np.zeros(nz); fproc = np.zeros(nz)
    fadveclens = np.zeros(nz); fred20 = np.zeros(nz); cont = np.zeros(nz); fgrav11 = np.zeros(nz)
    use_dop = doppler_terms_enabled()
    fdop = np.zeros(nz); fdopbar = np.zeros(nz); fv2 = np.zeros(nz)

    for zind in range(nz):
        bars_z = {key: np.atleast_1d(np.asarray(val)[zind]) for key, val in bars.items()}
        abg_z  = {key: abg_all[key][zind, :] for key in abg_all}
        bg = dict(z=zarr[zind], H=H[zind], taus=tausall[zind],
                  pab=paball[zind], xe=xearr[zind], Deltah=Deltah[zind])
        cc_row, fred11[zind], fproc[zind], fadveclens[zind], fred20[zind], fgrav11[zind] = assemble_at_z(
            bars_z, abg_z, moms_per_z[zind], bg, reduce_1)
        if use_dop:
            dop = assemble_dop_at_z(bars_z, abg_z, moms_per_z[zind], cc_row, bg, reduce_1)
            fdop[zind], fdopbar[zind], fv2[zind] = dop['fdop'], dop['fdopbar'], dop['fv2']
        for key in CROSS_CORR_KEYS:
            cc[key][zind] = cc_row[key]
        cont[zind] = assemble_cont_at_z(bars_z, karr1, zarr[zind], xe_full, reduce_1)

    ftot = fred11 + fproc + fadveclens + fred20
    if use_dop:
        ftot = ftot + fdop + fdopbar + fv2
    if grav11_enabled():
        ftot = ftot + fgrav11

    abar2 = np.array([moms_per_z[i]['dfdnu'] for i in range(nz)])
    cbar2 = np.array([moms_per_z[i]['taus_phi'] for i in range(nz)])
    hold = solve_x1squad(zarr, cc, ftot, abar2, cbar2, cont, xe_full,
                         zspan=(zarr[0], zarr[-1]))
    x1s2 = hold.sol(zarr)[0]
    xe2 = -x1s2 + cc['deltamxecross']

    out = dict(z=zarr, cross_corr=cc, fred11=fred11, fproc=fproc,
               fadveclens=fadveclens, fred20=fred20, ftot=ftot, cont=cont,
               fdop=fdop, fdopbar=fdopbar, fv2=fv2, fgrav11=fgrav11,
               x1s2=x1s2, xe2=xe2, k=k0)
    if verbose:
        _report_terms(out)
    return out


def _report_terms(out):
    """Print the per-term source budget so fred20 / fadveclens / conterm sizes are visible."""
    z = out['z']
    print("\n  per-term source budget (single mode):")
    print("      z      fred11       fproc        fadveclens   fred20       ftot         conterm      xe2")
    for zq in (1500, 1400, 1300, 1200, 1100, 1000, 900, 800):
        if zq > z.max() or zq < z.min():
            continue
        i = int(np.argmin(np.abs(z - zq)))
        print("   %5.0f  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e  %+.3e" % (
            z[i], out['fred11'][i], out['fproc'][i], out['fadveclens'][i],
            out['fred20'][i], out['ftot'][i], out['cont'][i], out['xe2'][i]))
    tot = np.max(np.abs(out['ftot'])) or 1.0
    print("   peak |term|/|ftot|_peak :  fred11 %.3f  fproc %.3f  fadveclens %.3e  fred20 %.3e" % (
        np.max(np.abs(out['fred11']))/tot, np.max(np.abs(out['fproc']))/tot,
        np.max(np.abs(out['fadveclens']))/tot, np.max(np.abs(out['fred20']))/tot))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=float, default=1e3/(3*cons.mpc),
                   help="physical wavenumber in cm^-1 (default: 1e3/(3 Mpc), the sound-wave mode)")
    p.add_argument("--theta", type=float, default=np.pi/2, help="angle between k and B0 [rad]")
    p.add_argument("--B0", type=float, default=1e-9, help="mean field in G (default 1 nG)")
    p.add_argument("--zstart", type=float, default=2500.0)
    p.add_argument("--zhandoff", type=float, default=1550.0)
    p.add_argument("--zend", type=float, default=100.0)
    p.add_argument("--zmin", type=float, default=605.0,
                   help="low-z end of the working grid (the f2bar tables stop at z=601)")
    p.add_argument("--outdir", type=str, default=None)
    a = p.parse_args()

    print(time.ctime())
    # working grid: the TLA handoff down to the f2bar table floor
    zarr = np.concatenate((np.arange(a.zhandoff, 800, -1),
                           np.logspace(np.log10(800), np.log10(a.zmin), 250)), axis=None)

    with open(PRE_STORED / 'abarinterpmaster.pkl', 'rb') as f: abarm = pickle.load(f)
    with open(PRE_STORED / 'bbarinterpmaster.pkl', 'rb') as f: bbarm = pickle.load(f)
    with open(PRE_STORED / 'cbarinterpmaster.pkl', 'rb') as f: cbarm = pickle.load(f)
    karr = k_grid()
    a_tck = kinterp_master(abarm, karr, a.k, zarr)
    b_tck = kinterp_master(bbarm, karr, a.k, zarr)
    c_tck = kinterp_master(cbarm, karr, a.k, zarr)

    delta, Theta_raw, dxe, Phiy, Phix = solve_first_order(
        a.k, a.theta, a.B0, zarr, zstart=a.zstart, zhandoff=a.zhandoff, zend=a.zend,
        abar_tck=a_tck, bbar_tck=b_tck, cbar_tck=c_tck)
    bars = bars_from_tf(delta, Theta_raw, dxe, zarr, Phiy=Phiy, Phix=Phix)
    out = single_mode_xe2(a.k, zarr, bars)

    outdir = Path(a.outdir) if a.outdir else (PROJECT_ROOT / "data/outputs/single_mode")
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"k{a.k:.4e}_B{round(1e12*a.B0)}pG_th{a.theta:.3f}"
    np.savez(outdir / f"single_mode_{tag}.npz",
             z=zarr, delta=delta, Theta=Theta_raw, dxe=dxe,
             fred11=out['fred11'], fproc=out['fproc'], fadveclens=out['fadveclens'],
             fred20=out['fred20'], ftot=out['ftot'], cont=out['cont'],
             x1s2=out['x1s2'], xe2=out['xe2'],
             **{f"cc_{k}": v for k, v in out['cross_corr'].items()})
    print(f"\nwrote {outdir}/single_mode_{tag}.npz")
    print(time.ctime())


if __name__ == '__main__':
    main()
