import sys
from pathlib import Path

# Path to lib/
LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt

import time
import os

  
from scipy.integrate import odeint, solve_ivp

from astropy.cosmology import Planck18 as cosmo

from scipy.interpolate import splrep, splev, splint
from scipy.linalg import solve_banded as sb

import pickle
import argparse

from pmhd import cons, pars
from pmhd.data.grids import (
    k_grid, z_grid,
    theta_grid,
    theta_gridfull,
    load_or_generate_z_arrays,
    load_or_generate_B0arr,
)

import math
from scipy.integrate import cumulative_trapezoid


# ---------------------------------------------------------------------------
# Photon-path optical depth (manuscript Eqs. 65-69; manuscript sync 2026-10, PREREG in
# paper_data_plots/manuscript_sync/PREREG.md).
#
# tau(x, n, eta) = int_eta^eta0 d eta' taudot(x_gamma(eta'), eta'),  x_gamma(eta') = x - c n (eta0 - eta'),
# so for one Fourier mode k the perturbation delta tau carries the phase exp(i k.n c (eta' - eta)) along the
# path. Averaging the products <delta tau delta tau> and <delta tau delta taudot> over the photon direction n
# (the transfer-function angle theta is w.r.t. B0 and independent of n) gives the kernel
#     j0(k c |eta' - eta''|) = sin(x)/x
# between any two path elements. With the tau increments m_j (Simpson, per fine-grid interval) at comoving
# distances chi_j = c int dz/H:
#     S(z_n) = sum_{i,j<=n} m_i m_j j0(k |chi_i - chi_j|)        (replaces dtau^2)
#     C(z_n) = (delta taudot/taudot)(z_n) sum_{j<=n} m_j j0(k |chi_n - chi_j|)   (replaces dtau * d_taudot)
# where the sums run over the path from the observer-side cutoff (z = 600) up to the scattering redshift z_n.
# j0 -> 1 recovers the fixed-position result S = dtau^2 exactly.
# PMHD_LOS_KERNEL: "1" (default) kernel on; "0" the pre-2026-10 fixed-position code path (bit-identical);
# "unit" the kernel code path with j0 forced to 1 (diagnostic of the restructuring alone).
# ---------------------------------------------------------------------------
LOS_KERNEL = os.environ.get("PMHD_LOS_KERNEL", "1")


def _chi_spline():
    """Comoving distance chi(z) = c int_600^z dz'/H(z') [cm] as a spline (only differences are used)."""
    # the TCR-era path of high-k modes reaches z_cross ~ 1e8; cover it so chi is never extrapolated
    zz = np.logspace(np.log10(600.0), np.log10(3.0e8), 400001)
    chi = cumulative_trapezoid(cons.c/pars.H(zz), zz, initial=0.0)
    return splrep(zz, chi)


def _los_accumulate(k, chi_pts, m, chi_prev=None, m_prev=None, S_prev=0.0, unit=False, window=200.0, block=256):
    """Cell-integrated photon-path kernel along one path segment (ascending z; chi increasing).

    The tau increment m_n belongs to the cell [chi_{n-1}, chi_n] (m[0] = 0), on which the rate A_n = m_n/dchi_n is
    taken constant. With P(p,q) = F(k|chi_p - chi_q|), F(x) = x Si(x) + cos x - 1, the exact double integral of
    j0(k|chi'-chi''|) over cells n and j is K_nj = [P(j,n-1) - P(j,n) - P(j-1,n-1) + P(j-1,n)]/k^2, so
        S_n = S_{n-1} + A_n^2 K_nn + 2 A_n sum_{j<n} A_j K_nj                           (replaces dtau^2)
        G_n = sum_{j<=n} A_j [Q(n,j-1) - Q(n,j)],  Q(p,q) = Si(k|chi_p - chi_q|)/k        (replaces dtau in dtau*dtaudot)
    K_nj -> dchi_n dchi_j for k dchi << 1 (discrete sum, and S = (sum m)^2 for j0 -> 1) and K_nn -> pi dchi_n/k for
    k dchi >> 1 (Markov limit), so no grid refinement is needed at high k. Pairs further apart than `window`/k are
    dropped (relative error ~ 1/window). `unit=True` replaces F, Si by their j0 -> 1 limits (x^2/2, x).
    chi_prev, m_prev, S_prev: an earlier segment of the same path (the FSR piece when the TCR piece is done).
    m (and m_prev) may be (N,) for one path or (T, N) for T paths sharing the same chi grid (the theta values of one
    mode): the kernel blocks, which dominate the cost, are then built once and applied to all T rows. S, G have the
    shape of m; S_prev is a scalar or (T,).
    """
    from scipy.special import sici
    if unit:
        F = lambda x: 0.5*x*x; Si = lambda x: x
    else:
        F = lambda x: x*sici(x)[0] + np.cos(x) - 1.0; Si = lambda x: sici(x)[0]
    one_d = (np.ndim(m) == 1)
    m2 = np.atleast_2d(m); mp2 = None if m_prev is None else np.atleast_2d(m_prev)
    chi_all = chi_pts if chi_prev is None else np.concatenate([chi_prev, chi_pts])
    m_all = m2 if mp2 is None else np.concatenate([mp2, m2], axis=1)
    n0 = 0 if chi_prev is None else len(chi_prev)            # index of the first point of this segment in chi_all
    dchi = np.diff(chi_all, prepend=chi_all[0]); dchi[0] = 1.0
    if not np.all(dchi[1:] >= 0):
        raise ValueError("photon-path kernel: chi(z) must be non-decreasing along the path")
    A = m_all/dchi[None, :]; A[:, 0] = 0.0
    if n0 > 0: A[:, n0] = 0.0                                 # the segment's first point closes no cell
    T, N = A.shape; dS = np.zeros((T, N)); G = np.zeros((T, N))
    wmax = np.inf if unit else window/k                       # the j0 -> 1 check must keep every pair
    for b0 in range(max(n0, 1), N, block):
        b1 = min(b0 + block, N)
        rows = np.arange(b0 - 1, b1)                          # need point n-1 for cell n
        jmin = int(np.searchsorted(chi_all, chi_all[b0 - 1] - wmax)); jmin = max(jmin, 1)
        cols = np.arange(jmin - 1, b1)                        # need point j-1 for cell j
        X = k*np.abs(chi_all[rows][:, None] - chi_all[cols][None, :])
        P = F(X); Q = Si(X)/k
        # K[n, j] for n in [b0,b1), j in [jmin, b1): P(j,n-1) - P(j,n) - P(j-1,n-1) + P(j-1,n)
        rn = np.arange(b0, b1) - (b0 - 1); rnm1 = rn - 1       # row offsets of n and n-1
        cj = np.arange(jmin, b1) - (jmin - 1); cjm1 = cj - 1   # column offsets of j and j-1
        K = (P[np.ix_(rnm1, cj)] - P[np.ix_(rn, cj)] - P[np.ix_(rnm1, cjm1)] + P[np.ix_(rn, cjm1)])/k**2
        nn = np.arange(b0, b1)[:, None]; jj = np.arange(jmin, b1)[None, :]
        Aj = A[:, jmin:b1]; An = A[:, b0:b1]                  # (T, nc), (T, nr)
        Koff = np.where(jj < nn, K, 0.0)                      # j < n
        Kdiag = K[np.arange(b1 - b0), np.arange(b0, b1) - jmin]
        dS[:, b0:b1] = An**2*Kdiag[None, :] + 2.0*An*(Aj @ Koff.T)
        Qn = Q[np.ix_(rn, cj)]; Qnm1j = Q[np.ix_(rn, cjm1)]
        G[:, b0:b1] = Aj @ np.where(jj <= nn, Qnm1j - Qn, 0.0).T
    S = np.reshape(S_prev, (-1, 1)) + np.cumsum(dS[:, n0:], axis=1)
    return (S[0], G[0, n0:]) if one_d else (S, G[:, n0:])



def main(input_bind, input_kind, tb=False):
    print(time.ctime())
    karr = k_grid()
    zarr = z_grid()
    thetaarr = theta_gridfull()

    zcrossarr, zfsarr = load_or_generate_z_arrays()
    zstartarr = np.append( np.full(np.argwhere(zfsarr<=1900)[0,0],1900), zfsarr[np.argwhere(zfsarr<=1900)[0,0]:] )

    B0arr = load_or_generate_B0arr()

    xe_full = pars.xe_full           # hydrogen ionization (used only via dxefunc/deltafunc, unchanged)
    xe_full_He = pars.xe_full_He     # He-inclusive free electrons: background weight on the
                                      # density perturbation in the Thomson optical depth (n_e ~ xe*nh)

    zsteps = 10**4

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    # Output base directory (override with PMHD_OUTDIR env var; default = repo data/outputs)
    OUTBASE = Path(os.environ.get("PMHD_OUTDIR", str(PROJECT_ROOT / "data/outputs")))
    # For this run, create subdir per B0
    Bdir = OUTBASE / f"Tfs/B_{round(1e12*B0arr[input_bind])}pG"
    
    diffalf_Tf_b = np.load(Bdir / f"TCRalf_k{input_kind}.npy")[:,1,-1].reshape(17,1)

    diffmag_Tf_b = np.load(Bdir / f"TCRmag_k{input_kind}.npy")[:,3,-1].reshape(17,1)
    
    # --tb: the T_b-coupled FSR transfer functions (same column layout for delta and delta x_e; TCR segment shared)
    maghold = np.load((OUTBASE / f"Tfs/Tb/B_{round(1e12*B0arr[input_bind])}pG" / f"FSRTLAmag_Tb_k{input_kind}.npy") if tb else (Bdir / f"FSRTLAmag_k{input_kind}.npy"))
    byhold = diffmag_Tf_b*maghold[:,3,:]
    Phiyhold = diffmag_Tf_b*maghold[:,2,:]
    Thetahold = diffmag_Tf_b*maghold[:,1,:]
    deltahold = diffmag_Tf_b*maghold[:,0,:]
    dxehold = diffmag_Tf_b*maghold[:,4,:]

    dtaufin = np.zeros( ( len(zarr), len(thetaarr) ) )
    d_taudotfin = np.zeros( ( len(zarr), len(thetaarr) ) )

    def deltafunc(z,thetaind):
        hold = splrep(np.logspace(np.log10(zstartarr[input_kind]),np.log10(600),10**4)[::-1],deltahold[thetaind,::-1])
        return splev(z, hold)
    def dxefunc(z,thetaind):
        hold = splrep(np.logspace(np.log10(zstartarr[input_kind]),np.log10(600),10**4)[::-1],dxehold[thetaind,::-1])
        return splev(z, hold)
    def dtauinteg(z,xe,thetaind):
        return cons.c*pars.nh(z)*cons.sigmat*(xe(z)*deltafunc(z,thetaind)+dxefunc(z,thetaind) )/pars.H(z)/(1+z)
    def d_taudot(z,xe,thetaind):
        return (deltafunc(z,thetaind)+dxefunc(z,thetaind)/xe(z) )
    
    # photon-path kernel state (see module docstring above main)
    use_los = LOS_KERNEL != "0"; los_unit = (LOS_KERNEL == "unit")
    kcm = float(karr[input_kind])
    chi_tck = _chi_spline() if use_los else None
    Sfin = np.zeros((len(zarr), len(thetaarr))); Cfin = np.zeros((len(zarr), len(thetaarr)))
    chi_fsr = [None]*len(thetaarr); m_fsr = [None]*len(thetaarr); S_fsr_total = np.zeros(len(thetaarr))
    dtau = np.zeros((10**4,17))
    dtauhold = np.zeros((10**4,17))
    
    for thetaind in range(int((len(thetaarr)-1)/2)+1):
        zinterp = np.logspace(np.log10(600),np.log10(zstartarr[input_kind]), num = zsteps)
        zmid = (zinterp[1:] + zinterp[:-1])/2

        dtauintegarr = dtauinteg(zinterp,xe_full_He,thetaind)
        dtauintegarrmid = dtauinteg(zmid,xe_full_He,thetaind)
        
        dtauhold[1:,thetaind] = ((zinterp[1:] - zinterp[:-1])/6)*(dtauintegarr[:-1] + 4*dtauintegarrmid[:] + dtauintegarr[1:])
        dtau[:,thetaind] = np.cumsum(dtauhold[:,thetaind])
    nth = int((len(thetaarr)-1)/2)+1
    if use_los:   # one kernel pass for all theta (same chi grid); the FSR segment is the start of every path
        chi_i = splev(zinterp, chi_tck)
        S_all, G_all = _los_accumulate(kcm, chi_i, dtauhold[:, :nth].T, unit=los_unit)
        for thetaind in range(nth):
            chi_fsr[thetaind] = chi_i; m_fsr[thetaind] = dtauhold[:,thetaind].copy(); S_fsr_total[thetaind] = S_all[thetaind, -1]
        if os.environ.get("PMHD_LOS_DEBUG"):   # validation dump (theta index 4, FSR segment)
            np.savez(os.environ["PMHD_LOS_DEBUG"], z=zinterp, chi=chi_i, m=dtauhold[:,4].copy(), S=S_all[4], G=G_all[4],
                     dtaudot=d_taudot(zinterp,xe_full_He,4), k=kcm)
    for thetaind in range(nth):
        if use_los: S_i, G_i = S_all[thetaind], G_all[thetaind]
        startzind = int(1900 - math.floor(zstartarr[input_kind]))

        
        dtaufin[startzind:, thetaind] = splev(zarr[startzind:],
                                     splrep( zinterp, dtau[:,thetaind] ), der =0 )
        d_taudotfin[startzind:, thetaind] = splev(zarr[startzind:],
                                     splrep( zinterp, d_taudot(zinterp,xe_full_He,thetaind) ), der =0 )
        if use_los:
            Sfin[startzind:, thetaind] = splev(zarr[startzind:], splrep(zinterp, S_i), der=0)
            Cfin[startzind:, thetaind] = splev(zarr[startzind:], splrep(zinterp, d_taudot(zinterp,xe_full_He,thetaind)*G_i), der=0)
        
    
    if zstartarr[input_kind]!=1900:

        # delta-tau accumulated across the FSR range [600, z_fs]. The TCR-era integral
        # continues from this value so dtaufin(z) is always the full integral from the
        # observer side up to z, not just the TCR-range piece.
        dtau_fsr_total = dtau[-1,:].copy()

        maghold = np.load(Bdir / f"TCRmag_k{input_kind}.npy")
        byhold = maghold[:,3,:]
        Phiyhold = maghold[:,2,:]
        Thetahold = maghold[:,1,:]
        deltahold = pars.TCR_DELTA_B*maghold[:,0,:]   # delta_b = 3 delta_gamma on the TCR segment

        def deltafunc(z,thetaind):
            hold = splrep(np.logspace(np.log10(zfsarr[input_kind]),np.log10(zcrossarr[input_kind]),10**4),deltahold[thetaind,::-1])
            return splev(z, hold)
        def dtauinteg(z,xe,thetaind):
            return cons.c*pars.nh(z)*cons.sigmat*(xe(z)*deltafunc(z,thetaind) )/pars.H(z)/(1+z)
        def d_taudot(z,xe,thetaind):
            return (deltafunc(z,thetaind) )
        
        dtau = np.zeros((10**4,17))
        dtauhold = np.zeros((10**4,17))


        for thetaind in range(int((len(thetaarr)-1)/2)+1):
            zinterp = np.logspace(np.log10(zfsarr[input_kind]),np.log10(zcrossarr[input_kind]), num = zsteps)
            zmid = (zinterp[1:] + zinterp[:-1])/2
            
            startzind = int(1900 - math.floor(zstartarr[input_kind]))
    
            dtauintegarr = dtauinteg(zinterp,xe_full_He,thetaind)
            dtauintegarrmid = dtauinteg(zmid,xe_full_He,thetaind)
            
            dtauhold[1:,thetaind] = ((zinterp[1:] - zinterp[:-1])/6)*(dtauintegarr[:-1] + 4*dtauintegarrmid[:] + dtauintegarr[1:])
            dtau[:,thetaind] = np.cumsum(dtauhold[:,thetaind]) + dtau_fsr_total[thetaind]
        if use_los:   # continue every theta's path through the TCR era in one kernel pass
            chi_i = splev(zinterp, chi_tck)
            if os.environ.get("PMHD_LOS_DEBUG_TCR"):   # validation dump: the TCR-segment inputs for every theta
                np.savez(os.environ["PMHD_LOS_DEBUG_TCR"], z=zinterp, chi=chi_i, m=dtauhold[:, :nth].T.copy(), chi_prev=chi_fsr[0],
                         m_prev=np.array(m_fsr[:nth]), S_prev=S_fsr_total[:nth].copy(), k=kcm)
            S_all, G_all = _los_accumulate(kcm, chi_i, dtauhold[:, :nth].T, chi_prev=chi_fsr[0],
                                           m_prev=np.array(m_fsr[:nth]), S_prev=S_fsr_total[:nth], unit=los_unit)
        for thetaind in range(nth):
            if use_los: S_i, G_i = S_all[thetaind], G_all[thetaind]
            startzind = int(1900 - math.floor(zstartarr[input_kind]))


            dtaufin[:startzind, thetaind] = splev(zarr[:startzind],
                                         splrep( zinterp, dtau[:,thetaind] ), der =0 )
            d_taudotfin[:startzind, thetaind] = splev(zarr[:startzind],
                                         splrep( zinterp, d_taudot(zinterp,xe_full_He,thetaind) ), der =0 )
            if use_los:
                Sfin[:startzind, thetaind] = splev(zarr[:startzind], splrep(zinterp, S_i), der=0)
                Cfin[:startzind, thetaind] = splev(zarr[:startzind], splrep(zinterp, d_taudot(zinterp,xe_full_He,thetaind)*G_i), der=0)
        


    for i in range(int((len(thetaarr)-1)/2)+1):
        dtaufin[:,-i+int((len(thetaarr)-1))] = -dtaufin[:,i]
        d_taudotfin[:,-i+int((len(thetaarr)-1))] = -d_taudotfin[:,i]
        Sfin[:,-i+int((len(thetaarr)-1))] = Sfin[:,i]
        Cfin[:,-i+int((len(thetaarr)-1))] = Cfin[:,i]



    dtaudtaubar = np.zeros( len(zarr) ) 
    dtaudtaudotbar = np.zeros( len(zarr) ) 
    

    print(time.ctime()) 

    for z in range(len(zarr)):
        
        if use_los:
            dtaudtaubar[z] = splint(thetaarr[0], thetaarr[-1], splrep(thetaarr, np.sin(thetaarr) * Sfin[z,:]))
            dtaudtaudotbar[z] = splint(thetaarr[0], thetaarr[-1], splrep(thetaarr, np.sin(thetaarr) * Cfin[z,:]))
            continue
        dtaudtaubar[z] = splint(
            thetaarr[0],thetaarr[-1], splrep(
                thetaarr, np.sin(thetaarr)  * dtaufin[z,:]**2 ) )
        dtaudtaudotbar[z] = splint(
            thetaarr[0],thetaarr[-1], splrep(
                thetaarr, np.sin(thetaarr)  * dtaufin[z,:]*d_taudotfin[z,:] ) )
        
        
    ang_avg_TLA = OUTBASE/(f"ang_avg/Tb/B_{round(1e12*B0arr[input_bind])}pG" if tb else f"ang_avg/TLA/B_{round(1e12*B0arr[input_bind])}pG")
    _sfx = "_Tb" if tb else ""
    ang_avg_TLA.mkdir(parents=True, exist_ok=True)
        
    # Save results
    np.save(ang_avg_TLA/f'dtaudtaubar{_sfx}_k{input_kind}.npy', dtaudtaubar)

    np.save(ang_avg_TLA/f'dtaudtaudotbar{_sfx}_k{input_kind}.npy', dtaudtaudotbar)

if __name__=='__main__':
    print(time.ctime())

    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", type=int, default=0)
    parser.add_argument("--kind", type=int, default=0)
    parser.add_argument("--tb", action="store_true", help="T_b-coupled transfer functions -> ang_avg/Tb dtaudtau*_Tb files")

    args = parser.parse_args()

    main(args.bind, args.kind, tb=args.tb)
        
print('job done',time.ctime())

