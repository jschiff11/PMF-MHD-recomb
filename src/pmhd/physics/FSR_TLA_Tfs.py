import numpy as np
import os
import time
import sys
from pathlib import Path

from scipy.integrate import solve_ivp
from scipy.interpolate import splrep, splev

from pmhd import cons, pars
from pmhd.data.grids import (
    k_grid,
    theta_grid,
    load_or_generate_z_arrays,
    load_or_generate_B0arr,
)

import pickle

ODE_METHOD = os.environ.get("PMHD_ODE_METHOD", "LSODA")

# Multiply all solver tolerances by this factor (e.g. 1e-2 for a 100x-tighter
# convergence test); default 1.0 leaves the production tolerances unchanged.
# Floored at rtol=1e-12 / atol=1e-14 (LSODA feasibility limit).
TOL_SCALE = float(os.environ.get("PMHD_TOL_SCALE", "1.0"))

def _rtol(base):
    return max(base * TOL_SCALE, 1e-12)

def _atol(base):
    return max(base * TOL_SCALE, 1e-14)


def main(input_bind, input_kind):
    print(time.ctime())

    karr = k_grid()
    thetaarr = theta_grid()

    zcrossarr, zfsarr = load_or_generate_z_arrays()
    B0arr = load_or_generate_B0arr()
    zstartarr = np.append( np.full(np.argwhere(zfsarr<=1900)[0,0],1900), zfsarr[np.argwhere(zfsarr<=1900)[0,0]:] )

    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    # Output base directory (override with PMHD_OUTDIR; reads He Saha-FSR + writes He TLA)
    OUTBASE = Path(os.environ.get("PMHD_OUTDIR", str(PROJECT_ROOT / "data/outputs"))) / "Tfs"

    # For this run, create subdir per B0
    Bdir = OUTBASE / f"B_{round(1e12*B0arr[input_bind])}pG"
    Bdir.mkdir(parents=True, exist_ok=True)

    xe_full = pars.xe_full           # hydrogen ionization (recombination, Acon/B1con/B2con, xeinit)
    xe_full_He = pars.xe_full_He     # He-inclusive free-electron fraction (drag + pressure)
    xesaha_full = pars.xesaha_full


    def Acon(z):
        return (1 - xe_full(z)) *  pars.nh(z) *  pars.sigmaa(z)

    def B1con(z):
        return xe_full(z)**2 * pars.nh(z) *  (pars.alpha1s(z) - pars.alphab(z))

    def B2con(z):
        return xe_full(z) * pars.nh(z) *  (pars.alpha1s(z) - pars.alphab(z)) * (2-xe_full(z))/(1-xe_full(z))


    with open(PROJECT_ROOT/'src/pmhd/data/pre_stored_data/abarinterpmaster.pkl', 'rb') as f:
        abarinterpmaster = pickle.load(f)

    with open(PROJECT_ROOT/'src/pmhd/data/pre_stored_data/bbarinterpmaster.pkl', 'rb') as f:
        bbarinterpmaster = pickle.load(f)

    with open(PROJECT_ROOT/'src/pmhd/data/pre_stored_data/cbarinterpmaster.pkl', 'rb') as f:
        cbarinterpmaster = pickle.load(f)

    def abarinterpfunc(z,kind):
        return splev(z, abarinterpmaster[kind])

    def bbarinterpfunc(z,kind):
        return splev(z, bbarinterpmaster[kind])

    def cbarinterpfunc(z,kind):
        return splev(z, cbarinterpmaster[kind])

    zend = 600

    fssahaalf = np.load(Bdir / f"FSRsahaalf_k{input_kind}.npy")
    fssahamag = np.load(Bdir / f"FSRsahamag_k{input_kind}.npy")
    
   
    print(time.ctime())

    resultsalf = np.zeros(( len(thetaarr) , 2, 10**4))
    resultsmag = np.zeros(( len(thetaarr), 5, 10**4 ))
    for thetaind in range(len(thetaarr)):
        zsplhold = np.logspace(np.log10(zfsarr[input_kind]),np.log10(zend),num = 10**4)[::-1] 
        Phixinit = splev( zstartarr[input_kind], splrep( zsplhold, fssahaalf[thetaind,0,::-1] ) )
        bxinit = splev( zstartarr[input_kind], splrep( zsplhold, fssahaalf[thetaind,1,::-1] ) )

        deltainit = splev( zstartarr[input_kind], splrep( zsplhold, fssahamag[thetaind,0,::-1] ) )
        Thetainit = splev( zstartarr[input_kind], splrep( zsplhold, fssahamag[thetaind,1,::-1] ) )
        Phiyinit = splev( zstartarr[input_kind], splrep( zsplhold, fssahamag[thetaind,2,::-1] ) )
        byinit = splev( zstartarr[input_kind], splrep( zsplhold, fssahamag[thetaind,3,::-1] ) )
        
        xeinit = -xesaha_full(zstartarr[input_kind]) * (
                (1-xesaha_full(zstartarr[input_kind]))/(2-xesaha_full(zstartarr[input_kind])))*deltainit

        sol = solve_ivp( pars.FSRTLAmag, [zstartarr[input_kind], zend], np.array([deltainit, Thetainit, Phiyinit, byinit, xeinit]) , args=(
                karr[input_kind], input_kind, thetaarr[thetaind], B0arr[input_bind], xe_full, xe_full_He, abarinterpfunc, bbarinterpfunc, cbarinterpfunc,
                Acon, B1con, B2con
            ), method = ODE_METHOD, dense_output=True, atol=_atol(1e-9), rtol = _rtol(1e-6) )

        resultsmag[thetaind,:,:] = sol.sol(np.logspace(np.log10(zstartarr[input_kind]),np.log10(zend), num = 10**4))

        # NB: this call historically used scipy's default tolerances
        # (rtol=1e-3, atol=1e-6); they are written out explicitly here so the
        # TOL_SCALE convergence test can tighten them too.
        sol = solve_ivp( pars.FSRTLAalf, [zstartarr[input_kind], zend], np.array([Phixinit, bxinit]), args=(
                karr[input_kind], thetaarr[thetaind], B0arr[input_bind],xe_full_He), method = ODE_METHOD,
                 dense_output=True, atol=_atol(1e-6), rtol=_rtol(1e-3) )

        resultsalf[thetaind,:,:] = sol.sol(np.logspace(np.log10(zstartarr[input_kind]),np.log10(zend), num = 10**4))
        
    # -----------------------------
    # Save the results
    # -----------------------------
    np.save(Bdir / f"FSRTLAalf_k{input_kind}.npy", resultsalf)
    np.save(Bdir / f"FSRTLAmag_k{input_kind}.npy", resultsmag)
    
    print(time.ctime())


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))