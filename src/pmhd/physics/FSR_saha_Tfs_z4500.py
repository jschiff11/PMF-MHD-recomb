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


def main(input_bind, input_kind):
    print(time.ctime())
    karr = k_grid()
    thetaarr = theta_grid()

    zcrossarr, zfsarr = load_or_generate_z_arrays()
    B0arr = load_or_generate_B0arr()

    # He-recombination variant ("hom"): hydrogen Saha xe (x_H) and the
    # He-inclusive free-electron fraction (x_tot). The Alfven mode (drag only)
    # uses x_tot; the magnetosonic mode uses FSRsahamag_hom which takes both
    # (x_H in the perturbed-ionization pressure factor, x_tot in the drag).
    xesaha_full = pars.xesaha_full          # hydrogen-only x_H
    xesaha_full_He = pars.xesaha_full_He    # He-inclusive x_tot

    # Direct z-integration (kept for reference; commented out because it hangs at
    # low B0 / theta~pi/4 -- e.g. bind 57/58, 7-8 pG). Use the log(z) version below.
    # def fssahaalfinteg(karr,thetaarr,B0arr,kind,bind,thetaind, zstart, vxinit, bxinit):
    #     sol = solve_ivp(pars.FSRsahaalf, [zstart, 600], [vxinit, bxinit], args=(
    #                 karr[kind], thetaarr[thetaind], B0arr[bind],xesaha_full_He), method = 'LSODA',
    #                 dense_output=True, atol=1e-9, rtol = 1e-6 )
    #     return sol.sol(np.logspace(np.log10(zstart),np.log10(600),num = 10**4))

    # Active: log(z) integration. d/d(log z) = z * d/dz, mathematically equivalent
    # but avoids the LSODA step-size hang.
    def fssahaalfinteg(karr,thetaarr,B0arr,kind,bind,thetaind, zstart, vxinit, bxinit):
        def rhs_logz(lz, v):
            z = np.exp(lz)
            dv = pars.FSRsahaalf(z, v, karr[kind], thetaarr[thetaind], B0arr[bind], xesaha_full_He)
            return [dv[0]*z, dv[1]*z]
        lzstart = np.log(zstart)
        lzgrid  = np.linspace(lzstart, np.log(600), num=10**4)
        sol = solve_ivp(rhs_logz, [lzstart, np.log(600)], [vxinit, bxinit],
                        method='LSODA', dense_output=True, atol=1e-9, rtol=1e-6)
        return sol.sol(lzgrid)

    # Direct z-integration (kept for reference; commented out -- use log(z) below).
    # def fssahainteg(karr,thetaarr,B0arr,kind,bind,thetaind, zstart, deltainit, vzinit, vyprimeinit, byinit):
    #     sol = solve_ivp(pars.FSRsahamag_hom, [zstart, 600], [deltainit, vzinit, vyprimeinit, byinit], args=(
    #                 karr[kind], thetaarr[thetaind], B0arr[bind], xesaha_full, xesaha_full_He), method = 'LSODA',
    #                 dense_output=True, atol=1e-9, rtol = 1e-6 )
    #     return sol.sol(np.logspace(np.log10(zstart),np.log10(600),num = 10**4))

    # Active: log(z) integration (same robustness rationale as the Alfven mode above).
    def fssahainteg(karr,thetaarr,B0arr,kind,bind,thetaind, zstart, deltainit, vzinit, vyprimeinit, byinit):
        def rhs_logz(lz, v):
            z = np.exp(lz)
            dv = pars.FSRsahamag_hom(z, v, karr[kind], thetaarr[thetaind], B0arr[bind], xesaha_full, xesaha_full_He)
            return [x*z for x in dv]
        lzstart = np.log(zstart)
        lzgrid  = np.linspace(lzstart, np.log(600), num=10**4)
        sol = solve_ivp(rhs_logz, [lzstart, np.log(600)],
                        [deltainit, vzinit, vyprimeinit, byinit],
                        method='LSODA', dense_output=True, atol=1e-9, rtol=1e-6)
        return sol.sol(lzgrid)
    
    print(time.ctime())

    # z4500 diagnostic variant: integrate from the fixed start z=4500 (instead of
    # zfsarr[input_kind]) for every theta, matching the notebook's Stokes-parameter panel.
    zstart_z4500 = 4500

    resultsalf = np.zeros(( len(thetaarr) ,2,10**4))
    resultsmag = np.zeros(( len(thetaarr),4,10**4 ))
    for thetaind in range(len(thetaarr)):
        resultsalf[thetaind,:,:] = fssahaalfinteg(karr,thetaarr,B0arr,
            input_kind, input_bind, thetaind,zstart_z4500, 0,
                1)
        resultsmag[thetaind,:,:] = fssahainteg(karr,thetaarr,B0arr,
            input_kind, input_bind, thetaind,zstart_z4500,
                                               0, 0,
                                               0, 1 )

    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    # Output base directory (override with PMHD_OUTDIR env var; default = repo data/outputs)
    OUTBASE = Path(os.environ.get("PMHD_OUTDIR", str(PROJECT_ROOT / "data/outputs"))) / "Tfs/z4500"

    # For this run, create subdir per B0
    Bdir = OUTBASE / f"B_{round(1e12*B0arr[input_bind])}pG"
    Bdir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Save the results
    # -----------------------------
    np.save(Bdir / f"FSRsahaalf_k{input_kind}.npy", resultsalf)
    np.save(Bdir / f"FSRsahamag_k{input_kind}.npy", resultsmag)
    
    print(time.ctime())


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))