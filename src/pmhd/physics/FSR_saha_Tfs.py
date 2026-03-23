import numpy as np
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

    xesaha_full = pars.xesaha_full

    def fssahaalfinteg(karr,thetaarr,B0arr,kind,bind,thetaind, zstart, vxinit, bxinit):
        sol = solve_ivp(pars.FSRsahaalf, [zstart, 600], [vxinit, bxinit], args=(
                    karr[kind], thetaarr[thetaind], B0arr[bind],xesaha_full), method = 'LSODA',
                    dense_output=True, atol=1e-9, rtol = 1e-6 )

        return sol.sol(np.logspace(np.log10(zstart),np.log10(600),num = 10**4))

    # If the Alfvén integration gets stuck (observed at kind=3, bind=56/57, thetaind=8),
    # switch to log(z) integration below. The variable change d/d(log z) = z * d/dz is
    # mathematically equivalent but avoids a pathological LSODA step-size failure at theta=pi/4.
    # def fssahaalfinteg(karr,thetaarr,B0arr,kind,bind,thetaind, zstart, vxinit, bxinit):
    #     def rhs_logz(lz, v):
    #         z = np.exp(lz)
    #         dv = pars.FSRsahaalf(z, v, karr[kind], thetaarr[thetaind], B0arr[bind], xesaha_full)
    #         return [dv[0]*z, dv[1]*z]
    #     lzstart = np.log(zstart)
    #     lzgrid  = np.linspace(lzstart, np.log(600), num=10**4)
    #     sol = solve_ivp(rhs_logz, [lzstart, np.log(600)], [vxinit, bxinit],
    #                     method='LSODA', dense_output=True, atol=1e-9, rtol=1e-6)
    #     return sol.sol(lzgrid)

    def fssahainteg(karr,thetaarr,B0arr,kind,bind,thetaind, zstart, deltainit, vzinit, vyprimeinit, byinit):
        sol = solve_ivp(pars.FSRsahamag, [zstart, 600], [deltainit, vzinit, vyprimeinit, byinit], args=(
                    karr[kind], thetaarr[thetaind], B0arr[bind],xesaha_full), method = 'LSODA',
                    dense_output=True, atol=1e-9, rtol = 1e-6 )

        return sol.sol(np.logspace(np.log10(zstart),np.log10(600),num = 10**4))

    # If the magnetosonic integration gets stuck, the same log(z) fix can be applied:
    # def fssahainteg(karr,thetaarr,B0arr,kind,bind,thetaind, zstart, deltainit, vzinit, vyprimeinit, byinit):
    #     def rhs_logz(lz, v):
    #         z = np.exp(lz)
    #         dv = pars.FSRsahamag(z, v, karr[kind], thetaarr[thetaind], B0arr[bind], xesaha_full)
    #         return [x*z for x in dv]
    #     lzstart = np.log(zstart)
    #     lzgrid  = np.linspace(lzstart, np.log(600), num=10**4)
    #     sol = solve_ivp(rhs_logz, [lzstart, np.log(600)],
    #                     [deltainit, vzinit, vyprimeinit, byinit],
    #                     method='LSODA', dense_output=True, atol=1e-9, rtol=1e-6)
    #     return sol.sol(lzgrid)
    
    print(time.ctime())

    resultsalf = np.zeros(( len(thetaarr) ,2,10**4))
    resultsmag = np.zeros(( len(thetaarr),4,10**4 ))
    for thetaind in range(len(thetaarr)):
        resultsalf[thetaind,:,:] = fssahaalfinteg(karr,thetaarr,B0arr,
            input_kind, input_bind, thetaind,zfsarr[input_kind], 0, 
                1)
        resultsmag[thetaind,:,:] = fssahainteg(karr,thetaarr,B0arr,
            input_kind, input_bind, thetaind,zfsarr[input_kind], 
                                               0, 0,
                                               0, 1 )
        
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    # Output base directory
    OUTBASE = PROJECT_ROOT / "data/outputs/Tfs"

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