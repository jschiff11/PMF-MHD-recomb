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

    zstart = 2300
    xepeebs = pars.odeint(pars.RHSsob, pars.sahataylor(zstart) , np.arange(zstart,200,-1))
    xesaha = pars.sahataylor(np.arange(3e4,zstart,-1) )
    xesaha2 = pars.sahataylor(np.arange(3e5,2660,-1) )
    xesaha3 = pars.sahapol(np.arange(2660,200,-1))

    ## Compute ionization fraction as given by TLA and by saha. xe_full(z) assumes saha for z > 2300 then TLA after, xesaha_full assumes saha for all redshifts

    xe_hold = splrep(
            np.flip(np.concatenate([np.linspace(1e11,3.1e4,10000) , np.arange(3e4,200,-1) ])),
            np.flip(np.concatenate([np.ones(10000),xesaha,xepeebs[:,0]]))
        )

    xesaha_hold = splrep(
            np.flip(np.concatenate([np.linspace(1e11,3.1e5,10000) , np.arange(3e5,200,-1) ])),
            np.flip(np.concatenate([np.ones(10000),xesaha2,xesaha3]))
        )

    def xe_full(z):
        return splev(z, xe_hold)
    def xesaha_full(z):
        return splev(z, xesaha_hold)


    def fssahaalfinteg(karr,thetaarr,B0arr,kind,bind,thetaind, zstart, vxinit, bxinit):
        sol = solve_ivp(pars.FSRsahaalf, [zstart, 600], [vxinit, bxinit], args=(
                    karr[kind], thetaarr[thetaind], B0arr[bind],xesaha_full), method = 'LSODA',
                    dense_output=True, atol=1e-9, rtol = 1e-6 )    
        
        return sol.sol(np.logspace(np.log10(zstart),np.log10(600),num = 10**4))

    def fssahainteg(karr,thetaarr,B0arr,kind,bind,thetaind, zstart, deltainit, vzinit, vyprimeinit, byinit):
        sol = solve_ivp(pars.FSRsahamag, [zstart, 600], [deltainit, vzinit, vyprimeinit, byinit], args=(
                    karr[kind], thetaarr[thetaind], B0arr[bind],xesaha_full), method = 'LSODA',
                    dense_output=True, atol=1e-9, rtol = 1e-6 )    
        
        return sol.sol(np.logspace(np.log10(zstart),np.log10(600),num = 10**4))
    
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