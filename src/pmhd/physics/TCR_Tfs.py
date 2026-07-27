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


# Multiply all solver tolerances by this factor (e.g. 1e-2 for a 100x-tighter
# convergence test); default 1.0 leaves the production tolerances unchanged.
# Floored at rtol=1e-12 / atol=1e-14 (LSODA feasibility limit).
TOL_SCALE = float(os.environ.get("PMHD_TOL_SCALE", "1.0"))

def _rtol(base):
    return max(base * TOL_SCALE, 1e-12)

def _atol(base):
    return max(base * TOL_SCALE, 1e-14)

ODE_METHOD = os.environ.get("PMHD_ODE_METHOD", "LSODA")


def main(input_bind, input_kind):
    print(time.ctime())
    karr = k_grid()
    thetaarr = theta_grid()

    # zcrossarr is already floored at neutrino decoupling (T=1 MeV) for modes
    # that would otherwise cross the horizon earlier -- see grids.py.
    zcrossarr, zfsarr = load_or_generate_z_arrays()
    B0arr = load_or_generate_B0arr()

    # He-recombination variant: hydrogen 3LA + helium electron contribution
    # (enters TCmag/TCalf only via the Thomson drag eta).
    xe_full = pars.xe_full_He

    def TCRalfinteg(karr,thetaarr,B0arr,kind,bind,thetaind, zstart,zend):
        sol = solve_ivp(pars.TCalf, [zstart, zend], [0, 1], args=(
                    karr[kind], thetaarr[thetaind], B0arr[bind], xe_full), method = ODE_METHOD,
                    dense_output=True, atol=_atol(1e-9), rtol = _rtol(1e-7) )

        return sol.sol(np.logspace(np.log10(zstart),np.log10(zend),num = 10**4))

    def TCRmaginteg(karr,thetaarr,B0arr,kind,bind,thetaind, zstart,zend):
        sol = solve_ivp(pars.TCmag, [zstart, zend], [0, 0, 0, 1], args=(
                    karr[kind], thetaarr[thetaind], B0arr[bind], xe_full), method = ODE_METHOD,
                    dense_output=True, atol=_atol(1e-10), rtol = _rtol(1e-9) )

        return sol.sol(np.logspace(np.log10(zstart),np.log10(zend),num = 10**4))

    print(time.ctime())

    resultsalf = np.zeros(( len(thetaarr), 2, 10**4))
    resultsmag = np.zeros(( len(thetaarr), 4, 10**4 ))
    for thetaind in range(len(thetaarr)):
        resultsalf[thetaind,:,:] = TCRalfinteg(karr,thetaarr,B0arr,
            input_kind, input_bind, thetaind, zcrossarr[input_kind],zfsarr[input_kind])
        resultsmag[thetaind,:,:] = TCRmaginteg(karr,thetaarr,B0arr,
            input_kind, input_bind, thetaind, zcrossarr[input_kind],zfsarr[input_kind])
    

    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    # Output base directory (override with PMHD_OUTDIR env var; default = repo data/outputs)
    OUTBASE = Path(os.environ.get("PMHD_OUTDIR", str(PROJECT_ROOT / "data/outputs"))) / "Tfs"

    # For this run, create subdir per B0
    Bdir = OUTBASE / f"B_{round(1e12*B0arr[input_bind])}pG"
    Bdir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Save the results
    # -----------------------------
    np.save(Bdir / f"TCRalf_k{input_kind}.npy", resultsalf)
    np.save(Bdir / f"TCRmag_k{input_kind}.npy", resultsmag)

    print(time.ctime())

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))