import numpy as np
import os
import time
import sys
from pathlib import Path

from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import splrep, splev

from pmhd import cons, pars
from pmhd.physics.theta_pool import theta_map
from pmhd.physics import ode_atol   # PMHD_ATOL_SCALED (addendum 20); off: the production calls
from pmhd.data.tables import tables_dir
from pmhd.data.grids import (
    k_grid,
    theta_grid,
    load_or_generate_z_arrays,
    load_or_generate_B0arr,
)

import pickle

ODE_METHOD = os.environ.get("PMHD_ODE_METHOD", "LSODA")
# PMHD_TOL_SCALE multiplies the solver tolerances (convergence tests), floored as in FSR_TLA_Tfs.py.
TOL_SCALE = float(os.environ.get("PMHD_TOL_SCALE", "1.0"))
def _rtol(base):
    return max(base*TOL_SCALE, 1e-12)
def _atol(base):
    return max(base*TOL_SCALE, 1e-14)


def main(input_bind, input_kind):
    print(time.ctime())

    karr = k_grid()
    thetaarr = theta_grid()

    zcrossarr, zfsarr = load_or_generate_z_arrays()
    B0arr = load_or_generate_B0arr()
    zstartarr = np.append( np.full(np.argwhere(zfsarr<=1900)[0,0],1900), zfsarr[np.argwhere(zfsarr<=1900)[0,0]:] )

    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    # Output base directory (override with PMHD_OUTDIR; Tb subdir following ang_avg/Tb convention)
    OUTBASE = Path(os.environ.get("PMHD_OUTDIR", str(PROJECT_ROOT / "data/outputs"))) / "Tfs"
    Bdir = OUTBASE / "Tb" / f"B_{round(1e12*B0arr[input_bind])}pG"
    Bdir.mkdir(parents=True, exist_ok=True)

    xe_full = pars.xe_full           # hydrogen ionization (recombination, dxe, Acon/B1con/B2con)
    xe_full_He = pars.xe_full_He     # He-inclusive free electrons (drag, pressure, Compton heating)
    xesaha_full = pars.xesaha_full

    def Acon(z):
        return (1 - xe_full(z)) *  pars.nh(z) *  pars.sigmaa(z)

    def B1con(z):
        return xe_full(z)**2 * pars.nh(z) *  (pars.alpha1s(z) - pars.alphab(z))

    def B2con(z):
        return xe_full(z) * pars.nh(z) *  (pars.alpha1s(z) - pars.alphab(z)) * (2-xe_full(z))/(1-xe_full(z))

    with open(tables_dir()/'abarinterpmaster.pkl', 'rb') as f:
        abarinterpmaster = pickle.load(f)

    with open(tables_dir()/'bbarinterpmaster.pkl', 'rb') as f:
        bbarinterpmaster = pickle.load(f)

    with open(tables_dir()/'cbarinterpmaster.pkl', 'rb') as f:
        cbarinterpmaster = pickle.load(f)

    def abarinterpfunc(z,kind):
        return splev(z, abarinterpmaster[kind])

    def bbarinterpfunc(z,kind):
        return splev(z, bbarinterpmaster[kind])

    def cbarinterpfunc(z,kind):
        return splev(z, cbarinterpmaster[kind])

    ## Solve for homogeneous baryon temperature (He-inclusive Compton heating)
    sol_Tb = odeint(pars.RHSTbhom, pars.Tcmb(1900), np.arange(1900,600,-0.01), args=(xe_full_He,))
    zhold = np.arange(1900,600,-0.01)
    Tbhomspl = splrep(zhold[::-1], sol_Tb.flatten()[::-1])

    def Tbhom(z):
        return splev(z, Tbhomspl)

    zend = 600

    fssahaalf = np.load(Bdir.parent.parent / f"B_{round(1e12*B0arr[input_bind])}pG" / f"FSRsahaalf_k{input_kind}.npy")
    fssahamag = np.load(Bdir.parent.parent / f"B_{round(1e12*B0arr[input_bind])}pG" / f"FSRsahamag_k{input_kind}.npy")

    print(time.ctime())

    resultsalf = np.zeros(( len(thetaarr) , 2, 10**4))
    resultsmag = np.zeros(( len(thetaarr), 6, 10**4 ))
    def _one_theta(thetaind):   # independent per angle: parallel with PMHD_NPROC > 1, bit-identical to the serial loop
        zsplhold = np.logspace(np.log10(zfsarr[input_kind]),np.log10(zend),num = 10**4)[::-1]
        Phixinit = splev( zstartarr[input_kind], splrep( zsplhold, fssahaalf[thetaind,0,::-1] ) )
        bxinit = splev( zstartarr[input_kind], splrep( zsplhold, fssahaalf[thetaind,1,::-1] ) )

        deltainit = splev( zstartarr[input_kind], splrep( zsplhold, fssahamag[thetaind,0,::-1] ) )
        Thetainit = splev( zstartarr[input_kind], splrep( zsplhold, fssahamag[thetaind,1,::-1] ) )
        Phiyinit = splev( zstartarr[input_kind], splrep( zsplhold, fssahamag[thetaind,2,::-1] ) )
        byinit = splev( zstartarr[input_kind], splrep( zsplhold, fssahamag[thetaind,3,::-1] ) )

        xeinit = -xesaha_full(zstartarr[input_kind]) * (
                (1-xesaha_full(zstartarr[input_kind]))/(2-xesaha_full(zstartarr[input_kind])))*deltainit

        # Tb^(1) -> 0 at high z: tight Compton coupling drives temperature perturbation to zero
        Tbinit = 0.0

        mag_row = ode_atol.solve_on_grid( pars.FSRTLAmag_Tb, [zstartarr[input_kind], zend],
                np.array([deltainit, Thetainit, Phiyinit, byinit, xeinit, Tbinit]),
                np.logspace(np.log10(zstartarr[input_kind]),np.log10(zend), num = 10**4),
                args=(
                    karr[input_kind], input_kind, thetaarr[thetaind], B0arr[input_bind],
                    xe_full, xe_full_He, abarinterpfunc, bbarinterpfunc, cbarinterpfunc,
                    Tbhom, Acon, B1con, B2con
                ), method=ODE_METHOD, atol=_atol(1e-9), rtol=_rtol(1e-6) )

        # scipy's default tolerances (1e-3, 1e-6) in production; with PMHD_ATOL_SCALED the magnetosonic tolerances
        # through _rtol/_atol, since Phi_x can be ~1e-12 of b_x (addenda 20, 21)
        alf_tol = dict(rtol=_rtol(1e-6), atol=_atol(1e-9)) if ode_atol.SCALED else {}
        alf_row = ode_atol.solve_on_grid( pars.FSRTLAalf, [zstartarr[input_kind], zend], np.array([Phixinit, bxinit]),
                np.logspace(np.log10(zstartarr[input_kind]),np.log10(zend), num = 10**4), args=(
                karr[input_kind], thetaarr[thetaind], B0arr[input_bind],xe_full_He), method = ODE_METHOD, **alf_tol )
        return alf_row, mag_row
    for thetaind, (a, m) in enumerate(theta_map(_one_theta, len(thetaarr))):
        resultsalf[thetaind,:,:] = a; resultsmag[thetaind,:,:] = m

    # -----------------------------
    # Save the results
    # -----------------------------
    np.save(Bdir / f"FSRTLAalf_Tb_k{input_kind}.npy", resultsalf)
    np.save(Bdir / f"FSRTLAmag_Tb_k{input_kind}.npy", resultsmag)

    print(time.ctime())


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
