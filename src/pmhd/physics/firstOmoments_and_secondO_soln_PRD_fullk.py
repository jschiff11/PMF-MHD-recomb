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
    load_or_generate_z_arrays,
    load_or_generate_B0arr,
)

from pmhd.physics import inhomo_moments as inhomomom
from pmhd.physics import hompsd

def main(input_nm, input_nmpm1, input_xsteps, input_zchunk, outbase=None,
         zsize=10, closure=True, doppler=True):
    """Moment tables for z-indices [zsize*zchunk, zsize*(zchunk+1)).

    closure : non-reflecting closure on the m=+-1 towers (m=0 always has it); False = hard truncation.
    doppler : first-order Doppler rate-factor ell=1 source in the Theta (B') and D^pm' bases.
    closure=False, doppler=False, nm=9, nmpm1=30 reproduces the pre-2026-09 production tables.
    Extra keys for the rest-frame-monopole (Doppler/aberration) second-order terms, per column:
      Qdop_mom1_total : line-average of D_nu^-1[ -tau p_sc (phi m1 + (1/2) d/dx(phi dm1/dx)) - (1/2) tau phi m1 ]
                        (Eqs. 420 + 416 applied to the ell=1 moment)
      fbar_mom1_total : int phi m1 dx   (rate term, Eq. 421)
    """
    karr = k_grid()
    zarr = z_grid()

    xe_full = pars.xe_full




    refshape = np.zeros(( len(karr), input_xsteps, 3))
    refshapem1p1 = np.zeros(( len(karr), input_xsteps, 1))
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    # Output base directory (override with --outbase for a non-default run,
    # e.g. to avoid touching the canonical data/outputs/fbars/ cache)
    OUTBASE = Path(outbase) if outbase is not None else PROJECT_ROOT / "data/outputs"

    f2bardir = OUTBASE/"fbars/f2bars_dict"
    f1barsdir = OUTBASE/"fbars/f1bars"
    f2bardir.mkdir(parents=True, exist_ok=True)
    f1barsdir.mkdir(parents=True, exist_ok=True)

    for i in range(zsize):
        zind = zsize*input_zchunk+i
        if zind >= len(zarr):
            break
        output_xs, dx = np.linspace( -1000.0, 1000.0, num=input_xsteps, retstep=True )
        hold = hompsd.psdHR(output_xs,zarr,zind,pars.taus(zarr,xe_full),pars.pab(zarr),pars.feq(zarr,xe_full))
        psd = splrep(output_xs,hold[2])
        print(zarr[zind],time.ctime())

        moments, fphi,fbar = inhomomom.fullk(
            zarr[zind], karr, xe_full, input_nm, input_xsteps, psd, doppler=doppler )
        p1_moments = inhomomom.fullkp1(
            zarr[zind], karr, xe_full, input_nmpm1, input_xsteps, psd, closure=closure, doppler=doppler )
        m1_moments = inhomomom.fullkm1(
            zarr[zind], karr, xe_full, input_nmpm1, input_xsteps, psd, closure=closure, doppler=doppler )
        print(moments.shape)
        mom_total_dict = {}
        m1_mom_total_dict = {}  
        p1_mom_total_dict = {} 
        for j in range(3):
            mom_total_dict[f'mom{j}_total'] = moments[:,j::input_nm,:]  
            mom_total_dict[f'dx_mom{j}_total'] = np.zeros(refshape.shape)
            p1_mom_total_dict[f'mom{j}_total'] = p1_moments[:,j::input_nmpm1]  
            p1_mom_total_dict[f'dx_mom{j}_total'] = np.zeros(refshapem1p1.shape)
            m1_mom_total_dict[f'mom{j}_total'] = m1_moments[:,j::input_nmpm1]  
            m1_mom_total_dict[f'dx_mom{j}_total'] = np.zeros(refshapem1p1.shape)

        mom_total_dict[f'dx_phi_dx_mom0_total'] = np.zeros(refshape.shape)
        p1_mom_total_dict[f'dx_phi_dx_mom0_total'] = np.zeros(refshapem1p1.shape)
        m1_mom_total_dict[f'dx_phi_dx_mom0_total'] = np.zeros(refshapem1p1.shape)

        mom_total_dict[f'phi_mom0_total'] = np.zeros(refshape.shape)
        p1_mom_total_dict[f'phi_mom0_total'] = np.zeros(refshapem1p1.shape)
        m1_mom_total_dict[f'phi_mom0_total'] = np.zeros(refshapem1p1.shape)
        
        
        for kind in range(len(karr)):
            for j in range(3):
                p1_mom_total_dict[f'dx_mom{j}_total'][kind,:,0] = splev(
                    output_xs, splrep(output_xs, p1_mom_total_dict[f'mom{j}_total'][kind,:,0]), der = 1)   # 1-D y for splrep (third audit, finding 2)
                m1_mom_total_dict[f'dx_mom{j}_total'][kind,:,0] = splev(
                    output_xs, splrep(output_xs, m1_mom_total_dict[f'mom{j}_total'][kind,:,0]), der = 1)
                for s in range(3):
                    mom_total_dict[f'dx_mom{j}_total'][kind,:,s] = splev(
                        output_xs, splrep(output_xs, mom_total_dict[f'mom{j}_total'][kind,:,s]), der = 1)

                    mom_total_dict[f'dx_phi_dx_mom0_total'][kind,:,s] = splev(
                        output_xs, splrep(output_xs, pars.voigt(output_xs,zarr[zind])*mom_total_dict[f'dx_mom{0}_total'][kind,:,s]), der = 1) 

                    mom_total_dict[f'phi_mom0_total'][kind,:,s] = pars.voigt(output_xs,zarr[zind])*mom_total_dict[f'mom{0}_total'][kind,:,s]
                    
            
            p1_mom_total_dict[f'dx_phi_dx_mom0_total'][kind,:,0] = splev(
                    output_xs, splrep(output_xs, pars.voigt(output_xs,zarr[zind]) * p1_mom_total_dict[f'dx_mom{0}_total'][kind,:,0]), der = 1)
            p1_mom_total_dict[f'phi_mom0_total'][kind,:,0] = pars.voigt(output_xs,zarr[zind]) * p1_mom_total_dict[f'mom{0}_total'][kind,:,0]

            m1_mom_total_dict[f'dx_phi_dx_mom0_total'][kind,:,0] = splev(
                    output_xs, splrep(output_xs, pars.voigt(output_xs,zarr[zind]) * m1_mom_total_dict[f'dx_mom{0}_total'][kind,:,0]), der = 1) 
            m1_mom_total_dict[f'phi_mom0_total'][kind,:,0] = pars.voigt(output_xs,zarr[zind]) * m1_mom_total_dict[f'mom{0}_total'][kind,:,0]

        # --- rest-frame-monopole (Doppler/aberration) keys: Eqs. (420)+(416) and (421) ---
        phi_x = pars.voigt(output_xs, zarr[zind])
        tauS_z = pars.taus(zarr[zind], xe_full)
        psc_z = 1 - pars.pab(zarr[zind])
        def _dop_op(a1):
            d_a1 = splev(output_xs, splrep(output_xs, a1), der=1)
            return (-tauS_z*psc_z*(phi_x*a1 + 0.5*splev(output_xs, splrep(output_xs, phi_x*d_a1), der=1))
                    - 0.5*tauS_z*phi_x*a1)
        mom_total_dict['Qdop_mom1_total'] = np.zeros(refshape.shape)
        p1_mom_total_dict['Qdop_mom1_total'] = np.zeros(refshapem1p1.shape)
        m1_mom_total_dict['Qdop_mom1_total'] = np.zeros(refshapem1p1.shape)
        fbar_mom1 = {'m0': np.zeros((len(karr), 3)), 'p1': np.zeros((len(karr), 1)), 'm1': np.zeros((len(karr), 1))}
        for kind in range(len(karr)):
            for s in range(3):
                a1 = mom_total_dict['mom1_total'][kind,:,s]
                mom_total_dict['Qdop_mom1_total'][kind,:,s] = _dop_op(a1)
                fbar_mom1['m0'][kind,s] = splint(output_xs[0], output_xs[-1], splrep(output_xs, phi_x*a1))
            for t, dct in (('p1', p1_mom_total_dict), ('m1', m1_mom_total_dict)):
                a1 = dct['mom1_total'][kind,:,0]
                dct['Qdop_mom1_total'][kind,:,0] = _dop_op(a1)
                fbar_mom1[t][kind,0] = splint(output_xs[0], output_xs[-1], splrep(output_xs, phi_x*a1))

        f2moms_dict = {}  
        f2bars_dict = {}  
        f2moms_m1_dict = {}  
        f2bars_m1_dict = {}  
        f2moms_p1_dict = {}  
        f2bars_p1_dict = {}
        for key, arr in mom_total_dict.items():
            f2moms_dict[key], f2bars_dict[key] = inhomomom.boltfullk2(zarr[zind], karr, arr, 3, xe_full,input_xsteps)
        for key, arr in m1_mom_total_dict.items():
            f2moms_m1_dict[key], f2bars_m1_dict[key] = inhomomom.boltfullk2(zarr[zind], karr, arr, 1, xe_full,input_xsteps)
        for key, arr in p1_mom_total_dict.items():
            f2moms_p1_dict[key], f2bars_p1_dict[key] = inhomomom.boltfullk2(zarr[zind], karr, arr, 1, xe_full,input_xsteps)
        f2bars_dict['fbar_mom1_total'] = fbar_mom1['m0']
        f2bars_p1_dict['fbar_mom1_total'] = fbar_mom1['p1']
        f2bars_m1_dict['fbar_mom1_total'] = fbar_mom1['m1']
        
        mom_total_dict[f'taus_phi'] = np.zeros(( 1,len(output_xs), 1))
        mom_total_dict[f'df/dnu'] = np.zeros(( 1,len(output_xs), 1))
        mom_total_dict[f'taus_phi'][0,:,0] = -pars.taus(zarr[zind],xe_full)*pars.voigt(output_xs,zarr[zind])
        mom_total_dict[f'df/dnu'][0,:,0] = splev( output_xs, psd, der=1 )
        f2moms_dict[f'taus_phi'], f2bars_dict[f'taus_phi'] = inhomomom.boltz2fullz(zarr, zsize*input_zchunk+i, zsize*input_zchunk+i+1, mom_total_dict[f'taus_phi'], 1, xe_full,input_xsteps)
        f2moms_dict[f'df/dnu'], f2bars_dict[f'df/dnu'] = inhomomom.boltz2fullz(zarr, zsize*input_zchunk+i, zsize*input_zchunk+i+1, mom_total_dict[f'df/dnu'], 1, xe_full,input_xsteps)

        # Save to file
        with open(f2bardir/f'f2bars_m0_dict_zch_{input_zchunk*zsize + i}.pkl', 'wb') as f:
            pickle.dump(f2bars_dict, f) 
        with open(f2bardir/f'f2bars_m1_dict_zch_{input_zchunk*zsize + i}.pkl', 'wb') as f:
            pickle.dump(f2bars_m1_dict, f)
        with open(f2bardir/f'f2bars_p1_dict_zch_{input_zchunk*zsize + i}.pkl', 'wb') as f:
            pickle.dump(f2bars_p1_dict, f)

        

        np.save(f1barsdir/f'abar_zchunk{input_zchunk*zsize + i}.npy', fbar[:,0])
        np.save(f1barsdir/f'bbar_zchunk{input_zchunk*zsize + i}.npy', fbar[:,1])
        np.save(f1barsdir/f'cbar_zchunk{input_zchunk*zsize + i}.npy', fbar[:,2])
        
        print(time.ctime())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # production since 2026-09: non-reflecting closure at ell_max = 14 in all three towers
    parser.add_argument("--nm", type=int, default=15)
    parser.add_argument("--nmpm1", type=int, default=15)
    parser.add_argument("--zsize", type=int, default=10, help="z-indices per chunk (130 chunks for 1300 z)")
    parser.add_argument("--no-closure", action="store_true", help="hard truncation on m=+-1 (pre-2026-09)")
    parser.add_argument("--no-doppler", action="store_true", help="omit the first-order Doppler ell=1 source")
    parser.add_argument("--xsteps", type=int, default=100001)
    parser.add_argument("--zchunk", type=int, default=0)
    parser.add_argument("--outbase", type=str, default=None,
                         help="Override the data/outputs base directory (default: <repo>/data/outputs)")

    args = parser.parse_args()

    main(args.nm, args.nmpm1, args.xsteps, args.zchunk, outbase=args.outbase,
         zsize=args.zsize, closure=not args.no_closure, doppler=not args.no_doppler)