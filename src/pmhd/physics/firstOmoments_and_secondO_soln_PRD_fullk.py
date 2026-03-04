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

def main(input_nm, input_nmpm1, input_xsteps, input_zchunk):
    karr = k_grid()
    zarr = z_grid()
    
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

    zsize = 5

    

    refshape = np.zeros(( len(karr), input_xsteps, 3))
    refshapem1p1 = np.zeros(( len(karr), input_xsteps, 1))
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    # Output base directory
    OUTBASE = PROJECT_ROOT / "data/outputs"

    f2bardir = OUTBASE/"fbars/f2bars_dict"
    f1barsdir = OUTBASE/"fbars/f1bars"
    f2bardir.mkdir(parents=True, exist_ok=True)
    f1barsdir.mkdir(parents=True, exist_ok=True)

    for i in range(zsize):
        zind = zsize*input_zchunk+i
        output_xs, dx = np.linspace( -1000.0, 1000.0, num=input_xsteps, retstep=True )
        hold = hompsd.psdHR(output_xs,zarr,zind,pars.taus(zarr,xe_full),pars.pab(zarr),pars.feq(zarr,xe_full))
        psd = splrep(output_xs,hold[2])
        print(zarr[zind],time.ctime())

        moments, fphi,fbar = inhomomom.fullk(
            zarr[zind], karr, xe_full, input_nm, input_xsteps, psd )
        p1_moments = inhomomom.fullkp1(
            zarr[zind], karr, xe_full, input_nmpm1, input_xsteps, psd )
        m1_moments = inhomomom.fullkm1(
            zarr[zind], karr, xe_full, input_nmpm1, input_xsteps, psd )
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
                    output_xs, splrep(output_xs, p1_mom_total_dict[f'mom{j}_total'][kind,:]), der = 1)
                m1_mom_total_dict[f'dx_mom{j}_total'][kind,:,0] = splev(
                    output_xs, splrep(output_xs, m1_mom_total_dict[f'mom{j}_total'][kind,:]), der = 1)
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
    parser.add_argument("--nm", type=int, default=9)
    parser.add_argument("--nmpm1", type=int, default=30)
    parser.add_argument("--xsteps", type=int, default=100001)
    parser.add_argument("--zchunk", type=int, default=0)

    args = parser.parse_args()

    main(args.nm, args.nmpm1, args.xsteps, args.zchunk)