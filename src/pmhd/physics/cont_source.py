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
    k_grid, z_grid, eps_grid,
    load_or_generate_B0arr,
)

def main(input_bind, input_epsind):
    print(time.ctime())
    karr = k_grid()
    B0arr = load_or_generate_B0arr()
    zarr = z_grid()

    epsarr = eps_grid()

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
    
    ## For conveneince make arrays of of taus, deltah, and H for all values of z under consideration 
    tausall = pars.taus(zarr,xe_full)
    Deltah = pars.Deltah(zarr)
    H = pars.H(zarr)

    def Acon(z):
        return (1 - xe_full(z)) *  pars.nh(z) *  pars.sigmaa(z)

    def B1con(z):
        return xe_full(z)**2 * pars.nh(z) *  pars.alpha1s(z)

    def B2con(z): 
        return xe_full(z) * pars.nh(z) *  pars.alpha1s(z) * (2-xe_full(z))/(1-xe_full(z))

    def kappa(k,z):
        return 1 - Acon(z)/(k*(1+z)) * np.arctan((k*(1+z))/Acon(z))    
    
    
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    OUTBASE = PROJECT_ROOT / "data/outputs"
    ang_avg_TLA = OUTBASE/f"ang_avg/TLA/B_{round(1e12*B0arr[input_bind])}pG"
    
    ang_avg_dict_master = {}
    for k in [
        'bxbxbar', 'bybybar', 'PhixPhixbar',
        'PhiyPhiybar', 'ThetaThetabar', 'PhiyThetabar',
        'deltamThetabar', 'deltamdeltambar', 'xexebar', 'xedeltambar', 'xeThetabar',
        'PhixPhixprimebar', 'PhiyPhiyprimebar', 'ThetaThetaprimebar'
    ]:    
        ang_avg_dict_master[k] = np.zeros( ( len(karr), len(zarr)) )

    for kind in range(len(karr)):
        with open(ang_avg_TLA/f'ang_avg_k{kind}.pkl', 'rb') as f:
            ang_avg_dict = pickle.load(f)      
        for k in [
        'bxbxbar', 'bybybar', 'PhixPhixbar',
        'PhiyPhiybar', 'ThetaThetabar', 'PhiyThetabar',
        'deltamThetabar', 'deltamdeltambar', 'xexebar', 'xedeltambar', 'xeThetabar',
        'PhixPhixprimebar', 'PhiyPhiyprimebar', 'ThetaThetaprimebar'
            ]:    
            ang_avg_dict_master[k][kind] = ang_avg_dict[k]

    xe2contsource = np.zeros( (len(zarr) ) )
    for z in range(len(zarr)):            
        ## Compute all relevant rms and cross correlations needed for second order perturbed ionization fraction 
        
        xe2contsource[z] = splint(
             karr[-1], karr[0], (splrep(
                np.flip(karr), np.flip(
                    karr**(epsarr[input_epsind] - 1) * kappa(karr,zarr[z]) * ( 
                        B1con(zarr[z])*ang_avg_dict_master['deltamdeltambar'][:,z] + B2con(zarr[z])*ang_avg_dict_master['xedeltambar'][:,z] ) 
            )))
        )
    
    source_fncs = OUTBASE/"source_fncs"
    source_fncs.mkdir(parents=True, exist_ok=True)

    # Save results
    np.save(source_fncs/f'xe2contsource_B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}.npy', xe2contsource)   

if __name__=='__main__':
    print(time.ctime())

    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", type=int, default=40)
    parser.add_argument("--epsind", type=int, default=9)

    args = parser.parse_args()

    main(args.bind, args.epsind)
    

print(time.ctime())