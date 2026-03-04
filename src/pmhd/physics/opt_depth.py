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


## Initialize redshift, spectral index, wavenumber, magnetic field strength arrays 


def main(input_bind):
    print(time.ctime())
    karr = k_grid()
    B0arr = load_or_generate_B0arr()
    zarr = z_grid()

    epsarr = eps_grid()
    Lambda = 1e3*cons.mpc # set the cutoff scale to 1 Gpc

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
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    OUTBASE = PROJECT_ROOT / "data/outputs"

    ## Set up all source terms and cross correlations that we wish to compute for all values of z, x, B0, \epsilon
    dtaurms = np.zeros( (len(zarr), len(epsarr) ) )
    dtaudtaudotcross = np.zeros( (len(zarr), len(epsarr) ) )
    
    
    dtaudtaubar = np.zeros( ( len(karr), len(zarr)) )
    dtaudtaudotbar = np.zeros( ( len(karr), len(zarr)) )
    
    ang_avg_TLA = OUTBASE/f"ang_avg/TLA/B_{round(1e12*B0arr[input_bind])}pG"
    ang_avg_TLA.mkdir(parents=True, exist_ok=True)
        

    for kind in range(len(karr)):
        dtaudtaubar[kind,:] = np.load(ang_avg_TLA/f'dtaudtaubar_k{kind}.npy')
        dtaudtaudotbar[kind,:] = np.load(ang_avg_TLA/f'dtaudtaudotbar_k{kind}.npy')
   
    for epsind in range(len(epsarr)):
        for z in range(len(zarr)):  
            ## Compute all relevant rms and cross correlations needed for second order perturbed ionization fraction 
            
            dtaurms[z,epsind] = splint(
                 karr[-1], karr[0], (splrep(
                    np.flip(karr), np.flip(
                        karr**(epsarr[epsind] - 1) * (
                        dtaudtaubar[:,z] ) 
                )))
            )
           
            dtaudtaudotcross[z,epsind] = splint(
                 karr[-1], karr[0], (splrep(
                    np.flip(karr), np.flip(
                        karr**(epsarr[epsind] - 1) * (
                        dtaudtaudotbar[:,z] ) 
                )))
            )

    cross_corrsave = OUTBASE/"cross_corr"
    # Save results
    np.save(cross_corrsave/f'dtaurms_B_{round(1e12*B0arr[input_bind])}pG.npy', dtaurms)
    np.save(cross_corrsave/f'dtaudtaudotcross_B_{round(1e12*B0arr[input_bind])}pG.npy', dtaudtaudotcross)

if __name__=='__main__':
    print(time.ctime())

    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", type=int, default=40)

    args = parser.parse_args()

    main(args.bind)
    
        

    