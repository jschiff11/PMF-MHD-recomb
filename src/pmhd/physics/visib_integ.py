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


def main(input_bind, input_epsind):
    print(time.ctime())
    karr = k_grid()
    B0arr = load_or_generate_B0arr()
    zarr = z_grid()

    epsarr = eps_grid()
    Lambda = 1e3*cons.mpc # set the cutoff scale to 1 Gpc

    xe_full = pars.xe_full
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    OUTBASE = PROJECT_ROOT / "data/outputs"
    tau0arr = np.load(PROJECT_ROOT/'src/pmhd/data/pre_stored_data/tau0arr.npy')

    xe2direc = OUTBASE/"xe2"

    xe2 = np.load(xe2direc/f'xe2_B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}.npy')

    cross_corrsave = OUTBASE/"cross_corr"
    dtaurms = (abs(epsarr[input_epsind])/4 ) * (Lambda/(2*np.pi) )**((epsarr[input_epsind])) * np.load(cross_corrsave/f'dtaurms_B_{round(1e12*B0arr[input_bind])}pG.npy')[:,input_epsind]

    dtaudtaudotcross = (abs(epsarr[input_epsind])/4 ) * (Lambda/(2*np.pi) )**((epsarr[input_epsind])) * np.load(cross_corrsave/f'dtaudtaudotcross_B_{round(1e12*B0arr[input_bind])}pG.npy')[:,input_epsind]

    dtau2 = np.zeros(len(zarr))
    z_min = zarr[-1]
    for z in range(len(zarr)):
        dtau2[z] = splint(z_min,zarr[z],splrep(zarr[::-1],
                                                (cons.c*pars.nh(zarr)*xe2*cons.sigmat/pars.H(zarr)/(1+zarr))[::-1] )
                         )
    
    # Ensure directory exists
    visibdirec = OUTBASE/"visib"
    visibdirec.mkdir(parents=True, exist_ok=True)    
    visibprefactor = (1 + (xe2/xe_full(zarr)) - dtaudtaudotcross )*np.exp(-dtau2 + dtaurms/2)
    np.save(visibdirec/f'visib_B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}.npy', visibprefactor)
    np.save(visibdirec/f'dtau2_B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}.npy', dtau2)

if __name__=='__main__':
    print(time.ctime())

    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", type=int, default=40)
    parser.add_argument("--epsind", type=int, default=9)

    args = parser.parse_args()

    main(args.bind, args.epsind)

print(time.ctime())


    