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
from pmhd.physics.source_assembly import assemble_cont_at_z, splint_reducer
from pmhd.data.grids import (
    k_grid, z_grid, eps_grid,
    load_or_generate_B0arr,
)

def main(input_bind, input_epsind, tb=False):
    print(time.ctime())
    karr = k_grid()
    B0arr = load_or_generate_B0arr()
    zarr = z_grid()

    epsarr = eps_grid()

    xe_full = pars.xe_full

    ## For conveneince make arrays of of taus, deltah, and H for all values of z under consideration
    tausall = pars.taus(zarr,xe_full)
    Deltah = pars.Deltah(zarr)
    H = pars.H(zarr)

    def Acon(z):
        return (1 - xe_full(z)) *  pars.nh(z) *  pars.sigmaa(z)

    def B1con(z):
        return xe_full(z)**2 * pars.nh(z) *  (pars.alpha1s(z) - pars.alphab(z))

    def B2con(z):
        return xe_full(z) * pars.nh(z) *  (pars.alpha1s(z) - pars.alphab(z)) * (2-xe_full(z))/(1-xe_full(z))

    def kappa(k,z):
        return 1 - Acon(z)/(k*(1+z)) * np.arctan((k*(1+z))/Acon(z))    
    
    
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    OUTBASE = Path(os.environ.get("PMHD_OUTDIR", str(PROJECT_ROOT / "data/outputs")))
    # --tb: build the continuum source from the T_b-coupled angle averages (Eq. B53 with the T_b transfer functions)
    ang_avg_TLA = OUTBASE/(f"ang_avg/Tb/B_{round(1e12*B0arr[input_bind])}pG" if tb else f"ang_avg/TLA/B_{round(1e12*B0arr[input_bind])}pG")
    _keys = ["deltamdeltambar", "xedeltambar"] if tb else None
    
    ang_avg_dict_master = {}
    for k in [
        'bxbxbar', 'bybybar', 'PhixPhixbar',
        'PhiyPhiybar', 'ThetaThetabar', 'PhiyThetabar',
        'deltamThetabar', 'deltamdeltambar', 'xexebar', 'xedeltambar', 'xeThetabar',
        'PhixPhixprimebar', 'PhiyPhiyprimebar', 'ThetaThetaprimebar'
    ]:    
        ang_avg_dict_master[k] = np.zeros( ( len(karr), len(zarr)) )

    for kind in range(len(karr)):
        with open(ang_avg_TLA/(f'ang_avg_k{kind}_Tb.pkl' if tb else f'ang_avg_k{kind}.pkl'), 'rb') as f:
            ang_avg_dict = pickle.load(f)      
        for k in _keys or [
        'bxbxbar', 'bybybar', 'PhixPhixbar',
        'PhiyPhiybar', 'ThetaThetabar', 'PhiyThetabar',
        'deltamThetabar', 'deltamdeltambar', 'xexebar', 'xedeltambar', 'xeThetabar',
        'PhixPhixprimebar', 'PhiyPhiyprimebar', 'ThetaThetaprimebar'
            ]:    
            ang_avg_dict_master[k][kind] = ang_avg_dict[k]

    # Assembly shared with the single-mode path via pmhd.physics.source_assembly.
    reduce_ps = splint_reducer(karr, epsarr[input_epsind])
    xe2contsource = np.zeros( (len(zarr) ) )
    for z in range(len(zarr)):
        ## Compute all relevant rms and cross correlations needed for second order perturbed ionization fraction
        bars = {'deltamdeltambar': ang_avg_dict_master['deltamdeltambar'][:, z],
                'xedeltambar':     ang_avg_dict_master['xedeltambar'][:, z]}
        xe2contsource[z] = assemble_cont_at_z(bars, karr, zarr[z], xe_full, reduce_ps)
    
    source_fncs = OUTBASE/"source_fncs"
    source_fncs.mkdir(parents=True, exist_ok=True)

    # Save results
    np.save(source_fncs/f'xe2contsource_{"Tb_" if tb else ""}B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}.npy', xe2contsource)   

if __name__=='__main__':
    print(time.ctime())

    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", type=int, default=40)
    parser.add_argument("--epsind", type=int, default=9)
    parser.add_argument("--tb", action="store_true", help="use the T_b-coupled angle averages (ang_avg/Tb)")

    args = parser.parse_args()

    main(args.bind, args.epsind, tb=args.tb)
    

print(time.ctime())