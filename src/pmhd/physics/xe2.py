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
    
    ## Load abar2 and cbar 2. Choose at random a kindex since it is independent of wavenumber
    abar2 = np.zeros((len(zarr)))
    cbar2 = np.zeros((len(zarr)))
    for zind in range(len(zarr)):
        with open(PROJECT_ROOT/f'src/pmhd/data/pre_stored_data/f2bars_dict/f2bars_m0_dict_zch_{zind}.pkl', 'rb') as f:
            f2bars_m0_master = pickle.load(f)  
        abar2[zind] = f2bars_m0_master[f'df/dnu'][0,0]
        cbar2[zind] = f2bars_m0_master[f'taus_phi'][0,0]

    cross_corrsave = OUTBASE/"cross_corr"
    source_fncs = OUTBASE/"source_fncs"

     # Load source functions for given B_0 and eps
    sbar2 = np.load(source_fncs/f'ftot_B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}.npy')
    
    # Load in cross correlations for given B_0 and eps
    with open(cross_corrsave/f'cross_corr_B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}.pkl', 'rb') as f:
        cross_corr = pickle.load(f)

    ## Define ODE function for x1s^{(2)} evolution
    def x1squad(z,x1s2,epsind,Lambda,abar2,sbar2,cbar2,cont):

        ## Define relevant interpolation functions (splines built once in enclosing scope)

        def f00xecrossinterp(z):
            return splev(z, _tck_f00xecross)

        def feqxecrossinterp(z):
            return splev(z, _tck_feqxecross)

        def dmxecrossinterp(z):
            return splev(z, _tck_dmxecross)

        def dxedxecrossinterp(z):
            return splev(z, _tck_dxedxecross)

        def conterminterp(z):
            return splev(z, _tck_conterm)

        def abar2interp(z):
            return splev(z, _tck_abar2)
        def sbar2interp(z):
            return splev(z, _tck_sbar2)
        def cbar2interp(z):
            return splev(z, _tck_cbar2)
        def pesc2(z):
            return (1-pars.pab(z)*cbar2interp(z)) / (1 + (1-pars.pab(z))*cbar2interp(z))
        
        def P3LA2(z):
            return (3*cons.Alya * pesc2(z) + cons.L2s1s)/(
                3*cons.Alya * pesc2(z) + cons.L2s1s + 4*pars.betab(z) )
        
        def pscinterp(z):
            return 1-pars.pab(z)
            
     
        return (-1/( (1+z)*pars.H(z))) * ( 
            ( (1-P3LA2(z) )*pars.nh(z)*(xe_full(z)**2/(1-xe_full(z)))*pars.alphab(z) - 
            2 * P3LA2(z) * pars.nh(z) * xe_full(z) * pars.alphab(z) - 4*pars.feq(z,xe_full)* pars.betab(z) )*x1s2
            -3* (1-P3LA2(z) ) * (1-xe_full(z))*cons.Alya * 
            ( ( sbar2interp(z) + (abar2interp(z)*(x1s2 - dmxecrossinterp(z) )/(1-xe_full(z))) ) / (1+pscinterp(z)*cbar2interp(z)) - 
             f00xecrossinterp(z)/(1-xe_full(z)) ) +
            2 * P3LA2(z) * pars.nh(z) * xe_full(z) * pars.alphab(z) * dmxecrossinterp(z) + 
            P3LA2(z) * pars.nh(z) * pars.alphab(z) * dxedxecrossinterp(z) - 
            ( (1-P3LA2(z) ) * (3*cons.Alya + cons.L2s1s ) - 4*P3LA2(z)*pars.betab(z) ) * feqxecrossinterp(z) - conterminterp(z)
        )
     
    ## Load in continuous term
    cont = np.load(source_fncs/f'xe2contsource_B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}.npy')

    ## Build splines once here (outside x1squad) so they are not rebuilt on every ODE evaluation
    _zarr_flip = zarr[::-1]
    _tck_f00xecross    = splrep(_zarr_flip, cross_corr['df00xecross'][::-1])
    _tck_feqxecross    = splrep(_zarr_flip, cross_corr['dfeqxecross'][::-1])
    _tck_dmxecross     = splrep(_zarr_flip, cross_corr['deltamxecross'][::-1])
    _tck_dxedxecross   = splrep(_zarr_flip, cross_corr['xerms'][::-1])
    _tck_conterm       = splrep(_zarr_flip, cont[::-1])
    _tck_abar2         = splrep(np.flip(zarr), np.flip(abar2))
    _tck_sbar2         = splrep(np.flip(zarr), np.flip(sbar2))
    _tck_cbar2         = splrep(np.flip(zarr), np.flip(cbar2))

    ## Solve ODE for x1s^{(2)}
    hold = solve_ivp(x1squad, [1900,600], [1e-20], args = (input_epsind,Lambda,abar2,sbar2,cbar2,cont), method = 'LSODA', dense_output=True, atol=1e-9, rtol = 1e-6 )
    ## Create \Delta x_e from solution
    xe2 = ( abs(epsarr[input_epsind])/4 ) * (Lambda/(2*np.pi) )**((epsarr[input_epsind]))* (-hold.sol(zarr) + cross_corr['deltamxecross'])

    # Ensure directory exists
    xe2direc = OUTBASE/"xe2"
    xe2direc.mkdir(parents=True, exist_ok=True)    
    ## Save results
    np.save(xe2direc/f'xe2_B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}', xe2[0])

if __name__=='__main__':
    print(time.ctime())

    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", type=int, default=40)
    parser.add_argument("--epsind", type=int, default=9)

    args = parser.parse_args()

    main(args.bind, args.epsind)

   
    print(time.ctime())
    