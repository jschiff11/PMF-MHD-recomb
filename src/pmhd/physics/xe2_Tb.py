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
    xesaha_full = pars.xesaha_full
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

     # Load Tb source functions for given B_0 and eps
    sbar2 = np.load(source_fncs/f'ftot_Tb_B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}.npy')

    # Load Tb cross correlations (all keys recomputed from Tb ang_avg)
    with open(cross_corrsave/f'cross_corr_Tb_B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}.pkl', 'rb') as f:
        cross_corr = pickle.load(f)

    sol = odeint(pars.RHSTbhom, pars.Tcmb(1900), np.arange(1900,600,-0.01), args=(xe_full, ))

    zhold = np.arange(1900,600,-0.01)
    Tbhomspl = np.array([splrep(zhold[::-1], sol.flatten()[::-1] )], dtype=object)

    def Tbhom(z):
        return splev(z,Tbhomspl[0])

    def x1squad_Tb(z, vect, xe):

        x1s2, Delta2_b = vect

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

        def dalphabpref(z):
            a=4.309; b=-0.6166; c=0.6703; d=0.53
            return (b + (b-d)*c*(Tbhom(z)/10**4)**d )/(1 + c*(Tbhom(z)/10**4)**d) 
        
        def dalphabpref2_11(z):
            a=4.309; b=-0.6166; c=0.6703; d=0.53
            return (1/(1 + c*(Tbhom(z)/10**4)**d)**2 ) * (
                b*(b-1)/2 
                + c*(Tbhom(z)/10**4)**d * (b*(b-1) - b*d - d*(d-1)/2 ) 
                + (c*(Tbhom(z)/10**4)**d)**2 * ( b*(b-1)/2 -b*d + d*(d+1)/2 )
            )

        def Tbxeinterp(z):
            return splev(z, _tck_Tbxecross)

        def Tbdeltainterp(z):
            return splev(z, _tck_Tbdelta)

        def Tbrmsinterp(z):
            return splev(z, _tck_Tbrms)

        def TbThetainterp(z):
            return splev(z, _tck_TbTheta)

        def deltaThetainterp(z):
            return splev(z, _tck_deltaTheta)

        def xexermsinterp(z):
            return splev(z, _tck_xexerms)
        
        fhe = cons.yhe/(4.0*(1.0 - cons.yhe))
       
        dGammacpref1 = (1 + fhe)/(xe(z)* (1 + xe(z) + fhe ) )
        dGammacpref2 = (1 + fhe)/(xe(z)* (1 + xe(z) + fhe )**2)
            
        gammaT = 8*cons.arad*xe(z)*pars.Tcmb(z)**4*cons.sigmat/(3*cons.me * (1+xe(z)+fhe) )/cons.c
    
        kineticderiv = splev(z, _tck_kinetic, der=1)
        magneticderiv = splev(z, _tck_magnetic, der=1)
        
        totalheat = (abs(epsarr[input_epsind])/4 * (Lambda/(2*np.pi))**epsarr[input_epsind] * (kineticderiv/2 + magneticderiv/(8*np.pi)) )

        return [(-1/( (1+z)*pars.H(z))) * ( 
            ( (1-P3LA2(z) )*pars.nh(z)*(xe_full(z)**2/(1-xe_full(z)))*pars.alphab(z) - 
            2 * P3LA2(z) * pars.nh(z) * xe_full(z) * pars.alphab(z) - 4*pars.feq(z,xe_full)* pars.betab(z) )*x1s2
            -3* (1-P3LA2(z) ) * (1-xe_full(z))*cons.Alya * 
            ( ( sbar2interp(z) + (abar2interp(z)*(x1s2 - dmxecrossinterp(z) )/(1-xe_full(z))) ) / (1+pscinterp(z)*cbar2interp(z)) - 
             f00xecrossinterp(z)/(1-xe_full(z)) ) +
            P3LA2(z) * pars.nh(z) * xe_full(z)**2 * pars.alphab(z) * (dalphabpref(z)*Tbdeltainterp(z) + dalphabpref2_11(z)*Tbrmsinterp(z) + dalphabpref(z) * Delta2_b ) +
            2 * P3LA2(z) * pars.nh(z) * xe_full(z) * pars.alphab(z) * ( dmxecrossinterp(z) + dalphabpref(z)*Tbxeinterp(z) ) + 
            P3LA2(z) * pars.nh(z) * pars.alphab(z) * dxedxecrossinterp(z) - 
            ( (1-P3LA2(z) ) * (3*cons.Alya + cons.L2s1s ) - 4*P3LA2(z)*pars.betab(z) ) * feqxecrossinterp(z) - conterminterp(z)) , 
            (-1/( (1+z)*pars.H(z) )) * ( 
                gammaT * (
                (pars.Tcmb(z)/Tbhom(z) - 1)*(-dGammacpref2 *xexermsinterp(z) - dGammacpref1 *x1s2)  
                - (pars.Tcmb(z)/Tbhom(z))*Delta2_b 
                - dGammacpref1*Tbxeinterp(z)
                ) 
                +(2/3)*( totalheat/(pars.nh(z)*(1+xe(z)+fhe)))/Tbhom(z) + (2/3)*(1+z)*(deltaThetainterp(z)-TbThetainterp(z))
            )]

     
    ## Load in continuous term
    cont = np.load(source_fncs/f'xe2contsource_B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}.npy')

    ## Load magnetic field angle-averaged squared amplitudes per k-mode (from angle_avging_Tb output)
    ang_avg_Tb_dir = OUTBASE/f"ang_avg/Tb/B_{round(1e12*B0arr[input_bind])}pG"
    bxbxbar = np.zeros((len(karr), len(zarr)))
    bybybar = np.zeros((len(karr), len(zarr)))
    for kind in range(len(karr)):
        with open(ang_avg_Tb_dir/f'ang_avg_k{kind}_Tb.pkl', 'rb') as f:
            ang_avg_Tb_dict = pickle.load(f)
        bxbxbar[kind,:] = ang_avg_Tb_dict['bxbxbar']
        bybybar[kind,:] = ang_avg_Tb_dict['bybybar']

    ## Integrate over k to get total magnetic energy per redshift bin
    bxsquare = np.zeros(len(zarr))
    bysquare = np.zeros(len(zarr))
    for zind in range(len(zarr)):
        bxsquare[zind] = splint(
            karr[-1], karr[0], splrep(
                np.flip(karr), np.flip(karr**(epsarr[input_epsind]-1) * bxbxbar[:,zind])))
        bysquare[zind] = splint(
            karr[-1], karr[0], splrep(
                np.flip(karr), np.flip(karr**(epsarr[input_epsind]-1) * bybybar[:,zind])))

    rhob0 = cosmo.Ob0 * cosmo.critical_density0.value
    kinetic_term = rhob0 * (1+zarr)**3 * (cross_corr['vxrms'] + cross_corr['vyrms'] + cross_corr['vzrms'])
    magnetic_term = bxsquare + bysquare

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
    _tck_Tbxecross     = splrep(_zarr_flip, cross_corr['Tbxecross'][::-1])
    _tck_Tbdelta       = splrep(_zarr_flip, cross_corr['Tbdeltacross'][::-1])
    _tck_Tbrms         = splrep(_zarr_flip, cross_corr['Tbrms'][::-1])
    _tck_TbTheta       = splrep(_zarr_flip, cross_corr['TbThetacross'][::-1])
    _tck_deltaTheta    = splrep(_zarr_flip, cross_corr['deltaThetacross'][::-1])
    _tck_xexerms       = splrep(_zarr_flip, cross_corr['xerms'][::-1])
    _tck_kinetic       = splrep(_zarr_flip, kinetic_term[::-1])
    _tck_magnetic      = splrep(_zarr_flip, magnetic_term[::-1])

    ## Solve coupled ODE for [x1s^{(2)}, Delta2_b]
    hold = solve_ivp(x1squad_Tb, [1900,600], [1e-20, 0.0], args=(xe_full,), method='LSODA', dense_output=True, atol=1e-9, rtol=1e-6)
    norm = (abs(epsarr[input_epsind])/4) * (Lambda/(2*np.pi))**epsarr[input_epsind]
    ## Create \Delta x_e from solution (component 0 is x1s^{(2)})
    xe2 = norm * (-hold.sol(zarr)[0] + cross_corr['deltamxecross'])
    ## Create \Delta_b^{(2)} = delta T_b^{(2)} / T_b^{(0)} from solution (component 1)
    dTb2 = norm * hold.sol(zarr)[1]

    # Ensure directory exists
    xe2direc = OUTBASE/"xe2"
    xe2direc.mkdir(parents=True, exist_ok=True)
    ## Save results
    np.save(xe2direc/f'xe2_Tb_B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}', xe2)
    np.save(xe2direc/f'dTb2_B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}', dTb2)

if __name__=='__main__':
    print(time.ctime())

    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", type=int, default=40)
    parser.add_argument("--epsind", type=int, default=9)

    args = parser.parse_args()

    main(args.bind, args.epsind)

   
    print(time.ctime())
    