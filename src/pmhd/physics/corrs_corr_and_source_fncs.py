# pyright: ignore[reportMissingImports]
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import time
import os
import sys
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))

from scipy.integrate import odeint, solve_ivp

from astropy.cosmology import Planck18 as cosmo

from scipy.interpolate import splrep, splev, splint
from scipy.linalg import solve_banded as sb
import pickle

from pmhd import cons, pars
from pmhd.data.grids import (
    k_grid, z_grid, eps_grid,
    load_or_generate_B0arr
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
    
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    with open(PROJECT_ROOT/'src/pmhd/data/pre_stored_data/abarinterpmaster.pkl', 'rb') as f:
        abarinterpmaster = pickle.load(f)

    with open(PROJECT_ROOT/'src/pmhd/data/pre_stored_data/bbarinterpmaster.pkl', 'rb') as f:
        bbarinterpmaster = pickle.load(f)

    with open(PROJECT_ROOT/'src/pmhd/data/pre_stored_data/cbarinterpmaster.pkl', 'rb') as f:
        cbarinterpmaster = pickle.load(f)

    def abarinterpfunc(z,kind):
        return splev(z, abarinterpmaster[kind])

    def bbarinterpfunc(z,kind):
        return splev(z, bbarinterpmaster[kind])

    def cbarinterpfunc(z,kind):
        return splev(z, cbarinterpmaster[kind])


    ## Define prefactors for cross correlations between feq and fsc and fluid variables with the conventions that
    ## dfeq = \alphaeq \delta_m + \betaeq \delta x_e/x_{1s} + \gammaeq \Theta/aH
    ## df00 = \alpha00 \delta_m + \beta00 \delta x_e/x_{1s} + \gamma00 \Theta/aH
    ## dfeq - pscdf00 = alphabar \delta_m + \betabar \delta x_{1s}/x_{1s} + \gammabar \Theta/aH
    ## so that we have: alphabar = alphaeq - p_sc alpha00, betabar = -betaeq + p_sc beta00, gammabar = gammaeq - p_sc gamma00,

    alphaeq = np.zeros( (len(zarr), len(karr) ) )
    betaeq = np.zeros( (len(zarr), len(karr) ) )
    gammaeq = np.zeros( (len(zarr), len(karr) ) )

    alpha00 = np.zeros( (len(zarr), len(karr) ) )
    beta00 = np.zeros( (len(zarr), len(karr) ) )
    gamma00 = np.zeros( (len(zarr), len(karr) ) )

    alphabar = np.zeros( (len(zarr), len(karr) ) )
    betabar = np.zeros( (len(zarr), len(karr) ) )
    gammabar = np.zeros( (len(zarr), len(karr) ) )

    paball = pars.pab(zarr)
    pscall = 1- paball
    alphaball = pars.alphab(zarr)
    betaball = pars.betab(zarr)
    nhall = pars.nh(zarr)

    xearr = xe_full(zarr)

    for k in range(len(karr)):
        alphaeq[:,k] = (
            3* cons.Alya *  ( (1 - cbarinterpfunc(zarr,k)* paball)/( 1 + cbarinterpfunc(zarr,k)* pscall ) )  + cons.L2s1s + 4*betaball  
        )**(-1) *(
            nhall * ((xearr**2)/(1-xearr) ) * alphaball + 
            3* cons.Alya * ( abarinterpfunc(zarr,k)/( 1 + cbarinterpfunc(zarr,k)*pscall )  ) 
        )
        
        betaeq[:,k] = (
            3* cons.Alya *  ( (1 - cbarinterpfunc(zarr,k)* paball)/( 1 + cbarinterpfunc(zarr,k)* pscall ) )  + cons.L2s1s + 4*betaball
        )**(-1) *(
            nhall * (xearr**2)/(1-xearr) * alphaball + ( 2 * nhall * alphaball * xearr ) - 
            3* cons.Alya * ( abarinterpfunc(zarr,k)/( 1 + cbarinterpfunc(zarr,k)*pscall)  ) 
        )
        
        gammaeq[:,k] = (
            3* cons.Alya *  ( (1 - cbarinterpfunc(zarr,k)* paball)/( 1 + cbarinterpfunc(zarr,k)* pscall ) )  + cons.L2s1s + 4*betaball
        )**(-1) *( 
            3* cons.Alya * ( bbarinterpfunc(zarr,k)/( 1 +  cbarinterpfunc(zarr,k)*pscall)  ) 
        )
        
        alpha00[:,k] = ( abarinterpfunc(zarr,k)/( 1 + cbarinterpfunc(zarr,k)*pscall ) ) + alphaeq[:,k] * ( cbarinterpfunc(zarr,k)/( 1 + cbarinterpfunc(zarr,k)*pscall) )

        beta00[:,k] = -( abarinterpfunc(zarr,k)/( 1 + cbarinterpfunc(zarr,k)*pscall ) ) + betaeq[:,k] * ( cbarinterpfunc(zarr,k)/( 1 + cbarinterpfunc(zarr,k)*pscall) )

        gamma00[:,k] = ( bbarinterpfunc(zarr,k)/( 1 + cbarinterpfunc(zarr,k)*pscall) ) + gammaeq[:,k] * ( cbarinterpfunc(zarr,k)/( 1 + cbarinterpfunc(zarr,k)*pscall) )
        
        alphabar[:,k] = alphaeq[:,k] - pscall*alpha00[:,k] 
        
        betabar[:,k] =  -betaeq[:,k] + pscall*beta00[:,k]
        
        gammabar[:,k] =  gammaeq[:,k] - pscall*gamma00[:,k]
        

    output_xs, dx = np.linspace( -1000.0, 1000.0, num=100001, retstep=True )

    ## For conveneince make arrays of of taus, deltah, and H for all values of z under consideration 
    tausall = pars.taus(zarr,xe_full)
    Deltah = pars.Deltah(zarr)
    H = pars.H(zarr)

    ## Prepare arrays
    fred11 = np.zeros(len(zarr))
    fred20 = np.zeros(len(zarr))
    fproc = np.zeros(len(zarr))
    fadveclens = np.zeros(len(zarr))
    cross_corr = {}

    for k in [
        'df00xecross', 'dfeqxecross', 'df00deltamcross',
        'dfeqdeltamcross', 'deltamxecross', 'xeThetacross',
        'xerms', 'deltamrms', 'vxrms', 'vyrms', 'vzrms'
    ]:    
        cross_corr[k] = np.zeros(len(zarr))
    
    ang_avg_dict_master = {}
    for k in [
        'bxbxbar', 'bybybar', 'PhixPhixbar',
        'PhiyPhiybar', 'ThetaThetabar', 'PhiyThetabar',
        'deltamThetabar', 'deltamdeltambar', 'xexebar', 'xedeltambar', 'xeThetabar',
        'PhixPhixprimebar', 'PhiyPhiyprimebar', 'ThetaThetaprimebar'
    ]:    
        ang_avg_dict_master[k] = np.zeros( ( len(karr), len(zarr)) )
    
    

    OUTBASE = PROJECT_ROOT / "data/outputs"
    ang_avg_TLA = OUTBASE/f"ang_avg/TLA/B_{round(1e12*B0arr[input_bind])}pG"
    
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
    
    f2bars_m0_master = {}
    f2bars_m1_master = {}
    f2bars_p1_master = {}
    for zind in range(len(zarr)):
        
        taus_phi = np.zeros( ( len(karr) ) )
        
        with open(PROJECT_ROOT/f'src/pmhd/data/pre_stored_data/f2bars_dict/f2bars_m0_dict_zch_{zind}.pkl', 'rb') as f:
                f2bars_m0_master = pickle.load(f) 
        with open(PROJECT_ROOT/f'src/pmhd/data/pre_stored_data/f2bars_dict/f2bars_m1_dict_zch_{zind}.pkl', 'rb') as f:
                f2bars_m1_master = pickle.load(f) 
        with open(PROJECT_ROOT/f'src/pmhd/data/pre_stored_data/f2bars_dict/f2bars_p1_dict_zch_{zind}.pkl', 'rb') as f:
                f2bars_p1_master = pickle.load(f) 
    
        A0 = f2bars_m0_master['mom0_total'][:,0]
        B0 = f2bars_m0_master['mom0_total'][:,1]
        C0 = f2bars_m0_master['mom0_total'][:,2]
        D0_m = f2bars_m1_master['mom0_total'][:,0]
        D0_p = f2bars_p1_master['mom0_total'][:,0]

        A1 = f2bars_m0_master['mom1_total'][:,0]
        B1 = f2bars_m0_master['mom1_total'][:,1]
        C1 = f2bars_m0_master['mom1_total'][:,2]
        D1_m = f2bars_m1_master['mom1_total'][:,0]
        D1_p = f2bars_p1_master['mom1_total'][:,0]

        A2 = f2bars_m0_master['mom2_total'][:,0]
        B2 = f2bars_m0_master['mom2_total'][:,1]
        C2 = f2bars_m0_master['mom2_total'][:,2]
        D2_m = f2bars_m1_master['mom2_total'][:,0]
        D2_p = f2bars_p1_master['mom2_total'][:,0]

        dx_A0 = f2bars_m0_master['dx_mom0_total'][:,0]
        dx_B0 = f2bars_m0_master['dx_mom0_total'][:,1]
        dx_C0 = f2bars_m0_master['dx_mom0_total'][:,2]
        dx_D0_m = f2bars_m1_master['dx_mom0_total'][:,0]
        dx_D0_p = f2bars_p1_master['dx_mom0_total'][:,0]

        dx_A1 = f2bars_m0_master['dx_mom1_total'][:,0]
        dx_B1 = f2bars_m0_master['dx_mom1_total'][:,1]
        dx_C1 = f2bars_m0_master['dx_mom1_total'][:,2]
        dx_D1_m = f2bars_m1_master['dx_mom1_total'][:,0]
        dx_D1_p = f2bars_p1_master['dx_mom1_total'][:,0]

        dx_A2 = f2bars_m0_master['dx_mom2_total'][:,0]
        dx_B2 = f2bars_m0_master['dx_mom2_total'][:,1]
        dx_C2 = f2bars_m0_master['dx_mom2_total'][:,2]
        dx_D2_m = f2bars_m1_master['dx_mom2_total'][:,0]
        dx_D2_p = f2bars_p1_master['dx_mom2_total'][:,0]

        dx_phi_dx_A0 = f2bars_m0_master['dx_phi_dx_mom0_total'][:,0]
        dx_phi_dx_B0 = f2bars_m0_master['dx_phi_dx_mom0_total'][:,1]
        dx_phi_dx_C0 = f2bars_m0_master['dx_phi_dx_mom0_total'][:,2]

        phi_A0 = f2bars_m0_master['phi_mom0_total'][:,0]
        phi_B0 = f2bars_m0_master['phi_mom0_total'][:,1]
        phi_C0 = f2bars_m0_master['phi_mom0_total'][:,2]

        taus_phi = f2bars_m0_master[f'taus_phi'][0,0]
           

        cross_corr['deltamrms'][zind] = splint(
             karr[-1], karr[0], (splrep(
                np.flip(karr), np.flip(
                    karr**(epsarr[input_epsind] - 1) * (
                    ang_avg_dict_master['deltamdeltambar'][:,zind] ) 
            )))
        )
        cross_corr['vxrms'][zind] = splint(
             karr[-1], karr[0], (splrep(
                np.flip(karr), np.flip(
                    karr**(epsarr[input_epsind] - 3) * (
                    ang_avg_dict_master['PhixPhixbar'][:,zind] ) 
                )))
        )
        cross_corr['vyrms'][zind] = splint(
             karr[-1], karr[0], (splrep(
                np.flip(karr), np.flip(
                    karr**(epsarr[input_epsind] - 3) * (
                    ang_avg_dict_master['PhiyPhiybar'][:,zind] ) 
            )))
        )
        cross_corr['vzrms'][zind] = splint(
             karr[-1], karr[0], (splrep(
                np.flip(karr), np.flip(
                    karr**(epsarr[input_epsind] - 3) * (
                    ang_avg_dict_master['ThetaThetabar'][:,zind] ) 
            )))
        )
        cross_corr['xerms'][zind] = splint(
             karr[-1], karr[0], (splrep(
                np.flip(karr), np.flip(
                    karr**(epsarr[input_epsind] - 1) * (
                    ang_avg_dict_master['xexebar'][:,zind] ) 
            )))
        )
        cross_corr['df00xecross'][zind] = splint(
             karr[-1], karr[0], (splrep(
                np.flip(karr), np.flip(
                    karr**(epsarr[input_epsind] - 1) * (
                    alpha00[zind,:]*ang_avg_dict_master['xedeltambar'][:,zind] + ( 
                        beta00[zind,:]/(1-xearr[zind]) )*ang_avg_dict_master['xexebar'][:,zind] + (  
                        gamma00[zind,:]*(1+zarr[zind])/H[zind]) * ang_avg_dict_master['xeThetabar'][:,zind]  ) 
            )))
        )
        cross_corr['dfeqxecross'][zind] = splint(
             karr[-1], karr[0], (splrep(
                np.flip(karr), np.flip(
                    karr**(epsarr[input_epsind] - 1) * (
                    alphaeq[zind,:]*ang_avg_dict_master['xedeltambar'][:,zind] + ( 
                        betaeq[zind,:]/(1-xearr[zind]) )*ang_avg_dict_master['xexebar'][:,zind] + (  
                        gammaeq[zind,:]*(1+zarr[zind])/H[zind]) * ang_avg_dict_master['xeThetabar'][:,zind]  ) 
            )))
        )
        cross_corr['df00deltamcross'][zind] = splint(
             karr[-1], karr[0], (splrep(
                np.flip(karr), np.flip(
                    karr**(epsarr[input_epsind] - 1) * (
                    alpha00[zind,:]*ang_avg_dict_master['deltamdeltambar'][:,zind] + ( 
                        beta00[zind,:]/(1-xearr[zind]) )*ang_avg_dict_master['xedeltambar'][:,zind] + (  
                        gamma00[zind,:]*(1+zarr[zind])/H[zind]) * ang_avg_dict_master['deltamThetabar'][:,zind]  ) 
            )))
        )
        cross_corr['dfeqdeltamcross'][zind] = splint(
             karr[-1], karr[0], (splrep(
                np.flip(karr), np.flip(
                    karr**(epsarr[input_epsind] - 1) * (
                    alphaeq[zind,:]*ang_avg_dict_master['deltamdeltambar'][:,zind] + ( 
                        betaeq[zind,:]/(1-xearr[zind]) )*ang_avg_dict_master['xedeltambar'][:,zind] + (  
                        gammaeq[zind,:]*(1+zarr[zind])/H[zind]) * ang_avg_dict_master['deltamThetabar'][:,zind]  ) 
            )))
        )
        cross_corr['deltamxecross'][zind] = splint(
             karr[-1], karr[0], (splrep(
                np.flip(karr), np.flip(
                    karr**(epsarr[input_epsind] - 1) * (
                    ang_avg_dict_master['xedeltambar'][:,zind] ) 
            )))
        )
        cross_corr['xeThetacross'][zind] = splint(
             karr[-1], karr[0], (splrep(
                np.flip(karr), np.flip(
                    karr**(epsarr[input_epsind] - 1) * (
                    ang_avg_dict_master['xeThetabar'][:,zind] ) 
            )))
        )

        ## Compute source functions
        fredintegrand = splrep(
                np.flip(karr), np.flip(
                    karr**(epsarr[input_epsind] - 1) * 
            ( 
                (-(1+zarr[zind])**2/(5*H[zind]**2 )) * (
                    ang_avg_dict_master['PhixPhixbar'][:,zind] + ang_avg_dict_master['PhiyPhiybar'][:,zind] 
                ) * ( dx_D2_p + dx_D2_m  ) 
                +  ((1+zarr[zind])/(3*H[zind])) * (ang_avg_dict_master['deltamThetabar'][:,zind]  - (1/ ( 1- xearr[zind] ) ) * ang_avg_dict_master['xeThetabar'][:,zind] 
                ) * ( (2/5)*dx_A2 - dx_A0  )  
                + ((1+zarr[zind])**2/(3*H[zind]**2) ) * (ang_avg_dict_master['ThetaThetabar'][:,zind]) * ( (2/5)*dx_B2 - dx_B0  ) 
                + ((1+zarr[zind])/(3*H[zind])) * (alphabar[zind,:] * ang_avg_dict_master['deltamThetabar'][:,zind] - betabar[zind,:]*ang_avg_dict_master['xeThetabar'][:,zind] + 
               ( gammabar[zind,:]*(1+zarr[zind])/H[zind] )*ang_avg_dict_master['ThetaThetabar'][:,zind]  ) * 
            ( (2/5)*dx_C2 - dx_C0  ) 
            ) 
                )
        )

        fred11[zind] = splint(karr[-1], karr[0], fredintegrand )

        fprocintegrand = splrep(
            np.flip(karr), np.flip(
                karr**(epsarr[input_epsind] - 1) 
            * ( (ang_avg_dict_master['deltamdeltambar'][:,zind] + (1/ ( 1- xearr[zind] )**2 ) * ang_avg_dict_master['xexebar'][:,zind] - 
               (2/ ( 1- xearr[zind] ) ) * ang_avg_dict_master['xedeltambar'][:,zind]  ) * 
            (tausall[zind]*paball[zind]*phi_A0 - (1-paball[zind])*(tausall[zind]/2) * dx_phi_dx_A0  )
               + ((1+zarr[zind])/H[zind]) * (ang_avg_dict_master['deltamThetabar'][:,zind] - 
               (1/ ( 1- xearr[zind] ) ) * ang_avg_dict_master['xeThetabar'][:,zind]  ) * 
            (tausall[zind]*paball[zind]*phi_B0 - (1-paball[zind])*(tausall[zind]/2) * dx_phi_dx_B0  ) 
               + (
                alphabar[zind,:] * ang_avg_dict_master['deltamdeltambar'][:,zind] - 
               ( (alphabar[zind,:] + betabar[zind,:])/(1-xearr[zind]) ) * ang_avg_dict_master['xedeltambar'][:,zind] + 
                (betabar[zind,:]/(1-xearr[zind])**2 ) * ang_avg_dict_master['xexebar'][:,zind] + 
                (gammabar[zind,:]*(1+zarr[zind])/H[zind]) * ang_avg_dict_master['deltamThetabar'][:,zind] - 
                (gammabar[zind,:]*(1+zarr[zind])/ ( H[zind] *(1-xearr[zind] ) ) ) * ang_avg_dict_master['xeThetabar'][:,zind] 
               ) * 
            (tausall[zind]*paball[zind]*phi_C0 - 
             (1-paball[zind])*(tausall[zind]/2) * dx_phi_dx_C0 + taus_phi   )
            )
            )
        )

        
        fproc[zind] = splint(karr[-1], karr[0], fprocintegrand )



        fred20[zind] = ((1+zarr[zind])/(3*H[zind] * cons.c**2) ) * splint(
            karr[-1], karr[0], 
        splrep(
            np.flip(karr), np.flip(
                karr**(epsarr[input_epsind] - 3) 
            * ( 
                16* np.pi * cons.G * cosmo.Ob0*cosmo.critical_density0.value * (1+zarr[zind]) * ang_avg_dict_master['deltamThetabar'][:,zind] - 
                H[zind] * ang_avg_dict_master['PhixPhixprimebar'][:,zind] - H[zind] * ang_avg_dict_master['PhiyPhiyprimebar'][:,zind] -
                4*H[zind] * ang_avg_dict_master['ThetaThetaprimebar'][:,zind]
            )
            ) ) )
            
        fadveclens[zind] = - ( (8*np.pi * cons.G * cosmo.Ob0*cosmo.critical_density0.value * Deltah[zind] * (1+zarr[zind])**2 )/(3*cons.c*H[zind])  ) * splint(
            karr[-1], karr[0], 
        splrep(
            np.flip(karr), np.flip(
                karr**(epsarr[input_epsind] - 2) 
            * (
                (A1 + alphabar[zind,:] * C1 ) * ang_avg_dict_master['deltamdeltambar'][:,zind] - 
               ((A1 + betabar[zind,:] * C1)/(1-xearr[zind]))*ang_avg_dict_master['xedeltambar'][:,zind] 
                +  ( ( B1 +  gammabar[zind,:] * C1 ) *(1+zarr[zind]) / H[zind] ) * ang_avg_dict_master['deltamThetabar'][:,zind] 
            )
        )
        )
        )
    
                            
    ftot = fred11 + fproc + fadveclens

    # Ensure directory exists
    # Output base directory
    OUTBASE = PROJECT_ROOT / "data/outputs"

    source_fncs = OUTBASE/"source_fncs"
    cross_corrsave = OUTBASE/"cross_corr"
    source_fncs.mkdir(parents=True, exist_ok=True)
    cross_corrsave.mkdir(parents=True, exist_ok=True)

    with open(cross_corrsave/f'cross_corr_B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}.pkl', 'wb') as f:
            pickle.dump(cross_corr, f)

    np.save(source_fncs/f'ftot_B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}.npy',ftot)

if __name__=='__main__':
    print(time.ctime())
        
    main(int(sys.argv[1]), int(sys.argv[2]))

    print(time.ctime())

    