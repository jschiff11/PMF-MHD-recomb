import os, sys

import numpy as np

lib_path = os.path.abspath('')
sys.path.append(lib_path)

from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo

from pmhd import cons, pars


from scipy.special import voigt_profile
from scipy.integrate import odeint
from scipy.linalg import solve_banded as sb
from scipy.interpolate import splrep, splev, splint
    
    
def psdHR(x_array,z_array,zind,taus,pab,feq):
          
    voigt = pars.voigt(x_array,z_array[zind]);
    
    voigtmids = (voigt[:-1] + voigt[1:]) /2
    
    fbb = pars.fbb(1000 , z_array, zind);  
    
    dx = 0.02 # step size for integration
    
    # Initialize matrix bands
    MHRjp1 = np.zeros(len(x_array))
    MHRj = np.zeros(len(x_array))
    MHRjm1 = np.zeros(len(x_array))

    # Neumann BC red side
    MHRjp1[1] = 1
    MHRj[0] = -1
    
    # First derivative 
    MHRjp1[2:] = 1
    MHRj[1:-1] = -1
    
    # Second derivative 
    MHRjp1[2:] += ( taus[zind] * ( 1 - pab[zind] ) / (
        2 * dx ) ) * voigtmids[1:]
    MHRj[1:-1] += ( - taus[zind] * pab[zind] * dx * voigt[1:-1] 
                   - ( taus[zind] * ( 1 - pab[zind] ) / (
                       2 * dx ) ) * ( voigtmids[:-1]
                                                                                    + voigtmids[1:] ) )
    MHRjm1[:-2] = ( taus[zind] * ( 1 - pab[zind] ) / (
        2 * dx ) ) * voigtmids[:-1]

    # Dirichlet BC blue side
    MHRj[-1] = 1

    #Construct RHS of matrix equation 
    bHR = np.zeros(len(x_array)) # leave first and last entries as zero for BCs
    bHR[1:-1] = - taus[zind] * pab[zind] * dx * voigt[1:-1]
    
    # Create banded matrix
    abHR = np.array([MHRjp1,
               MHRj,
               MHRjm1])
    
    # Solve for chi
    xiHR = sb((1, 1), abHR, bHR)
    
    # Chi*voigt profile
    xiphi = splrep( x_array, xiHR * voigt )
    # Chi averaged over voigt profile
    xibar = splint( x_array[0], x_array[-1], xiphi )
    
    # f_em, parameterizing true emission
    fem = (
            ( feq[zind] - ( 
                1.0 - pab[zind]) * ( 1.0 - xibar ) * fbb )/
            ( 1.0 - ( 1.0 - pab[zind] ) * ( 1.0 - xibar ) )
            )
    # Final psd result
    psdhomo = xiHR * ( fem - fbb ) + fbb
    
    return xiHR, xibar, psdhomo