import os, sys

import numpy as np

from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo

from . import cons

from scipy.special import voigt_profile
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from scipy.interpolate import splrep, splev, splint
import numdifftools as nd


def Tcmb(z):
    r"""
    Returns CMB temperature as a function of redshift, assuming T = T_0(1+z). T_0 = present day CMB temperature
    """
    return cons.T0*(1.0 + z)

def bb(nu, T):
    r"""
    Returns planck distribution at frequency nu, and temperature T
    """

    return 1.0/(np.exp(cons.h*nu/(cons.kb*T))-1.0)

# My function for bb psd
def fbb(x_array,z_array,zind):
    return (np.exp ( (cons.h * cons.nuly * ( 1 + x_array * Deltah(z_array)[zind] ) ) / ( cons.kb * Tcmb(z_array)[zind] ) ) -1 )**-1

# def H(z):
#     return cosmo.H(z).to('1/s').value

def H(z):
    r"""
    Returns hubble rate as a function of redshift
    """

    hrate = (cons.H0 * np.sqrt(cons.omat*(1.0 + z)**3 + cons.olda + 
	    cosmo.Ogamma0*(1 + 
		    cons.Neff*(7.0/8.0)*np.power(4.0/11.0,4.0/3.0))*(1.0+z)**4))
    return hrate


def nh(z):
    r"""
    Returns hydrogen number density in 1/cm^3
    """
    ndens = (cons.obh2*3.0*(1.0e+7/cons.mpc)**2*(1.0 + z)**3*(1.0 - cons.yhe)
		    /(8.0*np.pi*cons.G*cons.mh))
    return ndens


### Recombination coefficients taken from Pequignot, Petitjean, and Boisson 1991
def alpha1s(z):
    r"""
    Returns sum of recombination coefficients for recombination to the 1s level as a function of redshift
    """
    alpha = 1.0e-13*5.596*np.power(Tcmb(z)/1.0e4,-0.6038)/(1.0+(0.3436*np.power(Tcmb(z)/1.0e4,0.4479)))
    return alpha

def alphab(z):
    r"""
    Returns Case-B recombination coefficient as a function of redshift
    """
    return (1e-13)*((4.309*(Tcmb(z)/10**4)**-0.6166)/(1 + 0.6703*(Tcmb(z)/10**4)**0.53))

def betab(z):
    r"""
    Returns Case-B photoionization coefficient as a function of redshift
    """
    return ( (2 * np.pi * cons.me * cons.kb * Tcmb(z) )**(3/2) / (
        4* cons.h**3) ) * alphab(z) * np.exp( cons.en2 / ( cons.kb * Tcmb(z) ) )

def alphab_Tb(z,Tb):
    r"""
    Returns Case-B recombination coefficient as a function of redshift and matter temperature Tb which need not equal Tcmb
    """
    return (1e-13)*((4.309*(Tb(z)/10**4)**-0.6166)/(1 + 0.6703*(Tb(z)/10**4)**0.53))

def betab_Tb(z, Tb):
     r"""
     Returns Case-B photoionization coefficient as a function of redshift and matter temperature Tb which need not equal Tcmb
     """
     return ( (2 * np.pi * cons.me * cons.kb * Tcmb(z) )**(3/2) / (
        4* cons.h**3) ) * alphab_Tb(z,Tb) * np.exp( cons.en2 / ( cons.kb * Tb(z) ) ) 

def peebC_Tb(z,xe,Tb):
    return (3 * Rlya(z,xe) + cons.L2s1s ) / ( 3 * Rlya(z,xe) + cons.L2s1s + 4  * betab_Tb(z,Tb))

def RHSsob_Tb(xe,z,Tb):
    hold = ( peebC_Tb(z,xe,Tb) / ( (1+z) * H(z) ) ) * ( 
        nh(z) * xe**2  * alphab_Tb(z,Tb) - 4*(1-xe) *  betab(z) * np.exp(-(cons.en2 - cons.en1) / ( cons.kb * Tcmb(z) ) ) )
    return hold

def pab(z):
    r"""
    Returns absorption probability from the 2p state as a function of redshift assuming the dminant contribution is to the 3s and 3d states
    """
    return ((cons.A3s2p+5.0*cons.A3d2p)/(3.0*cons.Alya))*bb(5.0*cons.nuly/27.0, Tcmb(z))

def Deltah(z):
    r"""
    Returns the dimensionless doppler width for Ly alpha photons as a function of redshift 
    """
    return np.sqrt(2 * cons.kb * Tcmb(z) / (cons.mh * cons.c**2 ) ); 


def a(z):
    r"""
    Returns the dimensionless voigt parameter for lyman alpha photons as a function of redshift 
    """
    return ( 1 / ( 1 - pab(z) ) )*( cons.Alya ) / ( 4 * np.pi * cons.nuly * Deltah(z)) 

def voigt(x_array,z):
    r"""
    Returns voigt profile as a function of redshift for a frequency array defined in terms of 
    the deviation froim the central frequency in Doppler widths: x_array = (nu - nu_lya)/(nu_lya\Delta_H)
    """
    return voigt_profile(x_array,1/np.sqrt(2),a(z))

def voigtmids(x_array,z):
    r"""
    Returns voigt profile as a function of redshift at half doppler width values which is needed for 
    the second derivative term from the Fokker-Planck approximation of the Boltzmann equation
    """
    return (voigt(x_array,z)[:-1] + voigt(x_array,z)[1:]) /2

def voigtall(x_array,z_array):
    r"""
    Returns voigt profile for an array of redshifts z_array and frequencies x_array
    """
    hold = np.zeros( (len(z_array), len(x_array) ) )
    for i in range(len(z_array)):
        hold[i,:] = voigt_profile(x_array,1/np.sqrt(2),a(z_array[i]))
    return hold

def taus(z,xehomo):
    r"""
    Returns the sobolev optical depth as a function of redshift and ionization fraction. xehomo should itself be a function
    """
    return ( 3 * cons.llya**3 * nh(z) * (1 - xehomo(z)) ) / ( 8 * np.pi * H(z) ) * cons.Alya

## Sobolev approximation (xehomo)
## Define Rlya and Peebles C factor as function of redshift and ionization fraction

def sigmaa(z):
    r"""
    Returns ionization cross section from 1s as a function of redshift
    """
    sigma = ((1.0/4.0)*cons.me*cons.c**2*np.sqrt(2.0*np.pi*cons.me*cons.kb*Tcmb(z))*(3.0*cons.llya)**2/((4.0*cons.h*cons.c)**2))*(alpha1s(z) - alphab(z))
    return sigma


def f_lambda(z):
    r"""
    Returns force per unit mass, per unit ionization fraction and relative velocity if the universe were composed of only Hydrogen
    """
    return (4.0/3.0)*cons.sigmat*cons.arad*(Tcmb(z)**4/(cons.mh*cons.c))

def Rlya(z,xe):
    r"""
    Returns the escape rate of Ly \alpha photons per atom in the 2p state as a function of redshift and ionizatio fraction
    """
    return ( 8 * np.pi * H(z) ) / ( 3 * nh(z) * cons.llya**3 * (1 - xe))

def Rlyahompert(z,xe,theta,deltab):
    r"""
    Returns the perturbed escape rate of Ly \alpha photons per atom in the 2p state assuming a homogeneous perturbation in the matter density field, i.e. nh \to nh(1+deltab)
    and a homogeneous perturbation in the expansion of the fluid, i.e. H \to H + theta/3a
    """
    return ( 8 * np.pi * H(z)+(1+z)*theta/3 ) / ( 3 * nh(z)*(1+deltab) * cons.llya**3 * (1 - xe))

def Rlyapert(z,xe,eps,delta,theta,kind,bind,thetaind):
    r"""
    Returns the perturbed escape rate of Ly \alpha photons per atom in the 2p state assuming an inhomogeneous perturbation in the matter density field, i.e. nh \to nh(1+\epsilon*deltab)
    and an inhomogeneous perturbation in the expansion of the fluid, i.e. H \to H + epsilon*theta/3a. This is used to compute the perturbed ionization fraction with a local 3LA framework as in Lee and Ali-Haimoud 2021
    """
    return ( 8 * np.pi * (H(z) + eps*theta(z,kind,bind,thetaind)*(1+z)/3 ) ) / ( 3 * nh(z) * cons.llya**3 * (1 - xe) *(1 + eps*delta(z,kind,bind,thetaind)) )

def peebC(z,xe):
    r"""
    Returns the Peebles C-factor
    """
    return (3 * Rlya(z,xe) + cons.L2s1s ) / ( 3 * Rlya(z,xe) + cons.L2s1s + 4  * betab(z))

def peebChompert(z,xe,theta,deltab):
    r"""
    Returns the Peebles C-factor with a homogeneous perturbation in the matter density field and expansion of the fluid
    """
    return (3 * Rlyahompert(z,xe,theta,deltab) + cons.L2s1s ) / ( 3 * Rlyahompert(z,xe,theta,deltab) + cons.L2s1s + 4  * betab(z))

def peebCpert(z,xe,eps,delta,theta,kind,bind,thetaind):
    r"""
    Returns the Peebles C-factor with an inhomogeneous perturbation in the matter density field and expansion of the fluid. 
    This is used to compute the perturbed ionization fraction with a local 3LA framework as in Lee and Ali-Haimoud 2021
    """
    return (3 * Rlyapert(z,xe,eps,delta,theta,kind,bind,thetaind) + cons.L2s1s ) / ( 3 * Rlyapert(z,xe,eps,delta,theta,kind,bind,thetaind) + cons.L2s1s + 4  * betab(z))

## Define RHS of differential equation for ionization fraction for homogeneous case B recombination in the Sobolev approximation

def RHSsob(xe,z):
    r"""
    Returns the RHS of the recombination equation for the 3LA/Sobolev approximation
    """
    hold = ( peebC(z,xe) / ( (1+z) * H(z) ) ) * ( 
        nh(z) * xe**2  * alphab(z) - 4*(1-xe) *  betab(z) * np.exp(-(cons.en2 - cons.en1) / ( cons.kb * Tcmb(z) ) ) )
    return hold

def RHSsobhompert(xe,z,theta,deltab):
    r"""
    Returns the RHS of the recombination equation for the 3LA/Sobolev approximation with a homogeneous perturbation in the matter density field and expansion of the fluid
    """
    hold = ( peebChompert(z,xe,theta,deltab) / ( (1+z) * H(z) ) ) * ( 
        nh(z)*(1+deltab) * xe**2  * alphab(z) - 4*(1-xe) *  betab(z) * np.exp(-(cons.en2 - cons.en1) / ( cons.kb * Tcmb(z) ) ) )
    return hold

def RHSsobpert(xe,z,eps,delta,theta,kind,bind,thetaind):
    r"""
    Returns the RHS of the recombination equation for the 3LA/Sobolev approximation an inhomogeneous perturbation in the matter density field and expansion of the fluid. 
    This is used to compute the perturbed ionization fraction with a local 3LA framework as in Lee and Ali-Haimoud 2021
    """
    hold = ( peebCpert(z,xe,eps,delta,theta,kind,bind,thetaind) / ( (1+z) * (H(z) + eps*theta(z,kind,bind,thetaind)*(1+z)/3 ) ) ) * ( 
        nh(z)*(1+eps*delta(z,kind,bind,thetaind)) * xe**2  * alphab(z) - 4*(1-xe) *  betab(z) * np.exp(-(cons.en2 - cons.en1) / ( cons.kb * Tcmb(z) ) ) )
    return hold

def RHSsobpert2(xe,z,eps,delta,theta,kind,bind,thetaind):
    hold = ( peebCpert(z,xe,eps,delta,theta,kind,bind,thetaind) / ( (1+z) * H(z) ) ) * ( 
        nh(z)*(1+eps*delta(z,kind,bind,thetaind)) * xe**2  * alphab(z) - 4*(1-xe) *  betab(z) * np.exp(-(cons.en2 - cons.en1) / ( cons.kb * Tcmb(z) ) ) )
    return hold

def x2(z,xehomo):
    r"""
    Returns the fraction number of atoms in the n=2 state for the 3LA
    """
    return ( 4*( nh(z) * ( xehomo(z)**2 ) * alphab(z) + 
          ( 3 * Rlya(z, xehomo(z)) + cons.L2s1s )*( 1 - xehomo(z) )
          * np.exp( -( cons.en2 - cons.en1 ) / ( cons.kb * Tcmb(z) ) ) )
      / ( 3 * Rlya(z, xehomo(z) ) + cons.L2s1s + 4 * betab(z) ) )

def feq(z,xehomo):
    r"""
    Returns the fequilibrium PSD for the 3LA
    """
    return x2(z,xehomo)/( 4 * (1-xehomo(z)) )

### Create functions for computing the ionization fraction xe

def sahataylor(z):
    
    r"""
    Function for computing saha ionization fraction at very high redshifts (z \gapprox 2660). The regular saha result is not precise enough, so 
    the solution is given assuming that x_e = 1+\epsilon and then solving to lowest order in epsilon.
    """
    
    bofz = ( 2*np.pi*cons.me*cons.kb* Tcmb(z) )**1.5 / ( (cons.h**3) *  nh(z) ) * np.exp( cons.en1 / ( cons.kb * Tcmb(z) ) )
    
    return 1 - (1/bofz)


def sahapol(z):
    
    r"""
    Function for computing saha ionization fraction at lower redshifts (z \lapprox 2660).
    """
    
    bhold = ( 2*np.pi*cons.me*cons.kb* Tcmb(z) )**1.5 / ( (cons.h**3) *  nh(z) ) * np.exp( cons.en1 / ( cons.kb* Tcmb(z) ) )

    return (-bhold + np.sqrt(bhold**2 + 4* bhold ))/2



def xe_full(z):
    
    r"""
    Compute xe(z) assuming saha holds until redshift of 2300 and then use 3LA after. 
    """
    zstart = 2300
    xesaha = sahataylor(np.arange(3e4,zstart,-1) )

    xepeebs = odeint(RHSsob, sahataylor(zstart) , np.arange(zstart,200,-1))

    xe_hold = splrep(
        np.flip(np.concatenate([np.linspace(1e11,3.1e4,10000) , np.arange(3e4,200,-1) ])),
        np.flip(np.concatenate([np.ones(10000),xesaha,xepeebs[:,0]]))
    )
    
    return splev(z, xe_hold)

def xesaha_full(z):

    xesaha2 = sahataylor(np.arange(3e5,2660,-1) )
    xesaha3 = sahapol(np.arange(2660,200,-1))
    
    xesaha_hold = splrep( np.flip(np.arange(3e5,200,-1)),np.flip(np.append(xesaha2,xesaha3)) )
    
    return splev(z, xesaha_hold)


## Define useful functions for continuum contribution to the linearly perturbed ionization fraction
def Acon(z):
    r"""
    Returns A factor in \delta xe^{(1)}_{cont} equation as a function of redshift
    """
    return (1 - xe_full(z)) *  nh(z) *  sigmaa(z)

def B1con(z):
    r"""
    Returns B1 factor in \delta xe^{(1)}_{cont} equation as a function of redshift
    """
    return xe_full(z)**2 * nh(z) *  alpha1s(z)

def B2con(z): 
    r"""
    Returns B2 factor in \delta xe^{(1)}_{cont} equation as a function of redshift
    """
    return xe_full(z) * nh(z) *  alpha1s(z) * (2-xe_full(z))/(1-xe_full(z))

def R(z):
    r"""
    Returns ratio of baryon to photon energy density as a function of redshift
    """
    return (3/4) * ( cosmo.Ob0/cosmo.Ogamma0 ) / (1+z)

def Btild(B0):
    r"""
    Returns \tilde{B}_0n for a given mean field B_0 as defined in JKO 1998: \tilde{B}_0 = B_0/np.sqrt{4*pi*(\rho_gamma + p_gamma)} 
    """
    return B0 / np.sqrt(4*np.pi * (4/3) * cosmo.Ogamma0 * cosmo.critical_density0.value * cons.c**2 )

def lgamma(z, xe_full):
    r"""
    Returns the photon mean free path as a function of redshift and ionization fraction
    """
    return 1 / ( nh(z) * cons.sigmat * xe_full(z) )

def eta(z, xe_full):
    r"""
    Returns the diffusion coefficient as a function of redshift and ionization fraction
    """
    return (1/15) * lgamma(z, xe_full)

def Tevtoz(T):
    r"""
    Converts cmb temperture given in ev to redshift
    """
    return T / ( ( 8.617333262 * 10**(-5) ) * 2.7255 ) - 1;


def Hprime(z):
    r"""
    Returns the derivative of the Hubble parameter wrt to redshift, dH/dz
    """
    return nd.Derivative(H)(z)

def rhob(z):
    r"""
    Returns the baryon energy density as a function of redshift
    """
    return cosmo.Ob(z)*cosmo.critical_density(z).value

### Fluid ODEs: 
### All quantities are non-dimensionalized. We define Phi_x = ikv_x/H, Phi_y = ikv_y/H, Theta = ikv_z/H. 
### Transfer functions are computed by wrt the non-dimensional magnetic field b_i/B_0 which is set to 1 as an initial condition. 

### Tight-coupling regime 

def TCalf(z, vect, k, theta, Bin, xe_full):
    r"""
    System of equations for Alfven modes in the TCR
    """
    Phix, b = vect
    
    B0 = Btild(Bin)
        
    Phix2Phix = - ( Hprime(z)/ H(z) ) + (  H(z) * R(z) + ( 3 * cons.c * k**2 *(1+z)**2 * eta(z, xe_full) ) ) / (  H(z) * (1+z) * ( 1 + R(z) ) )
    Phix2b = (cons.c**2 * k**2/H(z)**2 ) * ( B0**2 * np.cos(theta) ) / ( 1 + R(z) ) 
        
    b2Phix = - np.cos(theta)
    
    return [( Phix2Phix * Phix + Phix2b *b), b2Phix*Phix]

def TCmag(z, vect, k, theta, Bin, xe_full):
    r"""
    System of equations for magnetosonic modes in the TCR
    """
    delta, Theta, Phiy, b = vect
    
    B0 = Btild(Bin)
    
    delta2Theta = 1/3
    
    Theta2delta = -(cons.c**2 * k**2 ) / (  H(z)**2 * ( 1 + R(z) ) )
    Theta2Theta = - ( Hprime(z)/ H(z) ) + (  H(z) * R(z) + ( 4 * cons.c * k**2 *(1+z)**2 * eta(z, xe_full) ) ) / (  H(z) * (1+z) * ( 1 + R(z) ) )
    Theta2b = - (cons.c**2 * k**2/H(z)**2 ) * ( B0**2 * np.sin(theta) ) / ( 1 + R(z) ) 
    
    Phiy2Phiy = - ( Hprime(z)/ H(z) ) + (  H(z) * R(z) + ( 3 * cons.c * k**2 *(1+z)**2 * eta(z, xe_full) ) ) / (  H(z) * (1+z) * ( 1 + R(z) ) )
    Phiy2b = (cons.c**2 * k**2/H(z)**2 ) * ( B0**2 * np.cos(theta) ) / ( 1 + R(z) ) 
        
    b2Theta =  np.sin(theta)
    b2Phiy = -np.cos(theta)
    
    return [( delta2Theta * Theta), (Theta2delta * delta + Theta2Theta * Theta + Theta2b *b ), (Phiy2Phiy * Phiy + Phiy2b * b ), (b2Theta * Theta + b2Phiy * Phiy ) ]


### Free-streaming regime

def FSRsahamag(z, vect, k, theta, Bin,xesaha_full):
    r"""
    System of equations for magnetosonic modes in the FSR assuming the ionization fraction is set by saha
    """
    delta, Theta, Phiy, b = vect
    
    B0 = Btild(Bin)
    
    xh = 1 - cons.yhe
    fhe = cons.yhe/(4.0*(1.0 - cons.yhe))
    rhob0 = cosmo.Ob0*cosmo.critical_density0.value

    delta2Theta = 1
    
    Theta2delta = (k**2/ H(z)**2)*( 
    (4.0*np.pi*cons.G*rhob0 * (1+z)/(k**2) ) - ((2.0 - fhe*xesaha_full(z))/(2.0 - xesaha_full(z)))*(1.0 - cons.yhe)*(
        cons.kb* Tcmb(z)/cons.mh) )
    Theta2Theta =  - ( Hprime(z)/ H(z) ) + (  H(z) +  f_lambda(z)*(1.0 - cons.yhe)*xesaha_full(z) ) / ( H(z)*(1+z) )
    Theta2b = - (cons.c**2 * k**2 * B0**2 * np.sin(theta) ) / (  H(z)**2 *  R(z)  )
    
    Phiy2Phiy = - ( Hprime(z)/ H(z) ) + (  H(z) +  f_lambda(z)*(1.0 - cons.yhe)*xesaha_full(z) ) / ( H(z)*(1+z) )
    Phiy2b = (cons.c**2 * k**2 * B0**2 * np.cos(theta) ) / (  H(z)**2 * R(z)  )
        
    b2Theta =  np.sin(theta)
    b2Phiy = -np.cos(theta)
    
    
    return [( delta2Theta * Theta), (Theta2delta * delta + Theta2Theta * Theta + Theta2b *b ), (Phiy2Phiy * Phiy + Phiy2b * b ), (b2Theta * Theta + b2Phiy * Phiy ) ]


def FSRsahaalf(z, vect, k, theta, Bin,xesaha_full):
    
    r"""
    System of equations for Alfven modes in the FSR assuming the ionization fraction is set by saha
    """
    B0 = Btild(Bin)
    Phix, bx = vect
    
    Phix2Phix =  - ( Hprime(z)/ H(z) ) + (  H(z) +  f_lambda(z)*(1.0 - cons.yhe)*xesaha_full(z) ) / ( H(z)*(1+z) )
    Phix2bx =  (cons.c**2 * k**2 * B0**2 * np.cos(theta) ) / (  H(z)**2 * R(z)  )
        
    bx2Phix = -np.cos(theta)
        
    return [
            ( Phix2Phix * Phix + ( Phix2bx ) * bx
                                               ), 
            bx2Phix * Phix ]



def FSRTLAmag(z, vect, k, kind, theta, Bin, xe_full, abarinterp, bbarinterp, cbarinterp, Acon, B1con, B2con):
    
    r"""
    System of equations for magnetosonic modes in the FSR assuming the ionization fraction is set by 3LA
    """

    delta, Theta, Phiy, b, dxe = vect
    
    rhob0 = cosmo.Ob0*cosmo.critical_density0.value
    B0 = Btild(Bin)

    xh = 1 - cons.yhe
    fhe = cons.yhe/(4.0*(1.0 - cons.yhe))
    
    def kappa(k,z):
        return 1 - Acon(z)/(k*(1+z)) * np.arctan((k*(1+z))/Acon(z))
    
    def pesc(kind,z):
        return (1 -  pab(z) * cbarinterp(z,kind) )/ (
            1 + (1 -  pab(z)) * cbarinterp(z,kind) )
    
    def P3LA(kind,z):
        return (3*cons.Alya * pesc(kind,z) + cons.L2s1s)/(
            3*cons.Alya * pesc(kind,z) + cons.L2s1s + 4* betab(z) )
    
    def sigA(kind,z):
        return 3*(1-P3LA(kind,z))*cons.Alya * ( abarinterp(z,kind) )/ (
            1 + (1 -  pab(z)) * cbarinterp(z,kind) )
    
    def sigB(kind,z):
        return 3*(1-P3LA(kind,z))*cons.Alya * ( bbarinterp(z,kind) )/ (
            1 + (1 -  pab(z)) * cbarinterp(z,kind) )
     

    delta2Theta =  1 

    Theta2delta = (4*np.pi*cons.G * rhob0 * (1+z) - ((xh * (1 + fhe + xe_full(z)) )*( 
        cons.kb* Tcmb(z)/cons.mh ) * k**2 ) )*(1/ H(z))**2 
    Theta2Theta = - ( Hprime(z)/ H(z) ) + ( H(z) +  f_lambda(z)*(1.0 - cons.yhe)*xe_full(z) ) / ( H(z) * (1+z) )
    Theta2b = - (cons.c**2 * k**2 * B0**2 * np.sin(theta) ) / (  H(z)**2 *  R(z)  )
    Theta2dxe = -(k**2/ H(z)**2)* xh * (cons.kb* Tcmb(z)/cons.mh)
    
    Phiy2Phiy = - ( Hprime(z)/ H(z) ) + ( H(z) +  f_lambda(z)*(1.0 - cons.yhe)*xe_full(z) ) / ( H(z) * (1+z) )
    Phiy2b = (k**2/ H(z)**2)* (cons.c**2 * B0**2 * np.cos(theta) ) / ( R(z)  )
        
    b2Theta =  np.sin(theta)
    b2Phiy = - np.cos(theta)

    dxe2Theta = -(1-xe_full(z))*sigB(kind,z)/ (H(z))
    dxe2delta = ( kappa(k,z)*B1con(z) + 
                P3LA(kind,z)* nh(z)*xe_full(z)**2 *  alphab(z) - 
                (1-xe_full(z))*sigA(kind,z) 
               )/( H(z)*(1+z))
    dxe2dxe = ( kappa(k,z)*B2con(z) - 
               ( 1-P3LA(kind,z))* nh(z)* (xe_full(z)**2/(1-xe_full(z))) *  alphab(z) +
               2*P3LA(kind,z)* nh(z)*xe_full(z)* alphab(z) +
               4* feq(z,xe_full)* betab(z)
               + sigA(kind,z)
              )/( H(z)*(1+z))
    
    return [( delta2Theta * Theta), (Theta2delta * delta + Theta2Theta * Theta + Theta2b *b + Theta2dxe*dxe ), 
          (Phiy2Phiy * Phiy + Phiy2b * b ), (b2Theta * Theta + b2Phiy * Phiy ),
          (dxe2delta * delta + dxe2Theta * Theta + dxe2dxe * dxe)
          ]


def FSRTLAalf(z, vect, k, theta, Bin, xe_full):
    r"""
    System of equations for Alfven modes in the FSR assuming the ionization fraction is set by 3LA
    """   
    B0 = Btild(Bin)
    Phix, bx = vect
    
    Phix2Phix = - ( Hprime(z)/ H(z) ) + ( H(z) +  f_lambda(z)*(1.0 - cons.yhe)*xe_full(z) ) / ( H(z) * (1+z) )
    Phix2bx =  (k**2/ H(z)**2) * (cons.c**2 * B0**2 * np.cos(theta) ) / ( R(z)  )
        
    bx2Phix = - np.cos(theta)
        
    return [
            ( Phix2Phix * Phix + ( Phix2bx ) * bx
                                               ), 
            bx2Phix * Phix ]


    
def FSRTLAmag_Tb(z, vect, k, kind, theta, Bin, xe_full, abarinterp, bbarinterp, cbarinterp, Tbhom, Acon, B1con, B2con):
    
    r"""
    System of equations for magnetosonic modes in the FSR assuming the ionization fraction is set by 3LA, with baryon heating included
    """

    delta, Theta, Phiy, b, dxe, Tb = vect
    
    rhob0 = cosmo.Ob0*cosmo.critical_density0.value
    B0 = Btild(Bin)

    xh = 1 - cons.yhe
    fhe = cons.yhe/(4.0*(1.0 - cons.yhe))
    # def alphab_Tb(z,Tb):
    # return (1e-13)*((4.309*(Tb(z)/10**4)**-0.6166)/(1 + 0.6703*(Tb(z)/10**4)**0.53))
    
    def kappa(k,z):
        return 1 - Acon(z)/(k*(1+z)) * np.arctan((k*(1+z))/Acon(z))
    
    def pesc(kind,z):
        return (1 -  pab(z) * cbarinterp(z,kind) )/ (
            1 + (1 -  pab(z)) * cbarinterp(z,kind) )
    
    def P3LA(kind,z):
        return (3*cons.Alya * pesc(kind,z) + cons.L2s1s)/(
            3*cons.Alya * pesc(kind,z) + cons.L2s1s + 4* betab(z) )
    
    def sigA(kind,z):
        return 3*(1-P3LA(kind,z))*cons.Alya * ( abarinterp(z,kind) )/ (
            1 + (1 -  pab(z)) * cbarinterp(z,kind) )
    
    def sigB(kind,z):
        return 3*(1-P3LA(kind,z))*cons.Alya * ( bbarinterp(z,kind) )/ (
            1 + (1 -  pab(z)) * cbarinterp(z,kind) )
     
    def dalphabpref(z):
        return -(0.6166 + 0.3553*(Tcmb(z)/10**4)**0.53/(1 + 0.6703*(Tcmb(z)/10**4)**0.53) )
    def Gammac(z):
        return 8*cons.arad*Tcmb(z)**4*xe_full(z)*cons.sigmat/( 3*cons.me*(1+xe_full(z) + fhe) )/cons.c
        # 8*cons.arad*xe(z)*parsf.Tcmb(z)**4*cons.sigmat/(3*cons.me * (1+xe(z)+fhe) )/cons.c
    def dGammacpref(z):
        return (1 + fhe)/(xe_full(z) * (1 + xe_full(z) + fhe ) )
    
    delta2Theta =  1 

    Theta2delta = (4*np.pi*cons.G * rhob0 * (1+z) - ((xh * (1 + fhe + xe_full(z)) )*( 
        cons.kb* Tcmb(z)/cons.mh ) * k**2 ) )*(1/ H(z))**2 
    Theta2Theta = - ( Hprime(z)/ H(z) ) + ( H(z) +  f_lambda(z)*(1.0 - cons.yhe)*xe_full(z) ) / ( H(z) * (1+z) )
    Theta2b = - (cons.c**2 * k**2 * B0**2 * np.sin(theta) ) / (  H(z)**2 *  R(z)  )
    Theta2dxe = -(k**2/ H(z)**2)* xh * (cons.kb* Tcmb(z)/cons.mh)
    
    Phiy2Phiy = - ( Hprime(z)/ H(z) ) + ( H(z) +  f_lambda(z)*(1.0 - cons.yhe)*xe_full(z) ) / ( H(z) * (1+z) )
    Phiy2b = (k**2/ H(z)**2)* (cons.c**2 * B0**2 * np.cos(theta) ) / ( R(z)  )
        
    b2Theta =  np.sin(theta)
    b2Phiy = - np.cos(theta)

    dxe2Theta = -(1-xe_full(z))*sigB(kind,z)/ (H(z))
    dxe2delta = ( kappa(k,z)*B1con(z) + 
                P3LA(kind,z)* nh(z)*xe_full(z)**2 *  alphab(z) - 
                (1-xe_full(z))*sigA(kind,z) 
               )/( H(z)*(1+z))
    dxe2dxe = ( kappa(k,z)*B2con(z) - 
               ( 1-P3LA(kind,z))* nh(z)* (xe_full(z)**2/(1-xe_full(z))) *  alphab(z) +
               2*P3LA(kind,z)* nh(z)*xe_full(z)* alphab(z) +
               4* feq(z,xe_full)* betab(z)
               + sigA(kind,z)
              )/( H(z)*(1+z))
    dxe2Tb = (-2*P3LA(kind,z)* nh(z)*xe_full(z)**2*alphab(z)/( H(z)*(1+z)) ) * dalphabpref(z)

    Tb2Tb =  Gammac(z)*(Tcmb(z)/Tbhom(z))/(H(z)*(1+z))
    Tb2Theta = (2/3)
    Tb2dxe = -dGammacpref(z) * Gammac(z) * (Tcmb(z)/Tbhom(z) - 1)/(H(z)*(1+z))
    
    return [( delta2Theta * Theta), (Theta2delta * delta + Theta2Theta * Theta + Theta2b *b + Theta2dxe*dxe ), 
          (Phiy2Phiy * Phiy + Phiy2b * b ), (b2Theta * Theta + b2Phiy * Phiy ),
          (dxe2delta * delta + dxe2Theta * Theta + dxe2dxe * dxe + dxe2Tb * Tb),
            (Tb2Tb * Tb + Tb2Theta * Theta + Tb2dxe * dxe)
          ]