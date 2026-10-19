import os, sys
from pathlib import Path

import numpy as np

from astropy.constants import codata2018 as const   # pinned to the set of cons.py (CODATA_SET)
from astropy.cosmology import Planck18 as cosmo

from . import cons

from scipy.special import voigt_profile
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from scipy.interpolate import splrep, splev, splint

# First-order Newtonian gravity (grad(varphi)/a in the FSR Euler equation, Poisson-sourced by the baryon density):
# PMHD_GRAVITY=0 switches it off in the FSR-Saha, TLA and T_b-TLA Theta equations, for the phi-subdominance
# test (paper_data_plots/phi_terms/PREREG.md). Default 1: multiplying by 1.0 is exact, so results are unchanged.
GRAVITY = float(os.environ.get("PMHD_GRAVITY", "1"))


# Tight-coupling density: TCmag evolves delta_gamma = delta T_gamma/T_gamma (Eq. 7a, d delta_gamma/dt = (1/3) d delta_b/dt),
# so the baryon density contrast on the TCR segment is delta_b = 3 delta_gamma. Consumers that read the TCRmag delta column
# as delta_b multiply by this factor (2026-10-12 fix; PMHD_TCR_DELTA_B=0 reproduces the pre-fix value 1).
TCR_DELTA_B = 3.0 if os.environ.get("PMHD_TCR_DELTA_B", "1") != "0" else 1.0


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

def H(z):
    r"""
    Returns hubble rate as a function of redshift, using astropy's Planck18
    cosmology directly (includes the massive-neutrino component that a
    manual flat-LCDM formula would omit).
    """
    return cosmo.H(z).to('1/s').value


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

def alphabT(T):
    r"""Case-B recombination coefficient as a function of temperature (Pequignot et al. fit, as alphab)."""
    return (1e-13)*((4.309*(T/10**4)**-0.6166)/(1 + 0.6703*(T/10**4)**0.53))

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

def RHSTbhom(Tm,z,xe):
    fhe = cons.yhe/(4.0*(1.0 - cons.yhe))
    gammaT = 8*cons.arad*xe(z)*Tcmb(z)**4*cons.sigmat/(3*cons.me * (1+xe(z)+fhe) )/cons.c
    return (
    2*Tm/(1+z) - gammaT/H(z) * (Tcmb(z) - Tm)/(1+z)
    )

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



# --- module-level cached splines (built once at import time) ---
_xe_zstart = 2300
_xe_saha_arr = sahataylor(np.arange(3e4, _xe_zstart, -1))
_xe_zgrid = np.arange(_xe_zstart, 200, -1)
def _xe_peebles_tight():
    """Homogeneous Peebles solve with tight tolerances (third audit, finding 1): the line departure f_eq - f_bb is a
    near-cancellation at z >= 1750 and inherits the ~1e-8 noise of the default odeint solve. Cached next to the tables;
    PMHD_XE_BG_OLD=1 restores the default-tolerance solve."""
    from scipy.integrate import solve_ivp
    sol = solve_ivp(lambda z, y: RHSsob(y, z), [_xe_zstart, 200], [sahataylor(_xe_zstart)], method="Radau",
                    rtol=1e-11, atol=1e-13, max_step=1.0, dense_output=True)
    return sol.sol(_xe_zgrid)[0]
_xe_bg_cache = Path(__file__).resolve().parent / "data/pre_stored_data/xe_peebs_tight.npy"
_xe_bg_provenance = _xe_bg_cache.with_suffix(".json")


def _bg_fingerprint():
    """What the cached background depends on besides the code: the constants set and the values entering the Peebles
    equation (PREREG addendum 17, item 3)."""
    return {"codata_set": cons.CODATA_SET, "h": cons.h, "kb": cons.kb, "me": cons.me, "en1": cons.en1, "c": cons.c,
            "H_1100": float(H(1100.0)), "nh_1100": float(nh(1100.0)), "Tcmb_1100": float(Tcmb(1100.0)),
            "solver": "Radau rtol=1e-11 atol=1e-13 max_step=1 from z=2300 (sahataylor) to z=201, unit steps"}


def _check_bg_fingerprint(recorded, current, rtol=1e-12):
    """Raise unless the cached background was built with the constants and cosmology in use now."""
    bad = [k for k in current if k not in recorded or (
        recorded[k] != current[k] if isinstance(current[k], str)
        else abs(recorded[k] - current[k]) > rtol*abs(current[k]))]
    if bad:
        raise RuntimeError(f"The cached background {_xe_bg_cache.name} was built with different inputs ({', '.join(bad)}): "
                           f"recorded {[recorded.get(k) for k in bad]}, now {[current[k] for k in bad]}. The moment "
                           "tables are built on that background; changing constants or cosmology requires rebuilding the "
                           "background and every table (PREREG addendum 17).")


if os.environ.get("PMHD_XE_BG_OLD", "0") != "0":
    _xe_peebs = odeint(RHSsob, sahataylor(_xe_zstart), _xe_zgrid)
elif _xe_bg_cache.exists():
    import json as _json
    if not _xe_bg_provenance.exists():
        raise RuntimeError(f"{_xe_bg_provenance} is missing: the background cache has no provenance record")
    _check_bg_fingerprint(_json.loads(_xe_bg_provenance.read_text()), _bg_fingerprint())
    _xe_peebs = np.load(_xe_bg_cache)[:, None]
else:
    import json as _json
    _xe_peebs = _xe_peebles_tight()[:, None]
    try:
        np.save(_xe_bg_cache, _xe_peebs[:, 0])
        _xe_bg_provenance.write_text(_json.dumps(_bg_fingerprint(), indent=1))
    except OSError:
        pass
_xe_hold = splrep(
    np.flip(np.concatenate([np.linspace(1e11, 3.1e4, 10000), np.arange(3e4, 200, -1)])),
    np.flip(np.concatenate([np.ones(10000), _xe_saha_arr, _xe_peebs[:, 0]]))
)

_xesaha2 = sahataylor(np.arange(3e5, 2660, -1))
_xesaha3 = sahapol(np.arange(2660, 200, -1))
_xesaha_hold = splrep(
    np.flip(np.concatenate([np.linspace(1e11, 3.1e5, 10000), np.arange(3e5, 200, -1)])),
    np.flip(np.concatenate([np.ones(10000), _xesaha2, _xesaha3]))
)

def xe_full(z):
    r"""
    Compute xe(z) assuming saha holds until redshift of 2300 and then use 3LA after.
    """
    return splev(z, _xe_hold)

def xesaha_full(z):
    return splev(z, _xesaha_hold)


### Helium electron contribution from HyRec --------------------------------
# The hydrogen-only ionization curves above (xe_full, xesaha_full) plateau at 1
# at high z. Helium contributes up to 2 extra electrons per He nucleus, so the
# total xe = n_e/n_H exceeds 1 before He recombines. The helium electron
# contribution is taken directly from a HyRec run (data/pre_stored_data/
# output_xe.dat, total xe = H+He). In the band where He is ionized (z>~1800)
# hydrogen is fully ionized, so the He electron fraction is simply (xe_total-1);
# it crosses zero at z~1707 where He goes neutral and H begins to recombine.

_hyrec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'data', 'pre_stored_data', 'output_xe.dat')
_hyrec_dat = np.loadtxt(_hyrec_path)
# file is z-descending; reverse to ascending z for splrep
_hyrec_z = _hyrec_dat[::-1, 0]
_hyrec_xe = _hyrec_dat[::-1, 1]
_z_lo_He = _hyrec_z[0]                       # 1.0
_z_hi_He = _hyrec_z[-1]                      # 8000.0
_xeHe_tab = np.clip(_hyrec_xe - 1.0, 0.0, None)
_xeHe_spl = splrep(_hyrec_z, _xeHe_tab)
_xeHe_hi = _xeHe_tab[-1]                      # = 2 fHe (He fully doubly ionized), ~0.1634

def xe_He(z):
    r"""
    Helium electron contribution to xe = n_e/n_H, taken from HyRec
    (data/pre_stored_data/output_xe.dat). Equals 2 fHe (~0.163) at z > 8000
    (He fully doubly ionized), steps down through fHe as HeIII->HeII->HeI
    recombine, and vanishes for z below ~1707 where helium is neutral.
    """
    z = np.asarray(z, dtype=float)
    out = np.where(z > _z_hi_He, _xeHe_hi,
                   np.where(z < _z_lo_He, 0.0, splev(z, _xeHe_spl)))
    return out[()] if out.ndim == 0 else out

def xe_full_He(z):
    r"""
    Total xe(z) = n_e/n_H with hydrogen via the 3-level atom (as in xe_full)
    plus the helium contribution from HyRec (xe_He).
    """
    return xe_full(z) + xe_He(z)

def xesaha_full_He(z):
    r"""
    Total xe(z) = n_e/n_H with hydrogen via Saha (as in xesaha_full) plus the
    helium contribution from HyRec (xe_He).
    """
    return xesaha_full(z) + xe_He(z)


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
    return xe_full(z)**2 * nh(z) *  (alpha1s(z) - alphab(z))

def B2con(z):
    r"""
    Returns B2 factor in \delta xe^{(1)}_{cont} equation as a function of redshift
    """
    return xe_full(z) * nh(z) *  (alpha1s(z) - alphab(z)) * (2-xe_full(z))/(1-xe_full(z))

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
    Returns the physical photon mean free path as a function of redshift and ionization fraction
    """
    return 1 / ( nh(z) * cons.sigmat * xe_full(z) )

# def eta(z, xe_full):
#     r"""
#     Returns the diffusion coefficient as a function of redshift and ionization fraction
#     """
#     return (1/15) * lgamma(z, xe_full)

def eta(z, xe_full):
    r"""
    Returns the diffusion coefficient as a function of redshift and ionization fraction
    """
    return (4/45) * lgamma(z, xe_full)

def Tevtoz(T):
    r"""
    Converts cmb temperture given in ev to redshift
    """
    return T / ( ( 8.617333262 * 10**(-5) ) * 2.7255 ) - 1;


def Hprime(z):
    r"""
    Returns the derivative of the Hubble parameter wrt to redshift, dH/dz
    """
    # dH/dz of the H used everywhere else (astropy Planck18, massive neutrinos): central difference, relative step 1e-5
    # (truncation ~1e-10, round-off ~1e-11). The pre-2026-10-12 closed form (matter + massless radiation) differed from it by
    # up to 4e-4 at z = 600 (PMHD_HPRIME_OLD=1 restores it).
    if os.environ.get("PMHD_HPRIME_OLD", "0") != "0":
        orad = cosmo.Ogamma0 * (1 + cons.Neff * (7.0/8.0) * np.power(4.0/11.0, 4.0/3.0))
        return cons.H0**2 * (3 * cons.omat * (1+z)**2 + 4 * orad * (1+z)**3) / (2 * H(z))
    d = 1e-5
    return (H(z*(1 + d)) - H(z*(1 - d)))/(2*z*d)

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

def FSRsahamag_hom(z, vect, k, theta, Bin, xesaha_full, xesaha_full_He):
    r"""
    Magnetosonic FSR equations for the ionization fraction set by Saha,
    helium-consistent ("hom"). xe is split by physical role:
      - the baryon pressure/sound-speed factor is written as the explicit sum
            (1 + fhe + x_e^tot) - x_H(1-x_H)/(2-x_H),
        i.e. helium free electrons are included in the homogeneous particle count
        (x_e^tot = xesaha_full_He) but the perturbed-ionization (Saha) response
        uses the hydrogen ionization fraction (x_H = xesaha_full); helium's own
        delta x is neglected;
      - the Thomson-drag terms use the He-inclusive free-electron fraction.
    """
    delta, Theta, Phiy, b = vect

    B0 = Btild(Bin)

    xh = 1 - cons.yhe
    fhe = cons.yhe/(4.0*(1.0 - cons.yhe))
    rhob0 = cosmo.Ob0*cosmo.critical_density0.value

    delta2Theta = 1

    Theta2delta = (k**2/ H(z)**2)*(
    (GRAVITY*4.0*np.pi*cons.G*rhob0 * (1+z)/(k**2) ) - ((1.0 + fhe + xesaha_full_He(z)) - xesaha_full(z)*(1.0 - xesaha_full(z))/(2.0 - xesaha_full(z)))*(1.0 - cons.yhe)*(
        cons.kb* Tcmb(z)/cons.mh) )
    Theta2Theta =  - ( Hprime(z)/ H(z) ) + (  H(z) +  f_lambda(z)*(1.0 - cons.yhe)*xesaha_full_He(z) ) / ( H(z)*(1+z) )
    Theta2b = - (cons.c**2 * k**2 * B0**2 * np.sin(theta) ) / (  H(z)**2 *  R(z)  )

    Phiy2Phiy = - ( Hprime(z)/ H(z) ) + (  H(z) +  f_lambda(z)*(1.0 - cons.yhe)*xesaha_full_He(z) ) / ( H(z)*(1+z) )
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




def FSRTLAmag(z, vect, k, kind, theta, Bin, xe_full, xe_full_He, abarinterp, bbarinterp, cbarinterp, Acon, B1con, B2con):

    r"""
    System of equations for magnetosonic modes in the FSR assuming the
    ionization fraction is set by 3LA. xe is split by physical role:
      - xe_full_He (He-inclusive free-electron fraction) in the homogeneous
        pressure particle count (1+fhe+xe) and the Thomson-drag terms;
      - xe_full (hydrogen ionization) in the entire perturbed-ionization (dxe)
        equation and recombination coefficients (Acon/B1con/B2con, alphab, betab,
        feq, P3LA, sigA/sigB, the (1-xe) and xe^2 factors).
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

    # He-inclusive: homogeneous pressure particle count (He electrons are real particles)
    Theta2delta = (GRAVITY*4*np.pi*cons.G * rhob0 * (1+z) - ((xh * (1 + fhe + xe_full_He(z)) )*(
        cons.kb* Tcmb(z)/cons.mh ) * k**2 ) )*(1/ H(z))**2
    # He-inclusive: Thomson drag (drag ~ total n_e)
    Theta2Theta = - ( Hprime(z)/ H(z) ) + ( H(z) +  f_lambda(z)*(1.0 - cons.yhe)*xe_full_He(z) ) / ( H(z) * (1+z) )
    Theta2b = - (cons.c**2 * k**2 * B0**2 * np.sin(theta) ) / (  H(z)**2 *  R(z)  )
    Theta2dxe = -(k**2/ H(z)**2)* xh * (cons.kb* Tcmb(z)/cons.mh)

    Phiy2Phiy = - ( Hprime(z)/ H(z) ) + ( H(z) +  f_lambda(z)*(1.0 - cons.yhe)*xe_full_He(z) ) / ( H(z) * (1+z) )
    Phiy2b = (k**2/ H(z)**2)* (cons.c**2 * B0**2 * np.cos(theta) ) / ( R(z)  )

    b2Theta =  np.sin(theta)
    b2Phiy = - np.cos(theta)

    # hydrogen ionization throughout the perturbed-ionization (dxe) equation
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

def FSRTLAmag_Tb(z, vect, k, kind, theta, Bin, xe_full, xe_full_He, abarinterp, bbarinterp, cbarinterp, Tbhom, Acon, B1con, B2con):

    r"""
    System of equations for magnetosonic modes in the FSR assuming the
    ionization fraction is set by 3LA, with baryon heating included. xe is
    split by role:
      - xe_full_He (He-inclusive free electrons): homogeneous pressure (1+fhe+xe),
        Thomson drag, and Compton heating (Gammac, dGammacpref);
      - xe_full (hydrogen ionization): recombination / perturbed-ionization (dxe)
        block and Acon/B1con/B2con.
    """

    delta, Theta, Phiy, b, dxe, Tb = vect

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

    def dalphabpref(z):
        a=4.309; b=-0.6166; c=0.6703; d=0.53
        return (b + (b-d)*c*(Tbhom(z)/10**4)**d )/(1 + c*(Tbhom(z)/10**4)**d)

    # Third audit, finding 4 (PREREG addenda 12 and 16): the pressure coefficient uses the homogeneous T_b, and so does
    # the alpha_B prefactor of the delta alpha_B response term (dxe2Tb). The background-balance coefficients (dxe2delta,
    # dxe2dxe) keep alpha_B(T_gamma), the value the background x_e and f_eq are solved with: their recombination and
    # photoionisation parts cancel to 1 part in ~500 at z ~ 1400, so a 1e-6 offset of alpha_B would be amplified into a
    # spurious ~5e-4 change. PMHD_TB_COEFF_TG=1 restores T_gamma everywhere.
    if os.environ.get("PMHD_TB_COEFF_TG", "0") != "0":
        Tcoef = Tcmb(z); alphab_c = alphab(z)
    else:
        Tcoef = Tbhom(z); alphab_c = alphabT(Tbhom(z))

    # Compton heating: He-inclusive free electrons couple the gas to photons
    def Gammac(z):
        return 8*cons.arad*Tcmb(z)**4*xe_full_He(z)*cons.sigmat/( 3*cons.me*(1+xe_full_He(z) + fhe) )/cons.c
    def dGammacpref(z):
        return (1 + fhe)/(xe_full_He(z) * (1 + xe_full_He(z) + fhe ) )

    delta2Theta =  1

    # He-inclusive: homogeneous pressure particle count
    Theta2delta = (GRAVITY*4*np.pi*cons.G * rhob0 * (1+z) - ((xh * (1 + fhe + xe_full_He(z)) )*(
        cons.kb* Tcoef/cons.mh ) * k**2 ) )*(1/ H(z))**2
    Theta2Tb = (- ((xh * (1 + fhe + xe_full_He(z)) )*(
        cons.kb* Tcoef/cons.mh ) * k**2 ) )*(1/ H(z))**2
    # He-inclusive: Thomson drag
    Theta2Theta = - ( Hprime(z)/ H(z) ) + ( H(z) +  f_lambda(z)*(1.0 - cons.yhe)*xe_full_He(z) ) / ( H(z) * (1+z) )
    Theta2b = - (cons.c**2 * k**2 * B0**2 * np.sin(theta) ) / (  H(z)**2 *  R(z)  )
    Theta2dxe = -(k**2/ H(z)**2)* xh * (cons.kb* Tcoef/cons.mh)

    Phiy2Phiy = - ( Hprime(z)/ H(z) ) + ( H(z) +  f_lambda(z)*(1.0 - cons.yhe)*xe_full_He(z) ) / ( H(z) * (1+z) )
    Phiy2b = (k**2/ H(z)**2)* (cons.c**2 * B0**2 * np.cos(theta) ) / ( R(z)  )

    b2Theta =  np.sin(theta)
    b2Phiy = - np.cos(theta)

    # hydrogen ionization throughout the recombination / dxe block
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
    dxe2Tb = (P3LA(kind,z)* nh(z)*xe_full(z)**2*alphab_c/( H(z)*(1+z)) ) * dalphabpref(z)

    Tb2Tb =  Gammac(z)*(Tcmb(z)/Tbhom(z))/(H(z)*(1+z))
    Tb2Theta = (2/3)
    Tb2dxe = -dGammacpref(z) * Gammac(z) * (Tcmb(z)/Tbhom(z) - 1)/(H(z)*(1+z))

    return [( delta2Theta * Theta), (Theta2delta * delta + Theta2Theta * Theta + Theta2b *b + Theta2dxe*dxe + Theta2Tb * Tb ),
          (Phiy2Phiy * Phiy + Phiy2b * b ), (b2Theta * Theta + b2Phiy * Phiy ),
          (dxe2delta * delta + dxe2Theta * Theta + dxe2dxe * dxe + dxe2Tb * Tb),
            (Tb2Tb * Tb + Tb2Theta * Theta + Tb2dxe * dxe)
          ]