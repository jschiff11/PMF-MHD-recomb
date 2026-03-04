import numpy as np

from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo


## Rydberg in astropy measured in 1/m as 1/wavelength of lyman continuum photon
en1 = -(const.Ryd*const.h*const.c).to('erg').value # Ground state ionization energy
en2 = en1/4 # 2p excitation energy 



# A few fundamental constants
h = const.h.cgs.value       # Planck's constant
hbar = h/2*np.pi            # Reduced Planck's constant
kb = const.k_B.cgs.value    # Boltzmann's constant
c = 29979245800             # Speed of light
amu = 1.66053892e-24        # Atomic mass unit
mp = 1.67262178e-24         # Proton mass
me = const.m_e.cgs.value    # Electron mass
na = 6.0221413e+23          # Avogadro's constant
e = 4.80320425e-10          # Electron charge
G = 6.67384e-8              # Newton's gravitational constant

# A few derived and measured quantities
mh = 1.00794*amu            # Mass of Hydrogen atom
nuhf = 1420.40575177e+6     # Frequency of the 1s hyperfine transition
Thf = h*nuhf/kb             # 1s hyperfine gap in temperature units
Ahf = 2.86888e-15           # Einstein A coeff. for the 1s hyperfine transition

llya = (4/(3*(const.Ryd))).cgs.value        # Wavelength of Lyman-alpha photon
nuly = c/llya                # Lyalpha frequency
Alya = 6.2648e+8            # Einstein A coeff. for the Lyman-alpha transition
glya = Alya/(4*np.pi)     # HWHM for Lyman lines, in Hz

A3d2p = 6.4651e+7 	    # Einstein A coeff. for 3d-2p
A3s2p = 6.3143e+6	    # Einstein A coeff. for 3s-2p
L2s1s = 8.22                # Two photon decay rate from 2s-1s

gammae = 1.760859708e7      # gyromagnetic moment of electron in s^{-1}G^{-1}
sigmat = 0.6652458734e-24   # Thomson cross section
arad = 7.565767e-15	    # Radiation constant
ev = 1.6021772e-12	    # 1 Electron volt

mpc = 3.08567758e+24        # 1 megaparsec

T0 = cosmo.Tcmb0.value      # CMB temperature at z=0, Ref: Fixsen (2009)
H0 = cosmo.H0.to('1/s').value            # Hubble rate at z=0
obh2 = cosmo.Ob0*cosmo.h**2             # Omega_b h^2
omat = cosmo.Om0            # Omega_m
olda = cosmo.Ode0            # Omega_lambda
yhe = 0.2454                # Helium mass fraction
Neff = cosmo.Neff           # Effective neutrino number, theoretical value

## Initialize redshift and linewidth vectors
z_array = np.linspace(2000,200,1801) # redshift range

## Integration parameters if one would like to find sobolev PSD using discretized ODE
width = 1000 # linewidth
bins = 50 # bins per line width
xlen = (bins*2*width)+1 #number of steps
xpm5 = np.linspace(-width - .1,width + .1,xlen + 10) #integration range with pm 5 steps on each side
step = 0.02

x_array = np.linspace(-width,width, xlen) # integration range

k0 = 0.05 		    # Planck pivot scale (in Mpc^{-1}) for curvature power spectrum
Ak0 = 2.105e-9	    # Curvature power at pivot scale (dimensionless)
ns = 0.9665		    # Running index for power with k, ns = 1 is SI

