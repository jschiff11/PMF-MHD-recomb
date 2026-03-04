import os
import numpy as np
from scipy import optimize as opt
from .. import cons
from .. import pars

def k_grid(kmin=20, kmax=26.9, nk=69):
    return 2*np.pi/np.logspace(kmin, kmax, nk, endpoint = False)

def eps_grid(epsmin=-0.01, epsmax=-1, num =100):
    return np.linspace(epsmin,epsmax,num)

def z_grid(zin=1900, zfin=600, stepsize=-1):
    return np.arange(zin,zfin,stepsize)

def theta_grid(thetamin = 0, thetamax=np.pi/2, thetanum=17):
    return np.linspace(thetamin,thetamax,thetanum)

def theta_gridfull(thetamin = 0, thetamax=np.pi, thetanum=33):
    return np.linspace(thetamin,thetamax,thetanum)

def B0_grid(Bmax, Bmin, nB):
    return np.logspace(np.log10(Bmin), np.log10(Bmax), nB)

# -----------------------------
# paths for saved arrays
# -----------------------------
data_dir = os.path.join(os.path.dirname(__file__), 'pre_stored_data')
zc_path = os.path.join(data_dir, 'zcrossarr.npy')
zf_path = os.path.join(data_dir, 'zfsarr.npy')
B0_path = os.path.join(data_dir, 'B0arr.npy')


# -----------------------------
# function to regenerate arrays if needed
# -----------------------------
def generate_B_arr(Barr = None):
    """Compute B0arr and save to data/"""
    if Barr is None:
        Barr = np.concatenate((5*np.logspace(-9,-12,101)[:20],5*np.logspace(-9,-12,101)[20::2])) 

    os.makedirs(data_dir, exist_ok=True)
    np.save(B0_path, Barr)
    return Barr

def generate_z_arrays(karr=None, lamarr=None):
    """Compute zcrossarr and zfsarr and save to data/"""
    if karr is None:
        karr = k_grid()
    if lamarr is None:
        lamarr = 2*np.pi / karr  # or your real formula

    zcrossarr = np.zeros(len(karr))
    for kind in range(len(karr)):
        def horizcross(z):
            return cons.c*(1+z)/pars.H(z) - lamarr[kind]
        zcrossarr[kind] = opt.brentq(horizcross, 0, 1e11)

    zfsarr = np.zeros(len(karr))
    for kind in range(len(karr)):
        def fscross(z):
            if z >= 3e4:
                hold = (1+z)// ( pars.nh(z) * cons.sigmat ) - lamarr[kind]
            else:
                hold = pars.lgamma(z,pars.xe_full)*(1+z) - lamarr[kind]
            return hold
        zfsarr[kind] = opt.brentq(fscross, 0, 1e11)

    os.makedirs(data_dir, exist_ok=True)
    np.save(zc_path, zcrossarr)
    np.save(zf_path, zfsarr)
    return zcrossarr, zfsarr


# -----------------------------
# function to load arrays (and optionally regenerate)
# -----------------------------
def load_or_generate_z_arrays(force=False):
    if not os.path.exists(zc_path) or not os.path.exists(zf_path) or force:
        return generate_z_arrays()
    zcrossarr = np.load(zc_path)
    zfsarr = np.load(zf_path)
    return zcrossarr, zfsarr

def load_or_generate_B0arr(force=False):
    if not os.path.exists(B0_path) or force:
        return generate_B_arr()
    Barr = np.load(B0_path)
    return Barr

# # -----------------------------
# # load by default on import
# # -----------------------------
# zcrossarr, zfsarr = load_or_generate_z_arrays()
# B0arr = load_or_generate_B0arr()
