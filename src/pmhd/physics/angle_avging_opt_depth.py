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
    k_grid, z_grid,
    theta_grid,
    theta_gridfull,
    load_or_generate_z_arrays,
    load_or_generate_B0arr,
)

import math

def main(input_bind, input_kind):
    print(time.ctime())
    karr = k_grid()
    zarr = z_grid()
    thetaarr = theta_gridfull()

    zcrossarr, zfsarr = load_or_generate_z_arrays()
    zstartarr = np.append( np.full(np.argwhere(zfsarr<=1900)[0,0],1900), zfsarr[np.argwhere(zfsarr<=1900)[0,0]:] )

    B0arr = load_or_generate_B0arr()

    xe_full = pars.xe_full

    zsteps = 10**4

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    # Output base directory
    OUTBASE = PROJECT_ROOT / "data/outputs"
    # For this run, create subdir per B0
    Bdir = OUTBASE / f"Tfs/B_{round(1e12*B0arr[input_bind])}pG"
    
    diffalf_Tf_b = np.load(Bdir / f"TCRalf_k{input_kind}.npy")[:,1,-1].reshape(17,1)

    diffmag_Tf_b = np.load(Bdir / f"TCRmag_k{input_kind}.npy")[:,3,-1].reshape(17,1)
    
    maghold = np.load(Bdir / f"FSRTLAmag_k{input_kind}.npy")
    byhold = diffmag_Tf_b*maghold[:,3,:]
    Phiyhold = diffmag_Tf_b*maghold[:,2,:]
    Thetahold = diffmag_Tf_b*maghold[:,1,:]
    deltahold = diffmag_Tf_b*maghold[:,0,:]
    dxehold = diffmag_Tf_b*maghold[:,4,:]

    dtaufin = np.zeros( ( len(zarr), len(thetaarr) ) )
    d_taudotfin = np.zeros( ( len(zarr), len(thetaarr) ) )

    def deltafunc(z,thetaind):
        hold = splrep(np.logspace(np.log10(zstartarr[input_kind]),np.log10(600),10**4)[::-1],deltahold[thetaind,::-1])
        return splev(z, hold)
    def dxefunc(z,thetaind):
        hold = splrep(np.logspace(np.log10(zstartarr[input_kind]),np.log10(600),10**4)[::-1],dxehold[thetaind,::-1])
        return splev(z, hold)
    def dtauinteg(z,xe,thetaind):
        return cons.c*pars.nh(z)*cons.sigmat*(xe(z)*deltafunc(z,thetaind)+dxefunc(z,thetaind) )/pars.H(z)/(1+z)
    def d_taudot(z,xe,thetaind):
        return (deltafunc(z,thetaind)+dxefunc(z,thetaind)/xe(z) )
    
    dtau = np.zeros((10**4,17))
    dtauhold = np.zeros((10**4,17))
    
    for thetaind in range(int((len(thetaarr)-1)/2)+1):
        zinterp = np.logspace(np.log10(600),np.log10(zstartarr[input_kind]), num = zsteps)
        zmid = (zinterp[1:] + zinterp[:-1])/2

        dtauintegarr = dtauinteg(zinterp,xe_full,thetaind)
        dtauintegarrmid = dtauinteg(zmid,xe_full,thetaind)
        
        dtauhold[1:,thetaind] = ((zinterp[1:] - zinterp[:-1])/6)*(dtauintegarr[:-1] + 4*dtauintegarrmid[:] + dtauintegarr[1:])
        dtau[:,thetaind] = np.cumsum(dtauhold[:,thetaind])
        
        
        startzind = int(1900 - math.floor(zstartarr[input_kind]))

        
        dtaufin[startzind:, thetaind] = splev(zarr[startzind:],
                                     splrep( zinterp, dtau[:,thetaind] ), der =0 )
        d_taudotfin[startzind:, thetaind] = splev(zarr[startzind:],
                                     splrep( zinterp, d_taudot(zinterp,xe_full,thetaind) ), der =0 )
        
    
    if zstartarr[input_kind]!=1900:
    
        maghold = np.load(Bdir / f"TCRmag_k{input_kind}.npy")
        byhold = maghold[:,3,:]
        Phiyhold = maghold[:,2,:]
        Thetahold = maghold[:,1,:]
        deltahold = maghold[:,0,:]

        def deltafunc(z,thetaind):
            hold = splrep(np.logspace(np.log10(zfsarr[input_kind]),np.log10(zcrossarr[input_kind]),10**4),deltahold[thetaind,::-1])
            return splev(z, hold)
        def dxefunc(z,thetaind):
            hold = splrep(np.logspace(np.log10(zfsarr[input_kind]),np.log10(zcrossarr[input_kind]),10**4),dxehold[thetaind,::-1])
            return splev(z, hold)
        def dtauinteg(z,xe,thetaind):
            return cons.c*pars.nh(z)*cons.sigmat*(xe(z)*deltafunc(z,thetaind) )/pars.H(z)/(1+z)
        def d_taudot(z,xe,thetaind):
            return (deltafunc(z,thetaind) )
        
        dtau = np.zeros((10**4,17))
        dtauhold = np.zeros((10**4,17))


        for thetaind in range(int((len(thetaarr)-1)/2)+1):
            zinterp = np.logspace(np.log10(zfsarr[input_kind]),np.log10(zcrossarr[input_kind]), num = zsteps)
            zmid = (zinterp[1:] + zinterp[:-1])/2
            
            startzind = int(1900 - math.floor(zstartarr[input_kind]))
    
            dtauintegarr = dtauinteg(zinterp,xe_full,thetaind)
            dtauintegarrmid = dtauinteg(zmid,xe_full,thetaind)
            
            dtauhold[1:,thetaind] = ((zinterp[1:] - zinterp[:-1])/6)*(dtauintegarr[:-1] + 4*dtauintegarrmid[:] + dtauintegarr[1:])
            dtau[:,thetaind] = np.cumsum(dtauhold[:,thetaind])
            
            
            startzind = int(1900 - math.floor(zstartarr[input_kind]))
    
            
            dtaufin[:startzind, thetaind] = splev(zarr[:startzind],
                                         splrep( zinterp, dtau[:,thetaind] ), der =0 )
            d_taudotfin[:startzind, thetaind] = splev(zarr[:startzind],
                                         splrep( zinterp, d_taudot(zinterp,xe_full,thetaind) ), der =0 )
        


    for i in range(int((len(thetaarr)-1)/2)+1):
        dtaufin[:,-i+int((len(thetaarr)-1))] = -dtaufin[:,i]
        d_taudotfin[:,-i+int((len(thetaarr)-1))] = -d_taudotfin[:,i]



    dtaudtaubar = np.zeros( len(zarr) ) 
    dtaudtaudotbar = np.zeros( len(zarr) ) 
    

    print(time.ctime()) 

    for z in range(len(zarr)):
        
        dtaudtaubar[z] = splint(
            thetaarr[0],thetaarr[-1], splrep(
                thetaarr, np.sin(thetaarr)  * dtaufin[z,:]**2 ) )
        dtaudtaudotbar[z] = splint(
            thetaarr[0],thetaarr[-1], splrep(
                thetaarr, np.sin(thetaarr)  * dtaufin[z,:]*d_taudotfin[z,:] ) )
        
        
    ang_avg_TLA = OUTBASE/f"ang_avg/TLA/B_{round(1e12*B0arr[input_bind])}pG"
    ang_avg_TLA.mkdir(parents=True, exist_ok=True)
        
    # Save results
    np.save(ang_avg_TLA/f'dtaudtaubar_k{input_kind}.npy', dtaudtaubar)

    np.save(ang_avg_TLA/f'dtaudtaudotbar_k{input_kind}.npy', dtaudtaudotbar)

if __name__=='__main__':
    print(time.ctime())

    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", type=int, default=0)
    parser.add_argument("--kind", type=int, default=0)

    args = parser.parse_args()

    main(args.bind, args.kind)
        
print('job done',time.ctime())

