import sys
from pathlib import Path

# Path to lib/
LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))
import numpy as np

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

    zsteps = 10**4

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    OUTBASE = PROJECT_ROOT / "data/outputs"

    # TCR normalization comes from standard (non-Tb) Tfs dir
    TfsBdir = OUTBASE / f"Tfs/B_{round(1e12*B0arr[input_bind])}pG"
    # Tb TFs from Tfs/Tb/
    TbBdir = OUTBASE / f"Tfs/Tb/B_{round(1e12*B0arr[input_bind])}pG"

    diffalf_Tf_b = np.load(TfsBdir / f"TCRalf_k{input_kind}.npy")[:,1,-1].reshape(-1,1)

    alfhold = np.load(TbBdir / f"FSRTLAalf_Tb_k{input_kind}.npy")
    bxhold = diffalf_Tf_b*alfhold[:,1,:]
    Phixhold = diffalf_Tf_b*alfhold[:,0,:]

    diffmag_Tf_b = np.load(TfsBdir / f"TCRmag_k{input_kind}.npy")[:,3,-1].reshape(-1,1)

    maghold = np.load(TbBdir / f"FSRTLAmag_Tb_k{input_kind}.npy")
    byhold    = diffmag_Tf_b*maghold[:,3,:]
    Phiyhold  = diffmag_Tf_b*maghold[:,2,:]
    Thetahold = diffmag_Tf_b*maghold[:,1,:]
    deltahold = diffmag_Tf_b*maghold[:,0,:]
    dxehold   = diffmag_Tf_b*maghold[:,4,:]
    Tbhold    = diffmag_Tf_b*maghold[:,5,:]

    bx    = np.zeros( ( len(zarr), len(thetaarr) ) )
    Phix  = np.zeros( ( len(zarr), len(thetaarr) ) )
    by    = np.zeros( ( len(zarr), len(thetaarr) ) )
    Phiy  = np.zeros( ( len(zarr), len(thetaarr) ) )
    delta = np.zeros( ( len(zarr), len(thetaarr) ) )
    Theta = np.zeros( ( len(zarr), len(thetaarr) ) )
    xebar = np.zeros( ( len(zarr), len(thetaarr) ) )
    Tb    = np.zeros( ( len(zarr), len(thetaarr) ) )

    for thetaind in range(int((len(thetaarr)-1)/2)+1):
        zinterp = np.logspace(np.log10(zstartarr[input_kind]),np.log10(600), num=zsteps)[::-1]
        startzind = int(1900 - math.floor(zstartarr[input_kind]))

        bx[startzind:, thetaind] = splev(zarr[startzind:],
                                     splrep( zinterp, bxhold[thetaind,::-1] ), der=0 )
        Phix[startzind:, thetaind] = pars.H(zarr[startzind:])*splev(zarr[startzind:],
                                     splrep( zinterp, Phixhold[thetaind,::-1] ), der=0 )
        by[startzind:, thetaind] = splev(zarr[startzind:],
                                     splrep( zinterp, byhold[thetaind,::-1] ), der=0 )
        Phiy[startzind:, thetaind] = pars.H(zarr[startzind:])*splev(zarr[startzind:],
                                     splrep( zinterp, Phiyhold[thetaind,::-1] ), der=0 )
        delta[startzind:, thetaind] = splev(zarr[startzind:],
                                     splrep( zinterp, deltahold[thetaind,::-1] ), der=0 )
        Theta[startzind:, thetaind] = pars.H(zarr[startzind:])*splev(zarr[startzind:],
                                     splrep( zinterp, Thetahold[thetaind,::-1] ), der=0 )
        xebar[startzind:, thetaind] = splev(zarr[startzind:],
                                     splrep( zinterp, dxehold[thetaind,::-1] ), der=0 )
        Tb[startzind:, thetaind] = splev(zarr[startzind:],
                                     splrep( zinterp, Tbhold[thetaind,::-1] ), der=0 )

    if zstartarr[input_kind] != 1900:
        # Fill TCR regime from standard TCR files; Tb and xebar are zero in tight coupling
        alfhold_tcr = np.load(TfsBdir / f"TCRalf_k{input_kind}.npy")
        bxhold_tcr   = alfhold_tcr[:,1,:]
        Phixhold_tcr = alfhold_tcr[:,0,:]

        maghold_tcr   = np.load(TfsBdir / f"TCRmag_k{input_kind}.npy")
        byhold_tcr    = maghold_tcr[:,3,:]
        Phiyhold_tcr  = maghold_tcr[:,2,:]
        Thetahold_tcr = maghold_tcr[:,1,:]
        deltahold_tcr = maghold_tcr[:,0,:]

        for thetaind in range(int((len(thetaarr)-1)/2)+1):
            zinterp = np.logspace(np.log10(zcrossarr[input_kind]),np.log10(zfsarr[input_kind]), num=zsteps)[::-1]
            startzind = int(1900 - math.floor(zstartarr[input_kind]))

            bx[:startzind, thetaind] = splev(zarr[:startzind],
                                         splrep( zinterp, bxhold_tcr[thetaind,::-1] ), der=0 )
            Phix[:startzind, thetaind] = pars.H(zarr[:startzind])*splev(zarr[:startzind],
                                         splrep( zinterp, Phixhold_tcr[thetaind,::-1] ), der=0 )
            by[:startzind, thetaind] = splev(zarr[:startzind],
                                         splrep( zinterp, byhold_tcr[thetaind,::-1] ), der=0 )
            Phiy[:startzind, thetaind] = pars.H(zarr[:startzind])*splev(zarr[:startzind],
                                         splrep( zinterp, Phiyhold_tcr[thetaind,::-1] ), der=0 )
            delta[:startzind, thetaind] = splev(zarr[:startzind],
                                         splrep( zinterp, deltahold_tcr[thetaind,::-1] ), der=0 )
            Theta[:startzind, thetaind] = pars.H(zarr[:startzind])*splev(zarr[:startzind],
                                         splrep( zinterp, Thetahold_tcr[thetaind,::-1] ), der=0 )
            # xebar and Tb remain zero in TCR regime (tight-coupling drives both to zero)

    # Symmetry fill under theta -> pi - theta
    for i in range(int((len(thetaarr)-1)/2)+1):
        bx[:,   -i+int((len(thetaarr)-1))] =  bx[:,   i]   # EVEN
        by[:,   -i+int((len(thetaarr)-1))] =  by[:,   i]   # EVEN
        Phix[:, -i+int((len(thetaarr)-1))] =  Phix[:, i]   # EVEN
        Phiy[:, -i+int((len(thetaarr)-1))] =  Phiy[:, i]   # EVEN
        delta[:,-i+int((len(thetaarr)-1))] = -delta[:,i]   # ODD
        Theta[:,-i+int((len(thetaarr)-1))] = -Theta[:,i]   # ODD
        xebar[:,-i+int((len(thetaarr)-1))] = -xebar[:,i]   # ODD
        Tb[:,   -i+int((len(thetaarr)-1))] = -Tb[:,   i]   # ODD

    ang_avg_dict = {}
    for k in [
        'bxbxbar', 'bybybar', 'PhixPhixbar', 'PhiyPhiybar',
        'ThetaThetabar', 'deltamThetabar', 'deltamdeltambar',
        'xexebar', 'xedeltambar', 'xeThetabar',
        'TbTbbar', 'Tbxebar', 'Tbdeltambar', 'TbThetabar'
    ]:
        ang_avg_dict[k] = np.zeros(len(zarr))

    print(time.ctime())

    for z in range(len(zarr)):
        ang_avg_dict['bxbxbar'][z] = splint(
            thetaarr[0], thetaarr[-1], splrep(
                thetaarr, np.sin(thetaarr) * bx[z,:]**2 ) )
        ang_avg_dict['bybybar'][z] = splint(
            thetaarr[0], thetaarr[-1], splrep(
                thetaarr, np.sin(thetaarr) * by[z,:]**2 ) )
        ang_avg_dict['PhixPhixbar'][z] = splint(
            thetaarr[0], thetaarr[-1], splrep(
                thetaarr, np.sin(thetaarr) * Phix[z,:]**2 ) )
        ang_avg_dict['PhiyPhiybar'][z] = splint(
            thetaarr[0], thetaarr[-1], splrep(
                thetaarr, np.sin(thetaarr) * Phiy[z,:]**2 ) )
        ang_avg_dict['ThetaThetabar'][z] = splint(
            thetaarr[0], thetaarr[-1], splrep(
                thetaarr, np.sin(thetaarr) * Theta[z,:]**2 ) )
        ang_avg_dict['deltamdeltambar'][z] = splint(
            thetaarr[0], thetaarr[-1], splrep(
                thetaarr, np.sin(thetaarr) * delta[z,:]**2 ) )
        ang_avg_dict['deltamThetabar'][z] = splint(
            thetaarr[0], thetaarr[-1], splrep(
                thetaarr, np.sin(thetaarr) * delta[z,:]*Theta[z,:] ) )
        ang_avg_dict['xexebar'][z] = splint(
            thetaarr[0], thetaarr[-1], splrep(
                thetaarr, np.sin(thetaarr) * xebar[z,:]**2 ) )
        ang_avg_dict['xedeltambar'][z] = splint(
            thetaarr[0], thetaarr[-1], splrep(
                thetaarr, np.sin(thetaarr) * xebar[z,:]*delta[z,:] ) )
        ang_avg_dict['xeThetabar'][z] = splint(
            thetaarr[0], thetaarr[-1], splrep(
                thetaarr, np.sin(thetaarr) * xebar[z,:]*Theta[z,:] ) )
        ang_avg_dict['TbTbbar'][z] = splint(
            thetaarr[0], thetaarr[-1], splrep(
                thetaarr, np.sin(thetaarr) * Tb[z,:]**2 ) )
        ang_avg_dict['Tbxebar'][z] = splint(
            thetaarr[0], thetaarr[-1], splrep(
                thetaarr, np.sin(thetaarr) * Tb[z,:]*xebar[z,:] ) )
        ang_avg_dict['Tbdeltambar'][z] = splint(
            thetaarr[0], thetaarr[-1], splrep(
                thetaarr, np.sin(thetaarr) * Tb[z,:]*delta[z,:] ) )
        ang_avg_dict['TbThetabar'][z] = splint(
            thetaarr[0], thetaarr[-1], splrep(
                thetaarr, np.sin(thetaarr) * Tb[z,:]*Theta[z,:] ) )

    ang_avg_Tb = OUTBASE / f"ang_avg/Tb/B_{round(1e12*B0arr[input_bind])}pG"
    ang_avg_Tb.mkdir(parents=True, exist_ok=True)

    with open(ang_avg_Tb / f'ang_avg_k{input_kind}_Tb.pkl', 'wb') as f:
        pickle.dump(ang_avg_dict, f)

if __name__ == '__main__':
    print(time.ctime())

    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", type=int, default=0)
    parser.add_argument("--kind", type=int, default=0)

    args = parser.parse_args()

    main(args.bind, args.kind)

print('job done', time.ctime())
