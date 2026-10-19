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
from pmhd.data.tables import tables_dir
from pmhd.data.grids import (
    k_grid, z_grid, eps_grid,
    load_or_generate_B0arr
)
from pmhd.physics.source_assembly import (
    build_abg, unpack_f2bars, assemble_at_z, splint_reducer,
    assemble_dop_at_z, doppler_terms_enabled, grav11_enabled,
    BAR_KEYS, CROSS_CORR_KEYS,
)



def main(input_bind, input_epsind):
    print(time.ctime())
    karr = k_grid()
    B0arr = load_or_generate_B0arr()
    zarr = z_grid()

    epsarr = eps_grid()

    xe_full = pars.xe_full

    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    with open(tables_dir()/'abarinterpmaster.pkl', 'rb') as f:
        abarinterpmaster = pickle.load(f)

    with open(tables_dir()/'bbarinterpmaster.pkl', 'rb') as f:
        bbarinterpmaster = pickle.load(f)

    with open(tables_dir()/'cbarinterpmaster.pkl', 'rb') as f:
        cbarinterpmaster = pickle.load(f)

    def abarinterpfunc(z,kind):
        return splev(z, abarinterpmaster[kind])

    def bbarinterpfunc(z,kind):
        return splev(z, bbarinterpmaster[kind])

    def cbarinterpfunc(z,kind):
        return splev(z, cbarinterpmaster[kind])


    ## alpha/beta/gamma prefactors -- now built by pmhd.physics.source_assembly.build_abg so
    ## that the single-mode path (single_mode_source.py) shares the identical construction.
    _abg = build_abg(zarr, karr, abarinterpfunc, bbarinterpfunc, cbarinterpfunc, xe_full)
    alphaeq, betaeq, gammaeq = _abg['alphaeq'], _abg['betaeq'], _abg['gammaeq']
    alpha00, beta00, gamma00 = _abg['alpha00'], _abg['beta00'], _abg['gamma00']
    alphabar, betabar, gammabar = _abg['alphabar'], _abg['betabar'], _abg['gammabar']

    paball = pars.pab(zarr)
    pscall = 1- paball
    xearr = xe_full(zarr)


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
    fgrav11 = np.zeros(len(zarr))    # Eq. (54) gravitational-redshift term (manuscript sync 2026-10)
    # rest-frame-monopole (Doppler/aberration) terms, Eqs. (420)+(416), (421), (417)
    use_dop = doppler_terms_enabled()
    fdop = np.zeros(len(zarr)); fdopbar = np.zeros(len(zarr)); fv2 = np.zeros(len(zarr))
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
    
    

    OUTBASE = Path(os.environ.get("PMHD_OUTDIR", str(PROJECT_ROOT / "data/outputs")))
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
    
    reduce_ps = splint_reducer(karr, epsarr[input_epsind])

    for zind in range(len(zarr)):
        with open(tables_dir()/f'f2bars_dict/f2bars_m0_dict_zch_{zind}.pkl', 'rb') as f:
                f2bars_m0_master = pickle.load(f)
        with open(tables_dir()/f'f2bars_dict/f2bars_m1_dict_zch_{zind}.pkl', 'rb') as f:
                f2bars_m1_master = pickle.load(f)
        with open(tables_dir()/f'f2bars_dict/f2bars_p1_dict_zch_{zind}.pkl', 'rb') as f:
                f2bars_p1_master = pickle.load(f)

        moms = unpack_f2bars(f2bars_m0_master, f2bars_m1_master, f2bars_p1_master)
        bars = {key: ang_avg_dict_master[key][:, zind] for key in BAR_KEYS}
        abg  = {key: _abg[key][zind, :] for key in _abg}
        bg   = dict(z=zarr[zind], H=H[zind], taus=tausall[zind],
                    pab=paball[zind], xe=xearr[zind], Deltah=Deltah[zind])

        cc_row, fred11[zind], fproc[zind], fadveclens[zind], fred20[zind], fgrav11[zind] = assemble_at_z(
            bars, abg, moms, bg, reduce_ps)
        if use_dop:
            dop = assemble_dop_at_z(bars, abg, moms, cc_row, bg, reduce_ps)
            fdop[zind], fdopbar[zind], fv2[zind] = dop['fdop'], dop['fdopbar'], dop['fv2']
        for key in CROSS_CORR_KEYS:
            cross_corr[key][zind] = cc_row[key]

    # fred20 (Eq. red_20) is now included alongside fred11/fproc/fadveclens.
    ftot = fred11 + fproc + fadveclens + fred20
    if use_dop:
        ftot = ftot + fdop + fdopbar + fv2
    if grav11_enabled():
        ftot = ftot + fgrav11

    # Ensure directory exists
    # Output base directory
    OUTBASE = Path(os.environ.get("PMHD_OUTDIR", str(PROJECT_ROOT / "data/outputs")))

    source_fncs = OUTBASE/"source_fncs"
    cross_corrsave = OUTBASE/"cross_corr"
    source_fncs.mkdir(parents=True, exist_ok=True)
    cross_corrsave.mkdir(parents=True, exist_ok=True)

    with open(cross_corrsave/f'cross_corr_B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}.pkl', 'wb') as f:
            pickle.dump(cross_corr, f)

    np.save(source_fncs/f'ftot_B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}.npy',ftot)
    # individual source components, for diagnostics
    np.savez(source_fncs/f'fcomp_B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}.npz',
             fred11=fred11, fproc=fproc, fadveclens=fadveclens, fred20=fred20,
             fdop=fdop, fdopbar=fdopbar, fv2=fv2, doppler_terms=use_dop, fgrav11=fgrav11)

if __name__=='__main__':
    print(time.ctime())
        
    main(int(sys.argv[1]), int(sys.argv[2]))

    print(time.ctime())

    