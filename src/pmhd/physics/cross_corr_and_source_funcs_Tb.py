# pyright: ignore[reportMissingImports]
import numpy as np
import os
import time
import sys
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))

from scipy.interpolate import splrep, splev, splint
from astropy.cosmology import Planck18 as cosmo
import pickle

from pmhd import cons, pars
from pmhd.data.tables import tables_dir
from pmhd.physics.source_assembly import (
    unpack_f2bars, assemble_dop_at_z, doppler_terms_enabled, splint_reducer,
    grav11_enabled, b30_old_coefficient, advlens_old_coefficient, dalpha_feq_enabled,
)
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

    xe_full = pars.xe_full
    xearr = xe_full(zarr)

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    OUTBASE = Path(os.environ.get("PMHD_OUTDIR", str(PROJECT_ROOT / "data/outputs")))

    with open(tables_dir()/'abarinterpmaster.pkl', 'rb') as f:
        abarinterpmaster = pickle.load(f)
    with open(tables_dir()/'bbarinterpmaster.pkl', 'rb') as f:
        bbarinterpmaster = pickle.load(f)
    with open(tables_dir()/'cbarinterpmaster.pkl', 'rb') as f:
        cbarinterpmaster = pickle.load(f)

    def abarinterpfunc(z, kind): return splev(z, abarinterpmaster[kind])
    def bbarinterpfunc(z, kind): return splev(z, bbarinterpmaster[kind])
    def cbarinterpfunc(z, kind): return splev(z, cbarinterpmaster[kind])

    alphaeq  = np.zeros((len(zarr), len(karr)))
    betaeq   = np.zeros((len(zarr), len(karr)))
    gammaeq  = np.zeros((len(zarr), len(karr)))
    alpha00  = np.zeros((len(zarr), len(karr)))
    beta00   = np.zeros((len(zarr), len(karr)))
    gamma00  = np.zeros((len(zarr), len(karr)))
    alphabar = np.zeros((len(zarr), len(karr)))
    betabar  = np.zeros((len(zarr), len(karr)))
    gammabar = np.zeros((len(zarr), len(karr)))
    # delta_alphaB = pref(z) delta_Tb in delta f_eq / delta fbar_00 (source_assembly.dalpha_feq_enabled); coefficients
    # per (z, k), already multiplied by pref(z) = dln alpha_B/dln T_b at the homogeneous T_b.
    epsT = np.zeros((len(zarr), len(karr))); eps00T = np.zeros((len(zarr), len(karr))); epsbarT = np.zeros((len(zarr), len(karr)))
    use_dalpha = dalpha_feq_enabled()
    from scipy.integrate import odeint
    _zh = np.arange(1900, 600, -0.01); _sol = odeint(pars.RHSTbhom, pars.Tcmb(1900), _zh, args=(pars.xe_full_He,)).flatten()
    _Tbhom = splev(zarr, splrep(_zh[::-1], _sol[::-1]))
    if use_dalpha:
        _a, _b, _c, _d = 4.309, -0.6166, 0.6703, 0.53     # Eq. (A42)-(A43a), as in FSRTLAmag_Tb / xe2_Tb
        pref = (_b + (_b - _d)*_c*(_Tbhom/1e4)**_d)/(1 + _c*(_Tbhom/1e4)**_d)

    paball    = pars.pab(zarr)
    pscall    = 1 - paball
    # third audit, finding 4 (PREREG addenda 12, 16): the delta f_eq linearisation coefficients use alpha_B(T_gamma),
    # consistent with the background f_eq the tables are built on; alpha_B(T_b) only in the delta alpha_B response epsT
    # (PMHD_TB_COEFF_TG=1 restores T_gamma there too)
    alphaball = pars.alphab(zarr)
    alphab_resp = pars.alphab(zarr) if os.environ.get("PMHD_TB_COEFF_TG", "0") != "0" else pars.alphabT(_Tbhom)
    betaball  = pars.betab(zarr)
    nhall     = pars.nh(zarr)

    for k in range(len(karr)):
        denom = (3*cons.Alya*(1 - cbarinterpfunc(zarr,k)*paball)/(1 + cbarinterpfunc(zarr,k)*pscall) + cons.L2s1s + 4*betaball)
        alphaeq[:,k] = (nhall*(xearr**2/(1-xearr))*alphaball + 3*cons.Alya*abarinterpfunc(zarr,k)/(1 + cbarinterpfunc(zarr,k)*pscall)) / denom
        betaeq[:,k]  = (nhall*xearr**2/(1-xearr)*alphaball + 2*nhall*alphaball*xearr - 3*cons.Alya*abarinterpfunc(zarr,k)/(1 + cbarinterpfunc(zarr,k)*pscall)) / denom
        gammaeq[:,k] = (3*cons.Alya*bbarinterpfunc(zarr,k)/(1 + cbarinterpfunc(zarr,k)*pscall)) / denom

        cbar_over_1pc = cbarinterpfunc(zarr,k) / (1 + cbarinterpfunc(zarr,k)*pscall)
        abar_over_1pc = abarinterpfunc(zarr,k) / (1 + cbarinterpfunc(zarr,k)*pscall)
        bbar_over_1pc = bbarinterpfunc(zarr,k) / (1 + cbarinterpfunc(zarr,k)*pscall)

        alpha00[:,k] =  abar_over_1pc + alphaeq[:,k]*cbar_over_1pc
        beta00[:,k]  = -abar_over_1pc + betaeq[:,k]*cbar_over_1pc
        gamma00[:,k] =  bbar_over_1pc + gammaeq[:,k]*cbar_over_1pc

        alphabar[:,k] = alphaeq[:,k] - pscall*alpha00[:,k]
        if use_dalpha:
            epsT[:,k]    = pref * (nhall*(xearr**2/(1-xearr))*alphab_resp) / denom      # delta f_eq   coefficient of delta_Tb
            eps00T[:,k]  = epsT[:,k]*cbar_over_1pc                                     # delta fbar_00 coefficient
            epsbarT[:,k] = epsT[:,k] - pscall*eps00T[:,k]                              # (delta f_eq - p_sc delta fbar_00)
        betabar[:,k]  = -betaeq[:,k] + pscall*beta00[:,k]
        gammabar[:,k] = gammaeq[:,k] - pscall*gamma00[:,k]

    tausall = pars.taus(zarr, xe_full)
    Deltah  = pars.Deltah(zarr)
    H       = pars.H(zarr)

    # Load Tb ang_avg main quantities (computed from 6-variable Tb ODE)
    ang_avg_Tb_dir = OUTBASE / f"ang_avg/Tb/B_{round(1e12*B0arr[input_bind])}pG"
    Tb_main_keys = [
        'bxbxbar', 'bybybar', 'PhixPhixbar', 'PhiyPhiybar',
        'ThetaThetabar', 'deltamThetabar', 'deltamdeltambar',
        'xexebar', 'xedeltambar', 'xeThetabar',
        'TbTbbar', 'Tbxebar', 'Tbdeltambar', 'TbThetabar',
    ]
    ang_avg_Tb = {k: np.zeros((len(karr), len(zarr))) for k in Tb_main_keys}
    for kind in range(len(karr)):
        with open(ang_avg_Tb_dir / f'ang_avg_k{kind}_Tb.pkl', 'rb') as f:
            d = pickle.load(f)
        for k in Tb_main_keys:
            ang_avg_Tb[k][kind] = d[k]

    # Load non-Tb ang_avg for prime quantities absent from Tb pkl
    ang_avg_TLA_dir = OUTBASE / f"ang_avg/TLA/B_{round(1e12*B0arr[input_bind])}pG"
    prime_keys = ['PhixPhixprimebar', 'PhiyPhiyprimebar', 'ThetaThetaprimebar']
    ang_avg_prime = {k: np.zeros((len(karr), len(zarr))) for k in prime_keys}
    for kind in range(len(karr)):
        with open(ang_avg_TLA_dir / f'ang_avg_k{kind}.pkl', 'rb') as f:
            d = pickle.load(f)
        for k in prime_keys:
            ang_avg_prime[k][kind] = d[k]

    cross_corr = {}
    for k in [
        'df00xecross', 'dfeqxecross', 'df00deltamcross',
        'dfeqdeltamcross', 'deltamxecross', 'xeThetacross',
        'xerms', 'deltamrms', 'vxrms', 'vyrms', 'vzrms',
        'Tbrms', 'Tbxecross', 'Tbdeltacross', 'TbThetacross', 'deltaThetacross',
    ]:
        cross_corr[k] = np.zeros(len(zarr))

    fred11     = np.zeros(len(zarr))
    fred20     = np.zeros(len(zarr))
    fproc      = np.zeros(len(zarr))
    fadveclens = np.zeros(len(zarr))
    fgrav11    = np.zeros(len(zarr))   # Eq. (54) gravitational-redshift term (manuscript sync 2026-10)
    # rest-frame-monopole (Doppler/aberration) terms -- shared implementation in source_assembly;
    # the pre-existing inline terms above are left untouched so regression mode stays bit-identical
    use_dop = doppler_terms_enabled()
    fdop = np.zeros(len(zarr)); fdopbar = np.zeros(len(zarr)); fv2 = np.zeros(len(zarr))
    reduce_dop = splint_reducer(karr, epsarr[input_epsind])

    for zind in range(len(zarr)):
        with open(tables_dir()/f'f2bars_dict/f2bars_m0_dict_zch_{zind}.pkl', 'rb') as f:
            f2bars_m0 = pickle.load(f)
        with open(tables_dir()/f'f2bars_dict/f2bars_m1_dict_zch_{zind}.pkl', 'rb') as f:
            f2bars_m1 = pickle.load(f)
        with open(tables_dir()/f'f2bars_dict/f2bars_p1_dict_zch_{zind}.pkl', 'rb') as f:
            f2bars_p1 = pickle.load(f)

        A2 = f2bars_m0['mom2_total'][:,0]; B2 = f2bars_m0['mom2_total'][:,1]; C2 = f2bars_m0['mom2_total'][:,2]
        D2_m = f2bars_m1['mom2_total'][:,0]; D2_p = f2bars_p1['mom2_total'][:,0]
        dx_A0 = f2bars_m0['dx_mom0_total'][:,0]; dx_B0 = f2bars_m0['dx_mom0_total'][:,1]; dx_C0 = f2bars_m0['dx_mom0_total'][:,2]
        dx_A2 = f2bars_m0['dx_mom2_total'][:,0]; dx_B2 = f2bars_m0['dx_mom2_total'][:,1]; dx_C2 = f2bars_m0['dx_mom2_total'][:,2]
        dx_D2_m = f2bars_m1['dx_mom2_total'][:,0]; dx_D2_p = f2bars_p1['dx_mom2_total'][:,0]
        dx_A1 = f2bars_m0['dx_mom1_total'][:,0]; dx_B1 = f2bars_m0['dx_mom1_total'][:,1]; dx_C1 = f2bars_m0['dx_mom1_total'][:,2]
        dx_phi_dx_A0 = f2bars_m0['dx_phi_dx_mom0_total'][:,0]
        dx_phi_dx_B0 = f2bars_m0['dx_phi_dx_mom0_total'][:,1]
        dx_phi_dx_C0 = f2bars_m0['dx_phi_dx_mom0_total'][:,2]
        phi_A0 = f2bars_m0['phi_mom0_total'][:,0]; phi_B0 = f2bars_m0['phi_mom0_total'][:,1]; phi_C0 = f2bars_m0['phi_mom0_total'][:,2]
        A1 = f2bars_m0['mom1_total'][:,0]; B1 = f2bars_m0['mom1_total'][:,1]; C1 = f2bars_m0['mom1_total'][:,2]
        taus_phi = f2bars_m0['taus_phi'][0, 0]

        def _k_int(integrand):
            return splint(karr[-1], karr[0], splrep(np.flip(karr), np.flip(integrand)))

        eps = epsarr[input_epsind]
        z   = zarr[zind]
        xe  = xearr[zind]

        cross_corr['deltamrms'][zind]    = _k_int(karr**(eps-1) * ang_avg_Tb['deltamdeltambar'][:,zind])
        cross_corr['vxrms'][zind]        = _k_int(karr**(eps-3) * ang_avg_Tb['PhixPhixbar'][:,zind])
        cross_corr['vyrms'][zind]        = _k_int(karr**(eps-3) * ang_avg_Tb['PhiyPhiybar'][:,zind])
        cross_corr['vzrms'][zind]        = _k_int(karr**(eps-3) * ang_avg_Tb['ThetaThetabar'][:,zind])
        cross_corr['xerms'][zind]        = _k_int(karr**(eps-1) * ang_avg_Tb['xexebar'][:,zind])
        cross_corr['deltamxecross'][zind]= _k_int(karr**(eps-1) * ang_avg_Tb['xedeltambar'][:,zind])
        cross_corr['xeThetacross'][zind] = _k_int(karr**(eps-1) * ang_avg_Tb['xeThetabar'][:,zind])

        cross_corr['df00xecross'][zind] = _k_int(karr**(eps-1) * (
            alpha00[zind,:]*ang_avg_Tb['xedeltambar'][:,zind] +
            (beta00[zind,:]/(1-xe))*ang_avg_Tb['xexebar'][:,zind] +
            (gamma00[zind,:]*(1+z)/H[zind])*ang_avg_Tb['xeThetabar'][:,zind]
            + eps00T[zind,:]*ang_avg_Tb['Tbxebar'][:,zind]
        ))
        cross_corr['dfeqxecross'][zind] = _k_int(karr**(eps-1) * (
            alphaeq[zind,:]*ang_avg_Tb['xedeltambar'][:,zind] +
            (betaeq[zind,:]/(1-xe))*ang_avg_Tb['xexebar'][:,zind] +
            (gammaeq[zind,:]*(1+z)/H[zind])*ang_avg_Tb['xeThetabar'][:,zind]
            + epsT[zind,:]*ang_avg_Tb['Tbxebar'][:,zind]
        ))
        cross_corr['df00deltamcross'][zind] = _k_int(karr**(eps-1) * (
            alpha00[zind,:]*ang_avg_Tb['deltamdeltambar'][:,zind] +
            (beta00[zind,:]/(1-xe))*ang_avg_Tb['xedeltambar'][:,zind] +
            (gamma00[zind,:]*(1+z)/H[zind])*ang_avg_Tb['deltamThetabar'][:,zind]
            + eps00T[zind,:]*ang_avg_Tb['Tbdeltambar'][:,zind]
        ))
        cross_corr['dfeqdeltamcross'][zind] = _k_int(karr**(eps-1) * (
            alphaeq[zind,:]*ang_avg_Tb['deltamdeltambar'][:,zind] +
            (betaeq[zind,:]/(1-xe))*ang_avg_Tb['xedeltambar'][:,zind] +
            (gammaeq[zind,:]*(1+z)/H[zind])*ang_avg_Tb['deltamThetabar'][:,zind]
            + epsT[zind,:]*ang_avg_Tb['Tbdeltambar'][:,zind]
        ))

        # Tb-specific integrated quantities
        cross_corr['Tbrms'][zind]       = _k_int(karr**(eps-1) * ang_avg_Tb['TbTbbar'][:,zind])
        cross_corr['Tbxecross'][zind]   = _k_int(karr**(eps-1) * ang_avg_Tb['Tbxebar'][:,zind])
        cross_corr['Tbdeltacross'][zind]= _k_int(karr**(eps-1) * ang_avg_Tb['Tbdeltambar'][:,zind])
        cross_corr['TbThetacross'][zind]= _k_int(karr**(eps-1) * ang_avg_Tb['TbThetabar'][:,zind])
        cross_corr['deltaThetacross'][zind] = _k_int(karr**(eps-1) * ang_avg_Tb['deltamThetabar'][:,zind])

        # Source function terms — Tb main quantities + non-Tb prime quantities
        fred11[zind] = _k_int(karr**(eps-1) * (
            (-(1+z)**2/(5*H[zind]**2)) * (
                ang_avg_Tb['PhixPhixbar'][:,zind] + ang_avg_Tb['PhiyPhiybar'][:,zind]
            ) * (dx_D2_p + dx_D2_m)
            + ((1+z)/(3*H[zind])) * (
                ang_avg_Tb['deltamThetabar'][:,zind] - (1/(1-xe))*ang_avg_Tb['xeThetabar'][:,zind]
            ) * ((2/5)*dx_A2 - dx_A0)
            + ((1+z)**2/(3*H[zind]**2)) * ang_avg_Tb['ThetaThetabar'][:,zind] * ((2/5)*dx_B2 - dx_B0)
            + ((1+z)/(3*H[zind])) * (
                alphabar[zind,:]*ang_avg_Tb['deltamThetabar'][:,zind] -
                (betabar[zind,:]/(1-xe))*ang_avg_Tb['xeThetabar'][:,zind] +
                (gammabar[zind,:]*(1+z)/H[zind])*ang_avg_Tb['ThetaThetabar'][:,zind] +
                epsbarT[zind,:]*ang_avg_Tb['TbThetabar'][:,zind]
            ) * ((2/5)*dx_C2 - dx_C0)
        ))

        fproc[zind] = _k_int(karr**(eps-1) * (
            (ang_avg_Tb['deltamdeltambar'][:,zind] + (1/(1-xe))**2*ang_avg_Tb['xexebar'][:,zind] -
             (2/(1-xe))*ang_avg_Tb['xedeltambar'][:,zind]) *
            (tausall[zind]*paball[zind]*phi_A0 - (1-paball[zind])*(tausall[zind]/2)*dx_phi_dx_A0)
            + ((1+z)/H[zind]) * (
                ang_avg_Tb['deltamThetabar'][:,zind] - (1/(1-xe))*ang_avg_Tb['xeThetabar'][:,zind]
            ) * (tausall[zind]*paball[zind]*phi_B0 - (1-paball[zind])*(tausall[zind]/2)*dx_phi_dx_B0)
            + (
                alphabar[zind,:]*ang_avg_Tb['deltamdeltambar'][:,zind] -
                ((alphabar[zind,:]+betabar[zind,:])/(1-xe))*ang_avg_Tb['xedeltambar'][:,zind] +
                (betabar[zind,:]/(1-xe)**2)*ang_avg_Tb['xexebar'][:,zind] +
                (gammabar[zind,:]*(1+z)/H[zind])*ang_avg_Tb['deltamThetabar'][:,zind] -
                (gammabar[zind,:]*(1+z)/(H[zind]*(1-xe)))*ang_avg_Tb['xeThetabar'][:,zind] +
                epsbarT[zind,:]*(ang_avg_Tb['Tbdeltambar'][:,zind] - ang_avg_Tb['Tbxebar'][:,zind]/(1-xe))
            ) * (tausall[zind]*paball[zind]*phi_C0 - (1-paball[zind])*(tausall[zind]/2)*dx_phi_dx_C0 + taus_phi)
        ))

        # Eq. (red_20): only the gravity-monopole term from (dnu/dt)^(2) survives the
        # monopole projection. The v-dot (velocity time-derivative) terms are dropped
        # throughout, consistent with the rest of the pipeline; they are no longer
        # included here. The (dnu/dt)^(2) term sources a number-density perturbation
        # only through the local spectral slope df/dnu, so it must be multiplied by it
        # (without this factor fred20 is ~1000x too large and dominates ftot).
        # Eq. (B30) coefficient 8 pi (see source_assembly.assemble_at_z; PMHD_B30_OLD=1 gives the old 16 pi/3).
        _c30 = (16/3) if b30_old_coefficient() else 8.0
        fred20[zind] = f2bars_m0['df/dnu'][0,0] * ((1+z)/(H[zind]*cons.c**2)) * _k_int(karr**(eps-3) * (
            _c30*np.pi*cons.G*cosmo.Ob0*cosmo.critical_density0.value*(1+z)*ang_avg_Tb['deltamThetabar'][:,zind]
        ))

        # Eq. (54), last term (gravitational redshift acting on the l=1 part of delta f^(1)); mirrors
        # source_assembly.assemble_at_z.
        if grav11_enabled():
            fgrav11[zind] = -(4*np.pi*cons.G*cosmo.Ob0*cosmo.critical_density0.value*(1+z)**2/(3*cons.c*H[zind])) * _k_int(karr**(eps-2) * (
                (ang_avg_Tb['deltamdeltambar'][:,zind] - (1/(1-xe))*ang_avg_Tb['xedeltambar'][:,zind]) * dx_A1
                + ((1+z)/H[zind]) * ang_avg_Tb['deltamThetabar'][:,zind] * dx_B1
                + (alphabar[zind,:]*ang_avg_Tb['deltamdeltambar'][:,zind] - (betabar[zind,:]/(1-xe))*ang_avg_Tb['xedeltambar'][:,zind]
                   + (gammabar[zind,:]*(1+z)/H[zind])*ang_avg_Tb['deltamThetabar'][:,zind]
                   + epsbarT[zind,:]*ang_avg_Tb['Tbdeltambar'][:,zind]) * dx_C1
            ))

        # Eqs. (B32)+(B37) sum to -8 pi G rho_b a/(c H nu): 3x the old -8pi/3 (see source_assembly; PMHD_ADVLENS_OLD=1 gives the old).
        _cadv = 1.0 if advlens_old_coefficient() else 3.0
        fadveclens[zind] = -(8*np.pi*cons.G*cosmo.Ob0*cosmo.critical_density0.value*Deltah[zind]*(1+z)**2/(3*cons.c*H[zind])) * _cadv * _k_int(karr**(eps-2) * (
            (A1 + alphabar[zind,:]*C1)*ang_avg_Tb['deltamdeltambar'][:,zind] -
            ((A1 + betabar[zind,:]*C1)/(1-xe))*ang_avg_Tb['xedeltambar'][:,zind] +
            ((B1 + gammabar[zind,:]*C1)*(1+z)/H[zind])*ang_avg_Tb['deltamThetabar'][:,zind] +
            (epsbarT[zind,:]*C1)*ang_avg_Tb['Tbdeltambar'][:,zind]
        ))

        if use_dop:
            dop = assemble_dop_at_z(
                {key: ang_avg_Tb[key][:, zind] for key in
                 ('PhixPhixbar', 'PhiyPhiybar', 'deltamThetabar', 'xeThetabar', 'ThetaThetabar')},
                dict(alphabar=alphabar[zind,:], betabar=betabar[zind,:], gammabar=gammabar[zind,:]),
                unpack_f2bars(f2bars_m0, f2bars_m1, f2bars_p1),
                {key: cross_corr[key][zind] for key in ('vxrms', 'vyrms', 'vzrms')},
                dict(z=z, H=H[zind], xe=xe), reduce_dop,
                coefC_extra=(epsbarT[zind,:]*ang_avg_Tb['TbThetabar'][:,zind] if use_dalpha else None))
            fdop[zind], fdopbar[zind], fv2[zind] = dop['fdop'], dop['fdopbar'], dop['fv2']

    # fred20 (Eq. red_20) is now included alongside fred11/fproc/fadveclens.
    ftot = fred11 + fproc + fadveclens + fred20
    if use_dop:
        ftot = ftot + fdop + fdopbar + fv2
    if grav11_enabled():
        ftot = ftot + fgrav11

    cross_corrsave = OUTBASE / "cross_corr"
    source_fncs    = OUTBASE / "source_fncs"
    cross_corrsave.mkdir(parents=True, exist_ok=True)
    source_fncs.mkdir(parents=True, exist_ok=True)

    Btag  = round(1e12*B0arr[input_bind])
    epstag = round(epsarr[input_epsind], 3)

    with open(cross_corrsave / f'cross_corr_Tb_B_{Btag}pG_e{epstag}.pkl', 'wb') as f:
        pickle.dump(cross_corr, f)
    np.save(source_fncs / f'ftot_Tb_B_{Btag}pG_e{epstag}.npy', ftot)
    np.savez(source_fncs / f'fcomp_Tb_B_{Btag}pG_e{epstag}.npz',
             fred11=fred11, fproc=fproc, fadveclens=fadveclens, fred20=fred20,
             fdop=fdop, fdopbar=fdopbar, fv2=fv2, doppler_terms=use_dop, fgrav11=fgrav11)

    print(time.ctime())


if __name__ == '__main__':
    print(time.ctime())
    import sys
    main(int(sys.argv[1]), int(sys.argv[2]))
    print(time.ctime())
