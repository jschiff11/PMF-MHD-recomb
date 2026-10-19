# pyright: ignore[reportMissingImports]
"""
Reusable assembly of the first-order cross-correlations and the second-order Lyman-alpha
source terms (fred11 / fproc / fadveclens / fred20), shared by:

  * ``cross_corr_and_source_fncs.py`` -- the stochastic power-spectrum path, which reduces
    each per-k integrand with  ``int dk k**(eps-n) X(k)``;
  * ``single_mode_source.py``         -- a single (k, theta) mode, which evaluates the same
    integrands at one k with no spectrum weight.

The ONLY difference between the two is the k-reduction, supplied as the callable

    ``reduce_k(per_k_array, n)``

where ``n`` is the term's k-power index (the power-spectrum weight is ``k**(eps - n)``):
``n=1`` for a plain variance, ``n=2`` for ``fadveclens`` (one inverse power of k from the
advection/lensing displacement), ``n=3`` for ``fred20`` and the velocity rms (Poisson 1/k^2).

The expressions below are a VERBATIM extraction of the bodies previously inlined in
``cross_corr_and_source_fncs.py`` (only the array accessors are renamed, e.g.
``ang_avg_dict_master['X'][:,zind]`` -> ``bars['X']``), so the power-spectrum path is
bit-for-bit unchanged.  This is regression-checked against the committed paper data.
"""
import numpy as np
from scipy.interpolate import splrep, splev, splint
from scipy.integrate import solve_ivp
from astropy.cosmology import Planck18 as cosmo

from pmhd import cons, pars

#: keys of the angle-averaged bilinear products consumed by :func:`assemble_at_z`
BAR_KEYS = (
    'bxbxbar', 'bybybar', 'PhixPhixbar',
    'PhiyPhiybar', 'ThetaThetabar', 'PhiyThetabar',
    'deltamThetabar', 'deltamdeltambar', 'xexebar', 'xedeltambar', 'xeThetabar',
    'PhixPhixprimebar', 'PhiyPhiyprimebar', 'ThetaThetaprimebar',
)

#: keys produced by :func:`assemble_at_z` in its cross-correlation row
CROSS_CORR_KEYS = (
    'df00xecross', 'dfeqxecross', 'df00deltamcross',
    'dfeqdeltamcross', 'deltamxecross', 'xeThetacross',
    'xerms', 'deltamrms', 'vxrms', 'vyrms', 'vzrms',
)


# ---------------------------------------------------------------------------
#  k-reduction operators
# ---------------------------------------------------------------------------
def splint_reducer(karr, eps):
    """Power-spectrum reduction: ``int dk k**(eps-n) X(k)`` over the (descending) k-grid.

    Reproduces the inline expression used throughout the original
    ``cross_corr_and_source_fncs.py`` exactly.
    """
    def reduce_k(arr, n, pre=None):
        # `pre` is folded into the weight BEFORE multiplying the integrand so that the
        # floating-point association matches the original inline expressions exactly
        # (e.g. cont_source's  k**(eps-1) * kappa * (...) ).
        w = karr**(eps - n) if pre is None else karr**(eps - n) * pre
        return splint(karr[-1], karr[0], splrep(np.flip(karr), np.flip(w * arr)))
    return reduce_k


def single_mode_reducer(k0):
    """Single-mode reduction: evaluate at one k, keeping the RELATIVE physical k-powers.

    The power-spectrum measure ``k**(eps-1) dk`` is what defines a "plain variance"
    (``n=1``); a term carrying ``k**(eps-n)`` therefore differs from it by ``k**(1-n)``.
    So a single mode of amplitude set by the transfer functions contributes
    ``X(k0) * k0**(1-n)``: unweighted for n=1, ``/k0`` for ``fadveclens``, ``/k0**2`` for
    ``fred20`` and the velocity rms.
    """
    def reduce_k(arr, n, pre=None):
        val = np.asarray(arr).reshape(-1)[0]
        if pre is not None:
            val = np.asarray(pre).reshape(-1)[0] * val
        return float(val) * k0**(1 - n)
    return reduce_k


# ---------------------------------------------------------------------------
#  alpha / beta / gamma prefactors
# ---------------------------------------------------------------------------
def build_abg(zarr, karr, abarinterpfunc, bbarinterpfunc, cbarinterpfunc, xe_full):
    r"""Prefactors relating df_eq / df_00 to the fluid variables.

    Conventions (unchanged from the original)::

        dfeq = \alphaeq \delta_m + \betaeq \delta x_e/x_{1s} + \gammaeq \Theta/aH
        df00 = \alpha00 \delta_m + \beta00 \delta x_e/x_{1s} + \gamma00 \Theta/aH
        dfeq - psc*df00 = alphabar \delta_m + \betabar \delta x_{1s}/x_{1s} + \gammabar \Theta/aH

    so that ``alphabar = alphaeq - p_sc alpha00``, ``betabar = -betaeq + p_sc beta00``,
    ``gammabar = gammaeq - p_sc gamma00``.

    Returns a dict of nine ``(len(zarr), len(karr))`` arrays.
    """
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

    return dict(alphaeq=alphaeq, betaeq=betaeq, gammaeq=gammaeq,
                alpha00=alpha00, beta00=beta00, gamma00=gamma00,
                alphabar=alphabar, betabar=betabar, gammabar=gammabar)


def unpack_f2bars(f2bars_m0_master, f2bars_m1_master, f2bars_p1_master):
    """Unpack the per-k Lyman-alpha Boltzmann moment columns needed by :func:`assemble_at_z`."""
    m0, m1, p1 = f2bars_m0_master, f2bars_m1_master, f2bars_p1_master
    out = _unpack_core(m0, m1, p1)
    # rest-frame-monopole (Doppler/aberration) keys: present only in tables built since 2026-09
    if 'Qdop_mom1_total' in m0:
        out.update(
            QdopA=m0['Qdop_mom1_total'][:,0], QdopB=m0['Qdop_mom1_total'][:,1], QdopC=m0['Qdop_mom1_total'][:,2],
            Qdop_p=p1['Qdop_mom1_total'][:,0], Qdop_m=m1['Qdop_mom1_total'][:,0],
            fbarA1=m0['fbar_mom1_total'][:,0], fbarB1=m0['fbar_mom1_total'][:,1], fbarC1=m0['fbar_mom1_total'][:,2],
            fbar1_p=p1['fbar_mom1_total'][:,0], fbar1_m=m1['fbar_mom1_total'][:,0],
        )
    return out


def _unpack_core(m0, m1, p1):
    return dict(
        A1=m0['mom1_total'][:,0], B1=m0['mom1_total'][:,1], C1=m0['mom1_total'][:,2],
        dx_A0=m0['dx_mom0_total'][:,0], dx_B0=m0['dx_mom0_total'][:,1], dx_C0=m0['dx_mom0_total'][:,2],
        dx_A2=m0['dx_mom2_total'][:,0], dx_B2=m0['dx_mom2_total'][:,1], dx_C2=m0['dx_mom2_total'][:,2],
        dx_A1=m0['dx_mom1_total'][:,0], dx_B1=m0['dx_mom1_total'][:,1], dx_C1=m0['dx_mom1_total'][:,2],
        dx_D2_m=m1['dx_mom2_total'][:,0], dx_D2_p=p1['dx_mom2_total'][:,0],
        dx_phi_dx_A0=m0['dx_phi_dx_mom0_total'][:,0],
        dx_phi_dx_B0=m0['dx_phi_dx_mom0_total'][:,1],
        dx_phi_dx_C0=m0['dx_phi_dx_mom0_total'][:,2],
        phi_A0=m0['phi_mom0_total'][:,0], phi_B0=m0['phi_mom0_total'][:,1], phi_C0=m0['phi_mom0_total'][:,2],
        taus_phi=m0['taus_phi'][0,0], dfdnu=m0['df/dnu'][0,0],
    )


# ---------------------------------------------------------------------------
#  the per-z assembly
# ---------------------------------------------------------------------------
def assemble_at_z(bars, abg, moms, bg, reduce_k):
    """Assemble the cross-correlations and the four source terms at a single redshift.

    Parameters
    ----------
    bars : dict
        Angle-averaged bilinear products at this z, each a per-k array (see :data:`BAR_KEYS`).
    abg : dict
        ``alpha00/alphaeq/beta00/betaeq/gamma00/gammaeq/alphabar/betabar/gammabar`` at this z,
        each a per-k array.
    moms : dict
        Output of :func:`unpack_f2bars` at this z.
    bg : dict
        Background scalars at this z: ``z, H, taus, pab, xe, Deltah``.
    reduce_k : callable
        ``reduce_k(per_k_array, n)`` -- see :func:`splint_reducer` / :func:`single_mode_reducer`.

    Returns
    -------
    (cross_corr_row, fred11, fproc, fadveclens, fred20, fgrav11)
    """
    z, Hz, taus, pab, xe, Dh = bg['z'], bg['H'], bg['taus'], bg['pab'], bg['xe'], bg['Deltah']
    cc = {}

    cc['deltamrms'] = reduce_k(bars['deltamdeltambar'], 1)
    cc['vxrms']     = reduce_k(bars['PhixPhixbar'], 3)
    cc['vyrms']     = reduce_k(bars['PhiyPhiybar'], 3)
    cc['vzrms']     = reduce_k(bars['ThetaThetabar'], 3)
    cc['xerms']     = reduce_k(bars['xexebar'], 1)

    cc['df00xecross'] = reduce_k(
        abg['alpha00']*bars['xedeltambar'] + (
            abg['beta00']/(1-xe) )*bars['xexebar'] + (
            abg['gamma00']*(1+z)/Hz) * bars['xeThetabar'], 1)
    cc['dfeqxecross'] = reduce_k(
        abg['alphaeq']*bars['xedeltambar'] + (
            abg['betaeq']/(1-xe) )*bars['xexebar'] + (
            abg['gammaeq']*(1+z)/Hz) * bars['xeThetabar'], 1)
    cc['df00deltamcross'] = reduce_k(
        abg['alpha00']*bars['deltamdeltambar'] + (
            abg['beta00']/(1-xe) )*bars['xedeltambar'] + (
            abg['gamma00']*(1+z)/Hz) * bars['deltamThetabar'], 1)
    cc['dfeqdeltamcross'] = reduce_k(
        abg['alphaeq']*bars['deltamdeltambar'] + (
            abg['betaeq']/(1-xe) )*bars['xedeltambar'] + (
            abg['gammaeq']*(1+z)/Hz) * bars['deltamThetabar'], 1)
    cc['deltamxecross'] = reduce_k(bars['xedeltambar'], 1)
    cc['xeThetacross']  = reduce_k(bars['xeThetabar'], 1)

    ## Compute source functions
    fred11 = reduce_k(
        (
            (-(1+z)**2/(5*Hz**2 )) * (
                bars['PhixPhixbar'] + bars['PhiyPhiybar']
            ) * ( moms['dx_D2_p'] + moms['dx_D2_m']  )
            +  ((1+z)/(3*Hz)) * (bars['deltamThetabar']  - (1/ ( 1- xe ) ) * bars['xeThetabar']
            ) * ( (2/5)*moms['dx_A2'] - moms['dx_A0']  )
            + ((1+z)**2/(3*Hz**2) ) * (bars['ThetaThetabar']) * ( (2/5)*moms['dx_B2'] - moms['dx_B0']  )
            + ((1+z)/(3*Hz)) * (abg['alphabar'] * bars['deltamThetabar'] - (abg['betabar']/(1-xe))*bars['xeThetabar'] +
           ( abg['gammabar']*(1+z)/Hz )*bars['ThetaThetabar']  ) *
        ( (2/5)*moms['dx_C2'] - moms['dx_C0']  )
        ), 1)

    # Eq. (54), last term: the gravitational-redshift part of (dnu/dt)^(1), -c n.grad(phi)/a with the Poisson
    # potential phi = -4 pi G a^2 rho_b delta_b/(c k)^2, acting on the l=1 part of delta f^(1):
    #   -(4 pi G rho_b a/(3 c k H)) [ (T_dd - T_dx/x1s) dA1/dnu + T_dTheta/(aH) dB1/dnu
    #                                 + (abar T_dd - bbar T_dx/x1s + gbar T_dTheta/(aH)) dC1/dnu ].
    # rho_b a = rho_b0 (1+z)^2; the 1/k is the n=2 weight; one 1/c (phi/c^2 with the comoving gradient).
    fgrav11 = 0.0
    if grav11_enabled():
        fgrav11 = -(4*np.pi*cons.G*cosmo.Ob0*cosmo.critical_density0.value*(1+z)**2/(3*cons.c*Hz)) * reduce_k(
            (bars['deltamdeltambar'] - (1/(1-xe))*bars['xedeltambar']) * moms['dx_A1']
            + ((1+z)/Hz) * bars['deltamThetabar'] * moms['dx_B1']
            + (abg['alphabar']*bars['deltamdeltambar'] - (abg['betabar']/(1-xe))*bars['xedeltambar']
               + (abg['gammabar']*(1+z)/Hz)*bars['deltamThetabar']) * moms['dx_C1'], 2)

    fproc = reduce_k(
        ( (bars['deltamdeltambar'] + (1/ ( 1- xe )**2 ) * bars['xexebar'] -
           (2/ ( 1- xe ) ) * bars['xedeltambar']  ) *
        (taus*pab*moms['phi_A0'] - (1-pab)*(taus/2) * moms['dx_phi_dx_A0']  )
           + ((1+z)/Hz) * (bars['deltamThetabar'] -
           (1/ ( 1- xe ) ) * bars['xeThetabar']  ) *
        (taus*pab*moms['phi_B0'] - (1-pab)*(taus/2) * moms['dx_phi_dx_B0']  )
           + (
            abg['alphabar'] * bars['deltamdeltambar'] -
           ( (abg['alphabar'] + abg['betabar'])/(1-xe) ) * bars['xedeltambar'] +
            (abg['betabar']/(1-xe)**2 ) * bars['xexebar'] +
            (abg['gammabar']*(1+z)/Hz) * bars['deltamThetabar'] -
            (abg['gammabar']*(1+z)/ ( Hz *(1-xe ) ) ) * bars['xeThetabar']
           ) *
        (taus*pab*moms['phi_C0'] -
         (1-pab)*(taus/2) * moms['dx_phi_dx_C0'] + moms['taus_phi']   )
        ), 1)

    # Eq. (red_20): only the gravity-monopole term from (dnu/dt)^(2) survives the
    # monopole projection. The v-dot (velocity time-derivative) terms are dropped
    # throughout, consistent with the rest of the pipeline; they are no longer
    # included here. The (dnu/dt)^(2) term sources a number-density perturbation
    # only through the local spectral slope df/dnu, so it must be multiplied by it
    # (without this factor fred20 is ~1000x too large and dominates ftot).
    # Coefficient (Eq. B30): the two surviving monopole terms of (B29), -2<phi Theta>/(3a) and (4/3a)<v.grad phi>,
    # add with <v.grad phi> = -<phi Theta> (integration by parts) to -2<phi Theta>/a = 8 pi G rho_b a <delta Theta>/k^2,
    # i.e. 8 pi G rho_b a/(H c^2). The pre-2026-10 code used 16 pi/3 (PMHD_B30_OLD=1 reproduces it).
    _c30 = (16/3) if b30_old_coefficient() else 8.0
    fred20 = moms['dfdnu'] * ((1+z)/(Hz * cons.c**2) ) * reduce_k(
            _c30 * np.pi * cons.G * cosmo.Ob0*cosmo.critical_density0.value * (1+z) * bars['deltamThetabar'], 3)

    # Eqs. (B32) + (B37): advection -(8 pi G rho_b a/(3 c H nu)) and lensing -(16 pi G rho_b a/(3 c H nu)) act on the
    # same l=1 bracket and add to -(8 pi G rho_b a/(c H nu)), i.e. 3x the -8pi/3 that the code carried until 2026-10-08
    # (PMHD_ADVLENS_OLD=1 reproduces it). The 1/nu is carried by Deltah (x-variable form of the frequency derivative).
    _cadv = 1.0 if advlens_old_coefficient() else 3.0
    fadveclens = - ( (8*np.pi * cons.G * cosmo.Ob0*cosmo.critical_density0.value * Dh * (1+z)**2 )/(3*cons.c*Hz)  ) * _cadv * reduce_k(
            (moms['A1'] + abg['alphabar'] * moms['C1'] ) * bars['deltamdeltambar'] -
           ((moms['A1'] + abg['betabar'] * moms['C1'])/(1-xe))*bars['xedeltambar']
            +  ( ( moms['B1'] +  abg['gammabar'] * moms['C1'] ) *(1+z) / Hz ) * bars['deltamThetabar'], 2)

    return cc, fred11, fproc, fadveclens, fred20, fgrav11


# ---------------------------------------------------------------------------
#  rest-frame-monopole (Doppler / aberration) second-order terms
# ---------------------------------------------------------------------------
def doppler_terms_enabled():
    """Doppler/aberration second-order terms are on unless PMHD_DOPPLER_TERMS=0 (regression mode)."""
    import os
    return os.environ.get("PMHD_DOPPLER_TERMS", "1") != "0"


def _switch(name, default="1"):
    """Environment switch for the 2026-10 manuscript-sync terms (paper_data_plots/manuscript_sync/PREREG.md).
    Each defaults to on; set to 0 to reproduce the pre-sync (data_doppler) outputs."""
    import os
    return os.environ.get(name, default) != "0"


def eq63_term_enabled():
    """Eq. (63): +(1/a)<Theta^(1) delta x_e^(1)> in the <delta x_e^(2)> equation (convective derivative)."""
    return _switch("PMHD_EQ63_TERM")


def grav11_enabled():
    """Eq. (54): the gravitational-redshift piece of (dnu/dt)^(1) acting on the l=1 part of delta f^(1)."""
    return _switch("PMHD_GRAV11")


def b30_old_coefficient():
    """Eq. (B30): True reproduces the pre-sync coefficient 16 pi G rho_b a/(3 H c^2) instead of 8 pi G rho_b a/(H c^2)."""
    return _switch("PMHD_B30_OLD", "0")


def dalpha_feq_enabled():
    """T_b path only: delta f_eq^(1) (and hence delta fbar_00^(1)) carry the recombination-coefficient perturbation
    delta_alphaB = (dln alpha_B/dln T_b) delta_Tb through the n = 2 balance (Eq. 24):
        delta f_eq ⊃ n_H x_e^2 alpha_B delta_alphaB / (x_1s [3 A P_esc + Lambda_2s1s + 4 beta_B]),
    the delta_b recombination source of alpha_eq with delta_b -> delta_alphaB. PMHD_DALPHA_FEQ=0 reproduces the
    pre-2026-10-12 T_b outputs (manuscript_sync/PREREG.md addendum 8)."""
    return _switch("PMHD_DALPHA_FEQ")


def advlens_old_coefficient():
    """Eqs. (B32)+(B37): True reproduces the pre-2026-10-08 advection+lensing coefficient -8 pi G rho_b a/(3 c H nu)
    (the sum +8pi/3 - 16pi/3 of the earlier manuscript); the manuscript now has -8pi/3 - 16pi/3 = -8 pi G rho_b a/(c H nu),
    confirmed by the direct real-field evaluation (research_runs/claude_slurm/sync/oddl_sign_checks_v2.py)."""
    return _switch("PMHD_ADVLENS_OLD", "0")


def assemble_dop_at_z(bars, abg, moms, cc, bg, reduce_k, coefC_extra=None):
    r"""Second-order terms from the rest-frame Lyman-alpha monopole f_00 = f_00^N + delta f_00^Dop.

    The PSD is parametrised by (rest-frame nu, lab n_N).  The monopole seen by the atoms needs the
    aberration Jacobian, delta f_00^Dop = 2 <(v.n_N/c) delta f^(1)>.  In the paper's multipole
    convention this is (Eq. 406)
        delta f_00^Dop = -(2/3ck) [ -Phi_- df_11 + (1/2) Phi_+ df_1,-1 + Theta df_10 ].
    With the basis solutions and the spatial average this becomes (Eq. 418/419)
        -(2/3c) int dk k^(eps-2) [ -T_{Phi+Phi-}/aH (D1+ + 3 D1-) + (T_{Theta dm} - T_{Theta xe}/x1s) A1
                                   + T_{Theta Theta}/aH B1 + (abar T_{Theta dm} - bbar/x1s T_{Theta xe}
                                   + gbar/aH T_{Theta Theta}) C1 ].
    The factor 3 on D1- is the code normalisation (D- is solved with -1/6 at ell=2, like D+).

    Returns a dict with three contributions to the effective source sbar (same units as ftot):
      fdop    : in-scattering (420) + out-scattering (416).  The combination above, with each ell=1
                moment replaced by its 'Qdop' key (line average of D_nu^-1 of the collision operator).
      fdopbar : rate term (421).  The same combination with the 'fbar_mom1' keys, i.e. the line-averaged
                delta f_00^Dop.  It enters as sbar -> sbar + <delta fbar_00^Dop>, assuming the emission
                closure uses the rest-frame fbar_00.
      fv2     : Lorentz-factor term (417), (1/2)<v^2>/c^2 df/dnu, with <v^2> = vxrms+vyrms+vzrms and
                the line-averaged response to df/dx taken from the 'df/dnu' table.
    """
    for key in ('QdopA', 'QdopB', 'QdopC', 'Qdop_p', 'Qdop_m', 'fbarA1', 'fbarB1', 'fbarC1', 'fbar1_p', 'fbar1_m'):
        if key not in moms:
            raise KeyError(f"moment tables lack '{key}': they predate the Doppler/aberration terms. "
                           "Rebuild the tables, or set PMHD_DOPPLER_TERMS=0 for regression runs.")
    z, Hz, xe = bg['z'], bg['H'], bg['xe']
    invaH = (1+z)/Hz
    x1s = 1 - xe
    TPP = bars['PhixPhixbar'] + bars['PhiyPhiybar']
    TTd, TTx, TTT = bars['deltamThetabar'], bars['xeThetabar'], bars['ThetaThetabar']
    coefC = abg['alphabar']*TTd - (abg['betabar']/x1s)*TTx + abg['gammabar']*invaH*TTT
    if coefC_extra is not None:          # T_b path: + epsbar <delta_alphaB Theta> (dalpha_feq_enabled)
        coefC = coefC + coefC_extra

    # Overall sign (manuscript Eq. B20, +2/(3ck)): the real-field spatial average of the odd-l product
    # 2<(v.n/c) delta f^(1)> was evaluated directly in the (-i)^l convention of Eq. 40 (which the hierarchies use)
    # and agrees with B20; the earlier write-up's Eq. 406 (-2/3ck) dropped the complex conjugation of that average.
    # PMHD_DOP_SIGN_OLD=1 reproduces the pre-2026-10-07 sign (paper_data_plots/manuscript_sync/PREREG.md, addendum 1).
    sgn = -1.0 if _switch("PMHD_DOP_SIGN_OLD", "0") else +1.0

    def combo(QA, QB, QC, Qp, Qm):
        return sgn*(2/(3*cons.c)) * reduce_k(
            -(invaH*TPP)*(Qp + 3*Qm) + (TTd - TTx/x1s)*QA + invaH*TTT*QB + coefC*QC, 2)

    fdop = combo(moms['QdopA'], moms['QdopB'], moms['QdopC'], moms['Qdop_p'], moms['Qdop_m'])
    fdopbar = combo(moms['fbarA1'], moms['fbarB1'], moms['fbarC1'], moms['fbar1_p'], moms['fbar1_m'])
    v2 = cc['vxrms'] + cc['vyrms'] + cc['vzrms']
    fv2 = 0.5*v2/cons.c**2 * moms['dfdnu']
    return dict(fdop=fdop, fdopbar=fdopbar, fv2=fv2)


# ---------------------------------------------------------------------------
#  the continuum ("cont") source  -- shared with cont_source.py
# ---------------------------------------------------------------------------
def Acon(z, xe_full):
    return (1 - xe_full(z)) *  pars.nh(z) *  pars.sigmaa(z)


def B1con(z, xe_full):
    return xe_full(z)**2 * pars.nh(z) *  (pars.alpha1s(z) - pars.alphab(z))


def B2con(z, xe_full):
    return xe_full(z) * pars.nh(z) *  (pars.alpha1s(z) - pars.alphab(z)) * (2-xe_full(z))/(1-xe_full(z))


def kappa(k, z, xe_full):
    """Non-local Lyman-continuum escape kernel (k-dependent, so it cannot leave the integral)."""
    return 1 - Acon(z, xe_full)/(k*(1+z)) * np.arctan((k*(1+z))/Acon(z, xe_full))


def assemble_cont_at_z(bars, kvals, z, xe_full, reduce_k):
    """``xe2contsource`` at a single redshift (the `conterm` subtracted inside x1squad)."""
    return reduce_k(
        B1con(z, xe_full)*bars['deltamdeltambar'] + B2con(z, xe_full)*bars['xedeltambar'], 1,
        pre=kappa(kvals, z, xe_full))


# ---------------------------------------------------------------------------
#  the second-order x_1s ODE  -- shared with xe2.py
# ---------------------------------------------------------------------------
def solve_x1squad(zarr, cross_corr, sbar2, abar2, cbar2, cont, xe_full,
                  zspan=None, y0=1e-20, method='LSODA', atol=1e-9, rtol=1e-6, max_step=1.0):
    r"""Integrate the second-order neutral-fraction ODE for :math:`x_{1s}^{(2)}`.

    Verbatim extraction of the ``x1squad`` body previously inlined in ``xe2.py``; the caller
    supplies the sources in memory so both the power-spectrum path and the single-mode path
    use the identical ODE.

    Parameters
    ----------
    zarr : array
        Descending redshift grid on which ``cross_corr``/``sbar2``/``abar2``/``cbar2``/``cont``
        are tabulated.
    cross_corr : dict
        Needs ``df00xecross``, ``dfeqxecross``, ``deltamxecross``, ``xerms``, ``xeThetacross`` (Eq. 63 term).
    sbar2, abar2, cbar2, cont : arrays over ``zarr``
        ``ftot``, ``df/dnu``, ``taus_phi`` and the continuum source respectively.
    zspan : (z_start, z_end), optional
        Integration range; defaults to ``(zarr[0], zarr[-1])``.
    max_step : float, optional
        Largest LSODA step in z (default 1.0, the z-grid spacing). Unbounded steps (the pre-2026-09-25
        behaviour, ``max_step=np.inf``) let LSODA stride across z ~ 1500-1900 where x1s^(2) is small, and the
        dense-output interpolant between those steps was off by up to 3.8% of max|xe2| (5 pG, eps = -0.01).

    Returns
    -------
    The ``solve_ivp`` result (``dense_output=True``), i.e. ``hold`` with ``hold.sol(z)``.
    """
    ## Build splines once here (outside x1squad) so they are not rebuilt on every ODE evaluation
    _zarr_flip = zarr[::-1]
    _tck_f00xecross    = splrep(_zarr_flip, cross_corr['df00xecross'][::-1])
    _tck_feqxecross    = splrep(_zarr_flip, cross_corr['dfeqxecross'][::-1])
    _tck_dmxecross     = splrep(_zarr_flip, cross_corr['deltamxecross'][::-1])
    _tck_dxedxecross   = splrep(_zarr_flip, cross_corr['xerms'][::-1])
    # Eq. (63): the convective-derivative term +(1/a)<Theta^(1) delta x_e^(1)> in d<delta x_e^(2)>/dt enters
    # dx_1s^(2)/dt (x_1s^(2) = -<delta x_e^(2)>) as -(1+z)<Theta delta x_e>, i.e. it adds to the subtracted
    # continuum term. It is folded into the same spline as `cont` (this reproduces the pre-registered analysis
    # research_runs/claude_slurm/xeadv exactly). Theta = div v in s^-1 (cross_corr['xeThetacross']).
    cont_eff = cont + ((1 + zarr)*cross_corr['xeThetacross'] if eq63_term_enabled() else 0.0)
    _tck_conterm       = splrep(_zarr_flip, cont_eff[::-1])
    _tck_abar2         = splrep(np.flip(zarr), np.flip(abar2))
    _tck_sbar2         = splrep(np.flip(zarr), np.flip(sbar2))
    _tck_cbar2         = splrep(np.flip(zarr), np.flip(cbar2))

    def x1squad(z, x1s2):
        def f00xecrossinterp(z):
            return splev(z, _tck_f00xecross)

        def feqxecrossinterp(z):
            return splev(z, _tck_feqxecross)

        def dmxecrossinterp(z):
            return splev(z, _tck_dmxecross)

        def dxedxecrossinterp(z):
            return splev(z, _tck_dxedxecross)

        def conterminterp(z):
            return splev(z, _tck_conterm)

        def abar2interp(z):
            return splev(z, _tck_abar2)
        def sbar2interp(z):
            return splev(z, _tck_sbar2)
        def cbar2interp(z):
            return splev(z, _tck_cbar2)
        def pesc2(z):
            return (1-pars.pab(z)*cbar2interp(z)) / (1 + (1-pars.pab(z))*cbar2interp(z))

        def P3LA2(z):
            return (3*cons.Alya * pesc2(z) + cons.L2s1s)/(
                3*cons.Alya * pesc2(z) + cons.L2s1s + 4*pars.betab(z) )

        def pscinterp(z):
            return 1-pars.pab(z)

        return (-1/( (1+z)*pars.H(z))) * (
            ( (1-P3LA2(z) )*pars.nh(z)*(xe_full(z)**2/(1-xe_full(z)))*pars.alphab(z) -
            2 * P3LA2(z) * pars.nh(z) * xe_full(z) * pars.alphab(z) - 4*pars.feq(z,xe_full)* pars.betab(z) )*x1s2
            -3* (1-P3LA2(z) ) * (1-xe_full(z))*cons.Alya *
            ( ( sbar2interp(z) + (abar2interp(z)*(x1s2 - dmxecrossinterp(z) )/(1-xe_full(z))) ) / (1+pscinterp(z)*cbar2interp(z)) -
             f00xecrossinterp(z)/(1-xe_full(z)) ) +
            2 * P3LA2(z) * pars.nh(z) * xe_full(z) * pars.alphab(z) * dmxecrossinterp(z) +
            P3LA2(z) * pars.nh(z) * pars.alphab(z) * dxedxecrossinterp(z) -
            ( (1-P3LA2(z) ) * (3*cons.Alya + cons.L2s1s ) - 4*P3LA2(z)*pars.betab(z) ) * feqxecrossinterp(z) - conterminterp(z)
        )

    if zspan is None:
        zspan = (zarr[0], zarr[-1])
    return solve_ivp(x1squad, [zspan[0], zspan[1]], [y0], method=method,
                     dense_output=True, atol=atol, rtol=rtol, max_step=max_step)
