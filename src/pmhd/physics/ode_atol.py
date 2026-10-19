"""First-order transfer-function solves with component-scaled absolute tolerances (PREREG addenda 20-22).

The transfer-function states mix an O(1) magnetic component with fluid and ionization components that, in weak-field
low-k modes, are 1e-8..1e-12 (delta x_e; at the lowest k also delta, Theta, Phi). One shared absolute tolerance leaves
the error of those components uncontrolled (addenda 18, 19). With ``PMHD_ATOL_SCALED=1`` (the default since 2026-10-16)
the solve is repeated with a per-component absolute tolerance

    atol_i = PMHD_ATOL_FRAC * rtol * s_i,

where s_i is the largest |y_i| on the output grid of the previous pass (the first pass uses the scalar atol), until every
s_i is stable within a factor 2 or ``PMHD_ATOL_PASSES`` passes are done. A component that is identically zero keeps the
scalar atol, and a failed pass raises. An LSODA pass that runs longer than ``PMHD_ATOL_LSODA_BUDGET`` seconds (default
2400, as the production run wrapper; 0 disables) is redone with Radau and the same tolerances: LSODA stalls on the
stiffest strong-field TLA and T_b modes at these tolerances (addendum 21). ``PMHD_ATOL_SCALED=0`` makes the single
production call unchanged (bit-identical to the outputs before the switch existed, e.g. ``data_final4``).
"""
import os
import signal
import sys
import threading

import numpy as np
from scipy.integrate import solve_ivp

SCALED = os.environ.get("PMHD_ATOL_SCALED", "1") == "1"
FRAC = float(os.environ.get("PMHD_ATOL_FRAC", "1e-3"))
PASSES = int(os.environ.get("PMHD_ATOL_PASSES", "4"))
LSODA_BUDGET = float(os.environ.get("PMHD_ATOL_LSODA_BUDGET", "2400"))


class _OverBudget(Exception):
    pass


_expired = False


def _set_expired(signum, frame):
    # only set a flag: an exception raised here would surface inside whatever library code is running (e.g. astropy's
    # unit parser, which converts it and can be left in a broken state); the RHS guard raises at a clean point instead
    global _expired
    _expired = True


def _guard(fun):
    def guarded(t, y, *a):
        if _expired:
            raise _OverBudget()
        return fun(t, y, *a)
    return guarded


def _solve(fun, t_span, y0, method, args, tol):
    """One scaled-mode pass: solve_ivp with dense output; an LSODA pass over the budget is redone with Radau."""
    global _expired
    if not (method == "LSODA" and LSODA_BUDGET > 0 and threading.current_thread() is threading.main_thread()):
        return solve_ivp(fun, t_span, y0, method=method, dense_output=True, args=args, **tol)
    sol, _expired = None, False
    old = signal.signal(signal.SIGALRM, _set_expired)
    signal.setitimer(signal.ITIMER_REAL, LSODA_BUDGET)
    try:
        sol = solve_ivp(_guard(fun), t_span, y0, method=method, dense_output=True, args=args, **tol)
    except _OverBudget:
        sol = None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        _expired = False
    if sol is None:
        print(f"ode_atol: LSODA over {LSODA_BUDGET:g} s, Radau fallback", file=sys.stderr, flush=True)
        sol = solve_ivp(fun, t_span, y0, method="Radau", dense_output=True, args=args, **tol)
    return sol


def solve_on_grid(fun, t_span, y0, t_grid, *, method, args=None, **tol):
    """``solve_ivp(..., dense_output=True, **tol).sol(t_grid)``; with ``PMHD_ATOL_SCALED=1``, iterated as above.

    ``tol`` holds the production ``rtol``/``atol`` (none: scipy's defaults, 1e-3 and 1e-6)."""
    if not SCALED:
        return solve_ivp(fun, t_span, y0, method=method, dense_output=True, args=args, **tol).sol(t_grid)
    sol = _solve(fun, t_span, y0, method, args, tol)
    if not sol.success:
        raise RuntimeError(f"transfer-function solve failed (scalar atol): {sol.message}")
    y = sol.sol(t_grid)
    rtol, atol0 = tol.get("rtol", 1e-3), tol.get("atol", 1e-6)
    s = np.max(np.abs(y), axis=1)
    for _ in range(PASSES - 1):
        atol = np.where(s > 0, FRAC * rtol * s, atol0)
        sol = _solve(fun, t_span, y0, method, args, dict(rtol=rtol, atol=atol))
        if not sol.success:
            raise RuntimeError(f"transfer-function solve failed (scaled atol): {sol.message}")
        y = sol.sol(t_grid)
        s_new = np.max(np.abs(y), axis=1)
        stable = np.all((s == 0) | ((s_new <= 2 * s) & (s_new >= 0.5 * s)))
        s = s_new
        if stable:
            break
    return y
