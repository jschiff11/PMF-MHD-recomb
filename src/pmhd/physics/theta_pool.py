"""Run the per-angle (theta) integrations of a transfer-function script in parallel.

Each angle is an independent ODE solve, so the results are bit-identical to the serial loop. The number of worker
processes is PMHD_NPROC (default 1 = serial). Uses a forked pool so the per-script closures need not be picklable.

Forking a process that holds BLAS thread locks can deadlock the pool (observed 2026-10-13: workers asleep at 0% CPU
for hours on a handful of modes). Jobs that set PMHD_NPROC > 1 must also set OPENBLAS_NUM_THREADS=1 and
MKL_NUM_THREADS=1 (in addition to OMP_NUM_THREADS=1) before Python starts, and should wrap each call in a timeout
with a serial (PMHD_NPROC=1) fallback.
"""
import os
import multiprocessing as mp

_FN = None


def _call(i):
    return _FN(i)


def theta_map(fn, n, nproc=None):
    """[fn(0), ..., fn(n-1)] in order, computed by PMHD_NPROC forked workers (serially when PMHD_NPROC <= 1)."""
    global _FN
    nproc = int(os.environ.get("PMHD_NPROC", "1")) if nproc is None else int(nproc)
    if nproc <= 1 or n <= 1:
        return [fn(i) for i in range(n)]
    _FN = fn
    with mp.get_context("fork").Pool(min(nproc, n)) as pool:
        return pool.map(_call, range(n), chunksize=1)
