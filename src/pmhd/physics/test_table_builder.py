"""Smoke test of the moment-table builder entry point (third audit, finding 2).

Runs firstOmoments_and_secondO_soln_PRD_fullk.main for one redshift at reduced resolution and checks that all three
towers (m = 0, +1, -1) and the first-order line averages are written and finite, and that the signed-m dictionaries carry
the derivative keys built by the splrep/splev step. Run as a job (a few minutes):
    PYTHONPATH=<repo>/src python test_table_builder.py [--outdir DIR]
"""
import argparse, pickle, sys, tempfile
from pathlib import Path
import numpy as np

import firstOmoments_and_secondO_soln_PRD_fullk as builder


def main(outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    builder.main(5, 5, 4001, 0, outbase=str(out), zsize=1)       # z index 0 (z = 1900), nm = nmpm1 = 5, dx = 0.5
    ok = True
    for tower in ("m0", "m1", "p1"):
        f = out / f"fbars/f2bars_dict/f2bars_{tower}_dict_zch_0.pkl"
        if not f.exists():
            print(f"MISSING {f}"); ok = False; continue
        d = pickle.load(open(f, "rb"))
        bad = [k for k, v in d.items() if not np.all(np.isfinite(v))]
        need = [f"dx_mom{j}_total" for j in range(3)]
        missing = [k for k in need if k not in d]
        print(f"{tower}: {len(d)} keys, non-finite {bad}, missing derivative keys {missing}")
        ok = ok and not bad and not missing
    for name in ("abar", "bbar", "cbar"):
        f = out / f"fbars/f1bars/{name}_zchunk0.npy"
        if not f.exists():
            print(f"MISSING {f}"); ok = False; continue
        a = np.load(f); print(f"{name}: shape {a.shape}, finite {np.all(np.isfinite(a))}")
        ok = ok and np.all(np.isfinite(a))
    print("TABLE BUILDER TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--outdir", default=tempfile.mkdtemp(prefix="tables_test_"))
    sys.exit(main(p.parse_args().outdir))
