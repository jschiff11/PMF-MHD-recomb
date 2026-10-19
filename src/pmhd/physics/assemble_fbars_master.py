"""
Assemble the per-z-index {a,b,c}bar_zchunk{N}.npy outputs from
firstOmoments_and_secondO_soln_PRD_fullk.py into the three master
interpolation tables ({a,b,c}barinterpmaster.pkl) consumed by
FSR_TLA_Tfs.py / FSR_TLA_Tfs_Tb.py / cross_corr_and_source_fncs.py /
cross_corr_and_source_funcs_Tb.py.

Each master pkl is a length-len(k_grid()) list of scipy splrep tuples
(one spline per k), giving {a,b,c}bar(z) at that k. Confirmed by inspecting
the existing production pkls: the knot vector spans z_grid().min() to
z_grid().max() in ASCENDING order, i.e. z_grid() (which runs 1900->601,
descending) must be reversed before calling splrep.

Usage:
    python assemble_fbars_master.py --fbars-dir data/outputs/fbars_astropyHz/f1bars \\
        --out-dir /tmp/fbars_master_staging
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.interpolate import splrep

import sys
LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))
from pmhd.data.grids import k_grid, z_grid


def assemble(fbars_dir, out_dir):
    fbars_dir = Path(fbars_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    karr = k_grid()
    zarr = z_grid()
    nz = len(zarr)
    nk = len(karr)

    z_asc_order = np.argsort(zarr)
    z_asc = zarr[z_asc_order]

    for name in ("abar", "bbar", "cbar"):
        missing = []
        values = np.zeros((nz, nk))
        for zind in range(nz):
            fpath = fbars_dir / f"{name}_zchunk{zind}.npy"
            if not fpath.exists():
                missing.append(zind)
                continue
            values[zind] = np.load(fpath)
        if missing:
            raise FileNotFoundError(
                f"{name}: missing {len(missing)} z-index files, e.g. {missing[:10]}"
            )

        splines = []
        for kind in range(nk):
            vals_asc = values[z_asc_order, kind]
            tck = splrep(z_asc, vals_asc)
            splines.append(tck)

        out_path = out_dir / f"{name}interpmaster.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(splines, f)
        print(f"wrote {out_path} ({len(splines)} splines)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbars-dir", type=str, required=True,
                         help="Directory containing {a,b,c}bar_zchunk{N}.npy files")
    parser.add_argument("--out-dir", type=str, required=True,
                         help="Directory to write {a,b,c}barinterpmaster.pkl into")
    parser.add_argument("--f2bars-dir", type=str, default=None,
                         help="If given, copy this f2bars_dict/ into --out-dir so it is a complete "
                              "PMHD_TABLES_DIR (masters + f2bars_dict/)")
    args = parser.parse_args()
    assemble(args.fbars_dir, args.out_dir)
    if args.f2bars_dir is not None:
        import shutil
        dst = Path(args.out_dir) / "f2bars_dict"
        if dst.exists():
            raise FileExistsError(f"{dst} already exists; refusing to overwrite")
        shutil.copytree(args.f2bars_dir, dst)
        print(f"copied {args.f2bars_dir} -> {dst} ({len(list(dst.iterdir()))} files)")
