"""
SLURM driver for rebuilding the Lyman-alpha moment tables (f2bars_dict/ + f1bars -> {a,b,c}barinterpmaster.pkl).

Runs firstOmoments_and_secondO_soln_PRD_fullk.py once per --zchunk (ZSIZE z-indices per chunk; 1300 z-indices
in z_grid()), writing to PMHD_FBARS_OUTBASE so no existing table set is touched.  Afterwards run
assemble_fbars_master.py to build the masters and a complete PMHD_TABLES_DIR:

    python assemble_fbars_master.py --fbars-dir $OUTBASE/fbars/f1bars --out-dir <tables_dir> \\
        --f2bars-dir $OUTBASE/fbars/f2bars_dict

Physics defaults (since 2026-09): non-reflecting closure at ell_max = 14 in all three towers and the
first-order Doppler rate-factor source.  Reproduce the pre-2026-09 tables with
    PMHD_FBARS_NM=9 PMHD_FBARS_NMPM1=30 PMHD_FBARS_FLAGS="--no-closure --no-doppler"

Usage (run from anywhere):
    python slurmloop_fbars.py 0            # just chunk 0 (smoke test)
    python slurmloop_fbars.py all          # all chunks
    python slurmloop_fbars.py 5 12 30      # explicit list
"""
import subprocess
import os
import sys
import time

REPO = "/home/jonschiff/PMF-MHD-recomb"
PY = f"{REPO}/.conda/bin/python"
OUTBASE = os.environ.get("PMHD_FBARS_OUTBASE", f"{REPO}/data/outputs/fbars_doppler_nrbc14")
LOGDIR = os.environ.get("PMHD_FBARS_LOGDIR", f"{OUTBASE}/logs")
ZSIZE = int(os.environ.get("PMHD_FBARS_ZSIZE", 10))
NM = int(os.environ.get("PMHD_FBARS_NM", 15))
NMPM1 = int(os.environ.get("PMHD_FBARS_NMPM1", 15))
FLAGS = os.environ.get("PMHD_FBARS_FLAGS", "")
NZ = 1300
NCHUNK = -(-NZ // ZSIZE)

slurm_template = """#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=03:00:00
#SBATCH --job-name=fbars_c{chunk}
#SBATCH --output={logdir}/fbars_c{chunk}.out
#SBATCH --error={logdir}/fbars_c{chunk}.err
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH={repo}/src
cd {repo}/src/pmhd/physics
{py} firstOmoments_and_secondO_soln_PRD_fullk.py --zchunk {chunk} --zsize {zsize} --nm {nm} --nmpm1 {nmpm1} \\
    --outbase {outbase} {flags} && echo "CHUNK {chunk} DONE" || echo "FAILED chunk={chunk}"
"""


def parse_chunks(argv):
    if not argv:
        return [0]
    if argv == ["all"]:
        return list(range(NCHUNK))
    return [int(a) for a in argv]


if __name__ == "__main__":
    os.makedirs(LOGDIR, exist_ok=True)
    chunks = parse_chunks(sys.argv[1:])
    n = 0
    for chunk in chunks:
        job_script = slurm_template.format(chunk=chunk, repo=REPO, py=PY, outbase=OUTBASE, logdir=LOGDIR,
                                           zsize=ZSIZE, nm=NM, nmpm1=NMPM1, flags=FLAGS)
        job_filename = f"{LOGDIR}/jobfbars_c{chunk}.slurm"
        with open(job_filename, "w") as f:
            f.write(job_script)
        subprocess.run(["sbatch", job_filename])
        os.remove(job_filename)
        n += 1
        time.sleep(0.5)
    print(f"submitted {n} jobs (ZSIZE={ZSIZE}, nm={NM}, nmpm1={NMPM1}, flags='{FLAGS}') for chunks: {chunks}")
