"""
SLURM driver: He-inclusive 20 keV TCR test (TCR_Tfs_z20.py), one job per
(bind, kind) pair instead of one job per bind looping kind serially, so all
affected modes run in parallel across the cluster.

Output goes to PMHD_OUTDIR = paper_data_plots/data/Tfs/z20/, i.e.
alongside the already-computed astropy-H(z) + He-recombination baseline in
paper_data_plots/data/ (NOT the legacy lib/PRD/He_rec/save_hom_repo
location that slurmloop_TCR_z20.py hardcodes).

69 jobs total (3 binds x 23 affected k, kind 0-22) -- comfortably under the
cluster's 74-simultaneous-job-per-user QOS limit, so all run at once.

Usage (from anywhere):
    python slurmloop_TCR_z20_parallel.py            # default binds [0,30,60]
    python slurmloop_TCR_z20_parallel.py 0 30 60    # explicit list
"""
import subprocess
import os
import sys
import time

REPO = "/home/jonschiff/PMF-MHD-recomb"
PY = f"{REPO}/.conda/bin/python"
OUTDIR = f"{REPO}/paper_data_plots/data"
LOGDIR = f"{REPO}/paper_data_plots/data/logs_z20"
KMAX = 22  # affected modes: kind 0-22 (zcross > z20)
DEFAULT_BINDS = [0, 30, 60]

slurm_template = """#!/bin/bash
#SBATCH --job-name=z20T_{bind}_{kind}
#SBATCH --output={logdir}/tcr_z20_b{bind}_k{kind}.out
#SBATCH --error={logdir}/tcr_z20_b{bind}_k{kind}.err
#SBATCH -p batch
#SBATCH -t 08:00:00

export PYTHONPATH={repo}/src
export PMHD_OUTDIR={outdir}
cd {repo}/src/pmhd/physics

{py} TCR_Tfs_z20.py {bind} {kind} && echo "bind={bind} k={kind} DONE" || echo "FAILED bind={bind} k={kind}"
"""


def parse_binds(argv):
    if not argv:
        return DEFAULT_BINDS
    return [int(a) for a in argv]


if __name__ == "__main__":
    os.makedirs(LOGDIR, exist_ok=True)
    binds = parse_binds(sys.argv[1:])
    n = 0
    for bind in binds:
        for kind in range(KMAX + 1):
            job_script = slurm_template.format(bind=bind, kind=kind,
                                               repo=REPO, py=PY, outdir=OUTDIR, logdir=LOGDIR)
            job_filename = f"{LOGDIR}/job_z20_b{bind}_k{kind}.slurm"
            with open(job_filename, "w") as f:
                f.write(job_script)
            subprocess.run(["sbatch", job_filename], capture_output=True, text=True)
            os.remove(job_filename)
            n += 1
            if n % 20 == 0:
                time.sleep(1)
    print(f"submitted {n} jobs ({len(binds)} binds x {KMAX+1} kinds)")
