"""
SLURM driver (gitignored) for the He-inclusive FM transfer functions.

TCR_Tfs_FM.py solves the same TCmag equation as the standard TCR, with a
density-seeded IC [1,0,0,0] instead of the magnetically-seeded [0,0,0,1]. One
job per bind, looping all 69 k internally.

Needed only for bindarr = [0,10,20,30,40,50,60] (the notebook's bindarr), per
the paper_plot_new.ipynb reproduction plan.

Usage (from anywhere):
    python slurmloop_TCR_FM.py            # default: bindarr [0,10,...,60]
    python slurmloop_TCR_FM.py 0 10 20    # explicit list
"""
import subprocess
import os
import sys
import time

REPO = "/home/jonschiff/PMF-MHD-recomb"
PY = f"{REPO}/.conda/bin/python"
OUTDIR = os.environ.get("PMHD_OUTDIR", "/home/jonschiff/PMF-MHD-recomb/paper_data_plots/data")
LOGDIR = "/home/jonschiff/lib/PRD/He_rec/logs"
NKIND = 69
DEFAULT_BINDS = list(range(0, 61, 10))  # [0,10,20,30,40,50,60]

slurm_template = """#!/bin/bash
#SBATCH --job-name=FMHe_b{bind}
#SBATCH --output={logdir}/fm_b{bind}.out
#SBATCH --error={logdir}/fm_b{bind}.err
#SBATCH -p batch
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=1

export PYTHONPATH={repo}/src
export PMHD_OUTDIR={outdir}
cd {repo}/src/pmhd/physics

for k in $(seq 0 {kmax}); do
    {py} TCR_Tfs_FM.py {bind} $k || echo "FAILED bind={bind} k=$k"
done
echo "BIND {bind} DONE"
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
        job_script = slurm_template.format(bind=bind, kmax=NKIND - 1,
                                           repo=REPO, py=PY, outdir=OUTDIR, logdir=LOGDIR)
        job_filename = f"{LOGDIR}/jobfm_b{bind}.slurm"
        with open(job_filename, "w") as f:
            f.write(job_script)
        subprocess.run(["sbatch", job_filename])
        os.remove(job_filename)
        n += 1
        time.sleep(0.5)
    print(f"submitted {n} FM-He jobs for binds: {binds}")
