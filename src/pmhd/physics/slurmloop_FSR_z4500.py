"""
SLURM driver (gitignored) for the He-inclusive z4500 FSR-saha diagnostic
variant (FSR_saha_Tfs_z4500.py). Tiny scope, matching the notebook's
Stokes-parameter panel (cells 20-21): bindarr[::3] = [0,30,60] x
kind in range(3,73,10) = [3,13,23,33,43,53,63] (7 k, capped at 68).

One job per bind, looping its 7 k values.

Usage (from anywhere):
    python slurmloop_FSR_z4500.py            # default: binds [0,30,60]
"""
import subprocess
import os
import time

REPO = "/home/jonschiff/PMF-MHD-recomb"
PY = f"{REPO}/.conda/bin/python"
OUTDIR = os.environ.get("PMHD_OUTDIR", "/home/jonschiff/PMF-MHD-recomb/paper_data_plots/data")
LOGDIR = "/home/jonschiff/lib/PRD/He_rec/logs"
BINDS = [0, 30, 60]
KINDS = [k for k in range(3, 73, 10) if k < 69]  # [3,13,23,33,43,53,63]

slurm_template = """#!/bin/bash
#SBATCH --job-name=Z4500He_b{bind}
#SBATCH --output={logdir}/z4500_b{bind}.out
#SBATCH --error={logdir}/z4500_b{bind}.err
#SBATCH -p batch
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=1

export PYTHONPATH={repo}/src
export PMHD_OUTDIR={outdir}
cd {repo}/src/pmhd/physics

for k in {kind_list}; do
    {py} FSR_saha_Tfs_z4500.py {bind} $k || echo "FAILED bind={bind} k=$k"
done
echo "BIND {bind} DONE"
"""

if __name__ == "__main__":
    os.makedirs(LOGDIR, exist_ok=True)
    kind_list_str = " ".join(str(k) for k in KINDS)
    n = 0
    for bind in BINDS:
        job_script = slurm_template.format(bind=bind, kind_list=kind_list_str,
                                           repo=REPO, py=PY, outdir=OUTDIR, logdir=LOGDIR)
        job_filename = f"{LOGDIR}/jobz4500_b{bind}.slurm"
        with open(job_filename, "w") as f:
            f.write(job_script)
        subprocess.run(["sbatch", job_filename])
        os.remove(job_filename)
        n += 1
        time.sleep(0.5)
    print(f"submitted {n} z4500-He jobs for binds: {BINDS}, kinds: {KINDS}")
