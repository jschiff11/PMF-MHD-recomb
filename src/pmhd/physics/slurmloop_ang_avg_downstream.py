"""
SLURM driver (gitignored) for the helium angle-averaging downstream stage:
angle_avging_TLA -> angle_avging_TLA_Tb -> angle_avging_opt_depth.

One job per bind, looping all 69 k internally (friendly to the shared queue).
Must complete before slurmloop_xe2_visib.py (opt_depth needs the
dtaudtaubar/dtaudtaudotbar files this stage produces for ALL k).

Usage (from anywhere):
    python slurmloop_ang_avg_downstream.py           # binds 0, 40 (validation)
    python slurmloop_ang_avg_downstream.py all        # all 61 binds
    python slurmloop_ang_avg_downstream.py 0 30 60    # explicit list
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
NBIND = 61

slurm_template = """#!/bin/bash
#SBATCH --job-name=AngHe_b{bind}
#SBATCH --output={logdir}/angdown_b{bind}.out
#SBATCH --error={logdir}/angdown_b{bind}.err
#SBATCH -p batch
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=1

export PYTHONPATH={repo}/src
export PMHD_OUTDIR={outdir}
cd {repo}/src/pmhd/physics

for k in $(seq 0 {kmax}); do
    {py} angle_avging_TLA.py --bind {bind} --kind $k && \\
    {py} angle_avging_TLA_Tb.py --bind {bind} --kind $k && \\
    {py} angle_avging_opt_depth.py --bind {bind} --kind $k \\
    || echo "FAILED bind={bind} k=$k"
done
echo "BIND {bind} DONE"
"""


def parse_binds(argv):
    if not argv:
        return [0, 40]
    if argv == ["all"]:
        return list(range(NBIND))
    return [int(a) for a in argv]


if __name__ == "__main__":
    os.makedirs(LOGDIR, exist_ok=True)
    binds = parse_binds(sys.argv[1:])
    n = 0
    for bind in binds:
        job_script = slurm_template.format(bind=bind, kmax=NKIND - 1,
                                           repo=REPO, py=PY, outdir=OUTDIR, logdir=LOGDIR)
        job_filename = f"{LOGDIR}/jobangdown_b{bind}.slurm"
        with open(job_filename, "w") as f:
            f.write(job_script)
        subprocess.run(["sbatch", job_filename])
        os.remove(job_filename)
        n += 1
        time.sleep(0.5)
    print(f"submitted {n} ang-avg-downstream jobs for binds: {binds}")
