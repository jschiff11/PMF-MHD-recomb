"""
SLURM driver (gitignored) to rerun angle_avging_saha.py for all (bind, kind)
pairs, after the eta (diffusion coefficient) correction invalidated the
previously-computed ang_avg/saha pkls (they combine the corrected TCRalf/
TCRmag endpoints with the still-valid FSRsahaalf/FSRsahamag arrays).

angle_avging_saha.py does no ODE integration (just array loads + an angular
average), so one job per bind loops over all 69 kinds in-process (avoiding
per-kind Python startup cost) instead of one job per (bind, kind) -- only 61
jobs total, no throttling needed.

Usage:
    python slurmloop_ang_avg_saha_rerun.py all
    python slurmloop_ang_avg_saha_rerun.py 0 1 2
"""
import subprocess
import os
import sys

REPO = "/home/jonschiff/PMF-MHD-recomb"
PY = f"{REPO}/.conda/bin/python"
OUTDIR = os.environ.get("PMHD_OUTDIR", "/home/jonschiff/PMF-MHD-recomb/paper_data_plots/data")
LOGDIR = "/home/jonschiff/lib/PRD/He_rec/logs"
NBIND = 61
NKIND = 69
JOB_PREFIX = "AngSaha"

slurm_template = """#!/bin/bash
#SBATCH --job-name={prefix}_b{bind}
#SBATCH --output={logdir}/angsaha_b{bind}.out
#SBATCH --error={logdir}/angsaha_b{bind}.err
#SBATCH -p batch
#SBATCH -t 4:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M

export PYTHONPATH={repo}/src
export PMHD_OUTDIR={outdir}
cd {repo}/src/pmhd/physics

{py} - <<'PYEOF'
from angle_avging_saha import main
for k in range({nkind}):
    try:
        main({bind}, k)
    except Exception as e:
        print(f"FAILED bind={bind} k={{k}}: {{e}}")
PYEOF
echo "BIND {bind} ANG_AVG_SAHA DONE"
"""


def parse_binds(argv):
    if not argv:
        return [0]
    if argv == ["all"]:
        return list(range(NBIND))
    return [int(a) for a in argv]


def submit_one(bind):
    job_script = slurm_template.format(
        prefix=JOB_PREFIX, bind=bind, nkind=NKIND,
        repo=REPO, py=PY, outdir=OUTDIR, logdir=LOGDIR,
    )
    job_filename = f"{LOGDIR}/job{JOB_PREFIX}_b{bind}.slurm"
    with open(job_filename, "w") as f:
        f.write(job_script)
    subprocess.run(["sbatch", job_filename])
    os.remove(job_filename)


if __name__ == "__main__":
    os.makedirs(LOGDIR, exist_ok=True)
    binds = parse_binds(sys.argv[1:])
    for bind in binds:
        submit_one(bind)
    print(f"submitted {len(binds)} jobs (one per bind) for binds: {binds}")
