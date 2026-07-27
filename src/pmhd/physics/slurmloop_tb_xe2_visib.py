"""
SLURM driver (gitignored) for the helium Tb (baryon-heating) cross-corr / xe2 /
visibility stage, for a specific set of (bind, epsind) pairs -- not a full
cross product. Per bind, one job looping that bind's requested epsind list:

    cross_corr_and_source_funcs_Tb   (positional argv: bind epsind)
    xe2_Tb
    visib_integ_Tb

Prerequisites (must already exist, from the OTHER two drivers):
  - angle_avging_TLA_Tb (all 69 k)      <- slurmloop_ang_avg_downstream.py
  - cont_source + opt_depth          <- slurmloop_xe2_visib.py
  for the same (bind, epsind) pairs.

Default pairs (per user request):
    epsind=9,  binds 0,10,20,30,40,50,60
    bind=10,   epsind 0,4,8,12,16,20

Usage (from anywhere):
    python slurmloop_tb_xe2_visib.py             # default pairs above
"""
import subprocess
import os
import time
from collections import defaultdict

REPO = "/home/jonschiff/PMF-MHD-recomb"
PY = f"{REPO}/.conda/bin/python"
OUTDIR = os.environ.get("PMHD_OUTDIR", "/home/jonschiff/PMF-MHD-recomb/paper_data_plots/data")
LOGDIR = "/home/jonschiff/lib/PRD/He_rec/logs"

# union of: {(b, 9) for b in [0,10,20,30,40,50,60]} and {(10, e) for e in [0,4,8,12,16,20]}
DEFAULT_PAIRS = sorted(set(
    [(b, 9) for b in (0, 10, 20, 30, 40, 50, 60)]
    + [(10, e) for e in (0, 4, 8, 12, 16, 20)]
))

slurm_template = """#!/bin/bash
#SBATCH --job-name=TbXe2VisHe_b{bind}
#SBATCH --output={logdir}/tbxe2vis_b{bind}.out
#SBATCH --error={logdir}/tbxe2vis_b{bind}.err
#SBATCH -p batch
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=1

export PYTHONPATH={repo}/src
export PMHD_OUTDIR={outdir}
cd {repo}/src/pmhd/physics

for epsind in {epsind_list}; do
    {py} cross_corr_and_source_funcs_Tb.py {bind} $epsind && \\
    {py} xe2_Tb.py --bind {bind} --epsind $epsind && \\
    {py} visib_integ_Tb.py --bind {bind} --epsind $epsind \\
    || echo "FAILED bind={bind} epsind=$epsind"
done
echo "BIND {bind} DONE"
"""


def pairs_by_bind(pairs):
    d = defaultdict(list)
    for b, e in pairs:
        d[b].append(e)
    return d


if __name__ == "__main__":
    os.makedirs(LOGDIR, exist_ok=True)
    by_bind = pairs_by_bind(DEFAULT_PAIRS)
    n = 0
    for bind, epsinds in sorted(by_bind.items()):
        epsind_list_str = " ".join(str(e) for e in sorted(epsinds))
        job_script = slurm_template.format(bind=bind, epsind_list=epsind_list_str,
                                           repo=REPO, py=PY, outdir=OUTDIR, logdir=LOGDIR)
        job_filename = f"{LOGDIR}/jobtbxe2vis_b{bind}.slurm"
        with open(job_filename, "w") as f:
            f.write(job_script)
        subprocess.run(["sbatch", job_filename])
        os.remove(job_filename)
        n += 1
        time.sleep(0.5)
    print(f"submitted {n} Tb-xe2-visib jobs, pairs: {sorted(DEFAULT_PAIRS)}")
