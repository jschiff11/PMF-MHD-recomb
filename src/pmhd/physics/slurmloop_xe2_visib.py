"""
SLURM driver (gitignored) for the helium NON-Tb cross-corr / xe2 / visibility
stage, swept over a bind list x epsind list. Per bind, one job:

    opt_depth (once; needs dtaudtaubar/dtaudtaudotbar for all 69 k from
                  slurmloop_ang_avg_downstream.py -- must have already run)
    for each epsind:
        cont_source
        cross_corr_and_source_fncs        (positional argv: bind epsind)
        xe2
        visib_integ

Does NOT run the Tb-specific scripts -- see slurmloop_tb_xe2_visib.py for
that (it reuses this stage's cont_source/opt_depth outputs).

Usage (from anywhere):
    python slurmloop_xe2_visib.py                      # binds 0,40; epsind 9 (validation default)
    python slurmloop_xe2_visib.py --binds all --epsinds 0-25
    python slurmloop_xe2_visib.py --binds 0,10,20 --epsinds 0,5,9,20
"""
import subprocess
import os
import sys
import time
import argparse

REPO = "/home/jonschiff/PMF-MHD-recomb"
PY = f"{REPO}/.conda/bin/python"
OUTDIR = os.environ.get("PMHD_OUTDIR", "/home/jonschiff/PMF-MHD-recomb/paper_data_plots/data")
LOGDIR = "/home/jonschiff/lib/PRD/He_rec/logs"
NBIND = 61

slurm_template = """#!/bin/bash
#SBATCH --job-name=Xe2VisHe_b{bind}
#SBATCH --output={logdir}/xe2vis_b{bind}.out
#SBATCH --error={logdir}/xe2vis_b{bind}.err
#SBATCH -p batch
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=1

export PYTHONPATH={repo}/src
export PMHD_OUTDIR={outdir}
cd {repo}/src/pmhd/physics

{py} opt_depth.py --bind {bind} || echo "FAILED opt_depth bind={bind}"

for epsind in {epsind_list}; do
    {py} cont_source.py --bind {bind} --epsind $epsind && \\
    {py} cross_corr_and_source_fncs.py {bind} $epsind && \\
    {py} xe2.py --bind {bind} --epsind $epsind && \\
    {py} visib_integ.py --bind {bind} --epsind $epsind \\
    || echo "FAILED bind={bind} epsind=$epsind"
done
echo "BIND {bind} DONE"
"""


def parse_int_list(spec, default, maxval):
    """Parse 'all', 'a-b', or 'a,b,c' into a list of ints."""
    if spec is None:
        return default
    if spec == "all":
        return list(range(maxval))
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--binds", default=None, help="'all', 'a-b', or 'a,b,c' (default: 0,40)")
    p.add_argument("--epsinds", default=None, help="'all', 'a-b', or 'a,b,c' (default: 9)")
    args = p.parse_args()

    binds = parse_int_list(args.binds, [0, 40], NBIND)
    epsinds = parse_int_list(args.epsinds, [9], 100)
    epsind_list_str = " ".join(str(e) for e in epsinds)

    os.makedirs(LOGDIR, exist_ok=True)
    n = 0
    for bind in binds:
        job_script = slurm_template.format(bind=bind, epsind_list=epsind_list_str,
                                           repo=REPO, py=PY, outdir=OUTDIR, logdir=LOGDIR)
        job_filename = f"{LOGDIR}/jobxe2vis_b{bind}.slurm"
        with open(job_filename, "w") as f:
            f.write(job_script)
        subprocess.run(["sbatch", job_filename])
        os.remove(job_filename)
        n += 1
        time.sleep(0.5)
    print(f"submitted {n} xe2-visib jobs for binds: {binds}, epsinds: {epsinds}")
