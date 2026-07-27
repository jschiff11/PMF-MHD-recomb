"""
SLURM driver (gitignored) to rerun cross_corr_and_source_fncs.py -> xe2.py ->
visib_integ.py for all (bind, epsind) pairs, after promoting the f2bars_dict
cache (src/pmhd/data/pre_stored_data/f2bars_dict/) from its stale April
(custom-H) version to the astropy-H version staged during the Phase 3 fbars
rebuild (that promotion had only copied {a,b,c}barinterpmaster.pkl, missing
this second, separate raw cache that cross_corr_and_source_fncs.py/xe2.py
also read directly).

opt_depth.py and cont_source.py do NOT read f2bars_dict and are therefore
NOT rerun here -- their existing outputs remain valid.

Usage:
    python slurmloop_xe2_visib_f2barsfix.py --binds all --epsinds 0-24
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
MAX_QUEUED = 180
POLL_SECONDS = 30

slurm_template = """#!/bin/bash
#SBATCH --job-name=F2fix_b{bind}
#SBATCH --output={logdir}/f2fix_b{bind}.out
#SBATCH --error={logdir}/f2fix_b{bind}.err
#SBATCH -p batch
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M

export PYTHONPATH={repo}/src
export PMHD_OUTDIR={outdir}
cd {repo}/src/pmhd/physics

for epsind in {epsind_list}; do
    {py} cross_corr_and_source_fncs.py {bind} $epsind && \\
    {py} xe2.py --bind {bind} --epsind $epsind && \\
    {py} visib_integ.py --bind {bind} --epsind $epsind \\
    || echo "FAILED bind={bind} epsind=$epsind"
done
echo "BIND {bind} DONE"
"""


def parse_int_list(spec, default, maxval):
    if spec is None:
        return default
    if spec == "all":
        return list(range(maxval))
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def queued_count():
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h"],
                         capture_output=True, text=True).stdout
    return len(out.splitlines())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--binds", default=None)
    p.add_argument("--epsinds", default=None)
    args = p.parse_args()

    binds = parse_int_list(args.binds, [0, 40], NBIND)
    epsinds = parse_int_list(args.epsinds, [9], 100)
    epsind_list_str = " ".join(str(e) for e in epsinds)

    os.makedirs(LOGDIR, exist_ok=True)
    work = list(binds)
    total = len(work)
    submitted = 0
    while work:
        room = MAX_QUEUED - queued_count()
        if room <= 0:
            time.sleep(POLL_SECONDS)
            continue
        batch, work = work[:room], work[room:]
        for bind in batch:
            job_script = slurm_template.format(bind=bind, epsind_list=epsind_list_str,
                                               repo=REPO, py=PY, outdir=OUTDIR, logdir=LOGDIR)
            job_filename = f"{LOGDIR}/jobf2fix_b{bind}.slurm"
            with open(job_filename, "w") as f:
                f.write(job_script)
            subprocess.run(["sbatch", job_filename])
            os.remove(job_filename)
            submitted += 1
            time.sleep(0.5)
        print(f"submitted {submitted}/{total}")
        if work:
            time.sleep(POLL_SECONDS)
    print(f"submitted all {submitted} f2fix jobs for binds: {binds}, epsinds: {epsinds}")
