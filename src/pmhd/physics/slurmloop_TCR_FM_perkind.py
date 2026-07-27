"""
SLURM driver (gitignored): TCR_Tfs_FM.py one kind per job (the chunked
slurmloop_TCR_FM.py runs ~69x ~40-85min sequentially per job, which blows the
24h walltime; this per-kind variant parallelizes like slurmloop_TCR_only.py).

Usage:
    python slurmloop_TCR_FM_perkind.py            # default binds [0,10,...,60]
    python slurmloop_TCR_FM_perkind.py 0 10 20    # explicit list
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
DEFAULT_BINDS = list(range(0, 61, 10))
MAX_QUEUED = 180
POLL_SECONDS = 30

slurm_template = """#!/bin/bash
#SBATCH --job-name=FMk_b{bind}_k{kind}
#SBATCH --output={logdir}/fmk_b{bind}_k{kind}.out
#SBATCH --error={logdir}/fmk_b{bind}_k{kind}.err
#SBATCH -p batch
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M

export PYTHONPATH={repo}/src
export PMHD_OUTDIR={outdir}
cd {repo}/src/pmhd/physics

{py} TCR_Tfs_FM.py {bind} {kind} || echo "FAILED bind={bind} k={kind}"
echo "FM BIND {bind} KIND {kind} DONE"
"""


def queued_count():
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h"],
                         capture_output=True, text=True).stdout
    return len(out.splitlines())


if __name__ == "__main__":
    binds = [int(a) for a in sys.argv[1:]] or DEFAULT_BINDS
    os.makedirs(LOGDIR, exist_ok=True)
    work = [(b, k) for b in binds for k in range(NKIND)]
    total = len(work)
    submitted = 0
    while work:
        room = MAX_QUEUED - queued_count()
        if room <= 0:
            time.sleep(POLL_SECONDS)
            continue
        batch, work = work[:room], work[room:]
        for bind, kind in batch:
            script = slurm_template.format(bind=bind, kind=kind, repo=REPO, py=PY,
                                           outdir=OUTDIR, logdir=LOGDIR)
            fn = f"{LOGDIR}/jobfmk_b{bind}_k{kind}.slurm"
            with open(fn, "w") as f:
                f.write(script)
            subprocess.run(["sbatch", fn])
            os.remove(fn)
            submitted += 1
            time.sleep(0.5)
        print(f"submitted {submitted}/{total}, {len(work)} remaining")
        if work:
            time.sleep(POLL_SECONDS)
    print(f"submitted all {submitted} FM per-kind jobs for binds: {binds}")
