"""
SLURM driver (gitignored) to rerun TCR_Tfs.py ONLY (not the full
TCR->FSR_saha->angle_avging chain), one k-index per job, across the
requested B0 indices.

Used to regenerate the tight-coupling transfer functions after correcting
the diffusion coefficient eta(z) (1/15 -> 4/45, i.e. a factor of 4/3, to
match the photon Boltzmann hierarchy derivation in the paper appendix).
Downstream stages (FSR_saha_Tfs, angle_avging_saha, TLA, ...) are NOT rerun
here -- they depend on these corrected TCR outputs and must be resubmitted
separately once this stage is verified.

Submission is throttled exactly like slurmloop_repo.py: polls the account's
current queued+running job count and only submits as many new (bind, k)
jobs as fit under MAX_QUEUED, so it never violates the cluster's
MaxSubmitJobsPerAccount limit. Run this script in the background.

Usage (run from anywhere):
    python slurmloop_TCR_only.py 0            # just bind 0
    python slurmloop_TCR_only.py rest         # all binds except 0
    python slurmloop_TCR_only.py all          # all 61 binds
    python slurmloop_TCR_only.py 5 12 30      # explicit list

Env vars:
    PMHD_KINDS   explicit k-index subset to (re)run instead of 0..NKIND-1,
                 e.g. "8,9" or "6-68"
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
MAX_QUEUED = 180  # stay comfortably under the account's MaxSubmitJobsPerAccount cap
POLL_SECONDS = 30
JOB_PREFIX = "TCReta"


def parse_kinds_env():
    spec = os.environ.get("PMHD_KINDS")
    if not spec:
        return list(range(NKIND))
    kinds = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            kinds.update(range(int(lo), int(hi) + 1))
        else:
            kinds.add(int(part))
    return sorted(kinds)


slurm_template = """#!/bin/bash
#SBATCH --job-name={prefix}_b{bind}_k{kind}
#SBATCH --output={logdir}/tcreta_b{bind}_k{kind}.out
#SBATCH --error={logdir}/tcreta_b{bind}_k{kind}.err
#SBATCH -p batch
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M

export PYTHONPATH={repo}/src
export PMHD_OUTDIR={outdir}
cd {repo}/src/pmhd/physics

{py} TCR_Tfs.py {bind} {kind} || echo "FAILED bind={bind} k={kind}"
echo "BIND {bind} KIND {kind} DONE"
"""


def parse_binds(argv):
    if not argv:
        return [0]
    if argv == ["all"]:
        return list(range(NBIND))
    if argv == ["rest"]:
        return [b for b in range(NBIND) if b != 0]
    return [int(a) for a in argv]


def queued_count():
    out = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h"],
        capture_output=True, text=True,
    ).stdout
    return len(out.splitlines())


def submit_one(bind, kind):
    job_script = slurm_template.format(
        prefix=JOB_PREFIX, bind=bind, kind=kind,
        repo=REPO, py=PY, outdir=OUTDIR, logdir=LOGDIR,
    )
    job_filename = f"{LOGDIR}/job{JOB_PREFIX}_b{bind}_k{kind}.slurm"
    with open(job_filename, "w") as f:
        f.write(job_script)
    subprocess.run(["sbatch", job_filename])
    os.remove(job_filename)


if __name__ == "__main__":
    os.makedirs(LOGDIR, exist_ok=True)
    binds = parse_binds(sys.argv[1:])
    kinds = parse_kinds_env()
    work = [(bind, kind) for bind in binds for kind in kinds]
    total = len(work)
    print(f"{total} (bind, kind) jobs to submit for binds: {binds}")

    submitted = 0
    while work:
        room = MAX_QUEUED - queued_count()
        if room <= 0:
            time.sleep(POLL_SECONDS)
            continue
        batch, work = work[:room], work[room:]
        for bind, kind in batch:
            submit_one(bind, kind)
            submitted += 1
            time.sleep(0.5)
        print(f"submitted {submitted}/{total}, {len(work)} remaining")
        if work:
            time.sleep(POLL_SECONDS)

    print(f"submitted all {submitted} jobs for binds: {binds}")
