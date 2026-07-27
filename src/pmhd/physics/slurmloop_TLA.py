"""
SLURM driver (gitignored) for the He-inclusive TLA FSR transfer functions.

Reads the He Saha-FSR initial conditions (from stage 1) and runs both the TLA
and the Tb (baryon-heating) TLA He scripts.

Each job covers only KCHUNK=10 k-indices for one bind (rather than all 69 k's
in a single job), so individual jobs finish in minutes instead of an hour+.
Submission is throttled: this script polls the account's current queued+
running job count and only submits as many new (bind, k-chunk) jobs as fit
under MAX_QUEUED, waiting and re-polling as jobs finish and free up room,
until every job has been submitted -- so it never violates the cluster's
MaxSubmitJobsPerAccount limit. This means the script itself runs for as long
as it takes to submit everything (potentially the whole duration of this
pipeline stage); run it in the background.

Usage (from anywhere):
    python slurmloop_TLA.py 0          # bind 0 (smoke)
    python slurmloop_TLA.py all        # all 61 binds
    python slurmloop_TLA.py 0 30 60    # explicit list

Env vars:
    PMHD_KCHUNK   k-indices per job (default 10; set to 1 for one job/kind)
    PMHD_KINDS    explicit k-index subset to (re)run instead of 0..NKIND-1,
                  e.g. "8,9" or "6-68" or "6-20,40,55-68"
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
KCHUNK = int(os.environ.get("PMHD_KCHUNK", "10"))
MAX_QUEUED = 180  # stay comfortably under the account's MaxSubmitJobsPerAccount cap
POLL_SECONDS = 30
JOB_PREFIX = "TLAHe"


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
#SBATCH --job-name={prefix}_b{bind}_k{kstart}
#SBATCH --output={logdir}/tla_b{bind}_k{kstart}.out
#SBATCH --error={logdir}/tla_b{bind}_k{kstart}.err
#SBATCH -p batch
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M

export PYTHONPATH={repo}/src
export PMHD_OUTDIR={outdir}
cd {repo}/src/pmhd/physics

for k in {klist}; do
    {py} FSR_TLA_Tfs.py {bind} $k && \\
    {py} FSR_TLA_Tfs_Tb.py {bind} $k \\
    || echo "FAILED bind={bind} k=$k"
done
echo "BIND {bind} KCHUNK {kstart}-{kend} DONE"
"""


def parse_binds(argv):
    if not argv:
        return [0]
    if argv == ["all"]:
        return list(range(NBIND))
    return [int(a) for a in argv]


def kchunks():
    kinds = parse_kinds_env()
    return [kinds[i:i + KCHUNK] for i in range(0, len(kinds), KCHUNK)]


def queued_count():
    out = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h"],
        capture_output=True, text=True,
    ).stdout
    return len(out.splitlines())


def submit_one(bind, klist):
    kstart, kend = klist[0], klist[-1]
    job_script = slurm_template.format(
        prefix=JOB_PREFIX, bind=bind, kstart=kstart, kend=kend,
        klist=" ".join(str(k) for k in klist),
        repo=REPO, py=PY, outdir=OUTDIR, logdir=LOGDIR,
    )
    job_filename = f"{LOGDIR}/job{JOB_PREFIX}_b{bind}_k{kstart}.slurm"
    with open(job_filename, "w") as f:
        f.write(job_script)
    subprocess.run(["sbatch", job_filename])
    os.remove(job_filename)


if __name__ == "__main__":
    os.makedirs(LOGDIR, exist_ok=True)
    binds = parse_binds(sys.argv[1:])
    work = [(bind, klist) for bind in binds for klist in kchunks()]
    total = len(work)
    print(f"{total} (bind, k-chunk) jobs to submit for binds: {binds}")

    submitted = 0
    while work:
        room = MAX_QUEUED - queued_count()
        if room <= 0:
            time.sleep(POLL_SECONDS)
            continue
        batch, work = work[:room], work[room:]
        for bind, klist in batch:
            submit_one(bind, klist)
            submitted += 1
            time.sleep(0.5)
        print(f"submitted {submitted}/{total}, {len(work)} remaining")
        if work:
            time.sleep(POLL_SECONDS)

    print(f"submitted all {submitted} jobs for binds: {binds}")
