import numpy as np
import os
import time
import sys
from pathlib import Path
import subprocess

# -----------------------------
# Set library root
# -----------------------------
LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))

# -----------------------------
# Load arrays
# -----------------------------


from pmhd.data.grids import (
    k_grid,
    theta_grid,
    load_or_generate_z_arrays,
    load_or_generate_B0arr,
)
B0arr = load_or_generate_B0arr()
karr = k_grid()

# -----------------------------
# Slurm script template
# -----------------------------
slurm_template = """#!/bin/bash
#SBATCH --job-name=TCR_B{bind}_k{kind}
#SBATCH --output=job_out.log
#SBATCH --error=job_err.log
#SBATCH --partition=batch
#SBATCH --time=02:00:00

cd /home/jonschiff/PMF-MHD-recomb/src/pmhd/physics

python angle_avging_opt_depth.py --bind {bind} --kind {kind}
"""

# PMF-MHD-recomb/src/pmhd/physics/TCR_Tfs.py

# -----------------------------
# Loop & submit jobs
# -----------------------------
# for bind in range(50, len(B0arr)):
for bind in range(40, 41):
    for kind in range(len(karr)):
        print("Submitting bind =", bind, "kind =", kind, time.ctime())

        job_script = slurm_template.format(bind=bind, kind=kind)

        job_filename = f"slurm_job_B{bind}_K{kind}.slurm"
        with open(job_filename, "w") as f:
            f.write(job_script)

        subprocess.run(["sbatch", job_filename])

        os.remove(job_filename)

        time.sleep(0.5)
