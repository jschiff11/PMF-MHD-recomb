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

python TCR_Tfs.py {bind} {kind}
"""

# PMF-MHD-recomb/src/pmhd/physics/TCR_Tfs.py

# -----------------------------
# Loop & submit jobs
# -----------------------------
# for bind in range(50, len(B0arr)):
for bind in range(40, 41):
    for kind in range(len(karr)):
    # for kind in range(1):
        print("Submitting bind =", bind, "kind =", kind, time.ctime())

        job_script = slurm_template.format(bind=bind, kind=kind)

        job_filename = f"slurm_job_B{bind}_K{kind}.slurm"
        with open(job_filename, "w") as f:
            f.write(job_script)

        subprocess.run(["sbatch", job_filename])

        os.remove(job_filename)

        time.sleep(0.5)


# import numpy as np
# import os
# import time

# import sys
# from pathlib import Path

# LIB_DIR = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(LIB_DIR))

# import subprocess



# # load the arrays
# B0arr = np.load(LIB_DIR/'B0arr.npy')
# karr = np.load(LIB_DIR/'karr.npy')
# missarr = np.load(LIB_DIR/'missarrsparse.npy')

# # Slurm template
# slurm_template = """#!/bin/bash
# #SBATCH --job-name=TCR_B{bind}_e{kind}   # Job name
# #SBATCH --output=job_out.log     # File to which stdout will be written
# #SBATCH --error=job_error.log # File to which stderr will be written
# #SBATCH -p batch
# #SBATCH -t 00:30:00

# cd /home/jonschiff/lib
# source /home/jonschiff/.bashrc

# python PRD/TCR_PRD.py {bind} {kind} 
# """


# # Loop over each combination of all indices
# # for bind in range(len(B0arr)):
# for bind in range(50,len(B0arr)):
#     for kind in range(len(karr)):
#         print('bind=', bind, 'kind=', kind, time.ctime())
    
#         # Create the job script
#         job_script = slurm_template.format(
#             bind=bind, kind=kind
#         )
        
#         # Save the Slurm script to a file
#         job_filename = f"slurm_job_b{bind}_k{kind}.slurm"
#         with open(job_filename, 'w') as job_file:
#             job_file.write(job_script)
        
#         # Submit the job using sbatch
#         subprocess.run(['sbatch', job_filename])

#         # Delete the .slurm file after job submission
#         os.remove(job_filename)

#         # Optional: Add a small delay to prevent job flooding
#         time.sleep(0.5)


# # for bind, kind, zchunk in missarr.astype(int):
# #     print('bind=', bind, 'kind=', kind, 'zchunk=', zchunk, time.ctime())
    
# #     # Create the job script
# #     job_script = slurm_template.format(
# #         bind=bind, kind=kind, zchunk=zchunk, zsize=100
# #     )
    
# #     # Save the Slurm script to a file
# #     job_filename = f"slurm_job_b{bind}_k{kind}_zch{zchunk}_zsize10.slurm"
# #     with open(job_filename, 'w') as job_file:
# #         job_file.write(job_script)
    
# #     # Submit the job using sbatch
# #     subprocess.run(['sbatch', job_filename])

# #     # Delete the .slurm file after job submission
# #     os.remove(job_filename)

# #     # Optional: Add a small delay to prevent job flooding
# #     time.sleep(0.5)

