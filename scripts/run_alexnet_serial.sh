#!/bin/bash

#SBATCH --job-name=alex
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1 # number of requested GPUs (GPU nodes shared btwn multiple jobs)
#SBATCH --time=1:30:00 # wall-clock time limit†
#SBATCH --mem=30000
#SBATCH --nodes=1
#SBATCH --mail-type=ALL

# Set up modules.
module purge                               # Unload all currently loaded modules.
module load compiler/gnu/13.3              # Load required modules.
module load mpi/openmpi/4.1
module load devel/cuda/12.4

RESDIR=./job_${SLURM_JOB_ID}/
mkdir ${RESDIR}
cd ${RESDIR}

uv run alex_serial
