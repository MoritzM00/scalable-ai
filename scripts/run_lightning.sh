#!/bin/bash

#SBATCH --job-name=resnext
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:4   # number of requested GPUs (GPU nodes shared btwn multiple jobs)
#SBATCH --ntasks=4
#SBATCH --time=30:00 # wall-clock time limit
#SBATCH --mem=40000
#SBATCH --nodes=1
#SBATCH --mail-type=ALL

# Set up modules.
module purge                               # Unload all currently loaded modules.
module load compiler/gnu/13.3              # Load required modules.
module load mpi/openmpi/4.1
module load devel/cuda/12.4

unset SLURM_NTASKS_PER_TRES

srun uv run resnext_lightning
