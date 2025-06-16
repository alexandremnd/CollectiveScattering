#!/bin/bash

#SBATCH -J detuning
#SBATCH -p skylake
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --ntasks-per-core=4
#SBATCH --mail-user=alexandre.menard1@universite-paris-saclay.fr
#SBATCH --mail-type=END
#SBATCH --output=logs/%j.out

echo "Job $SLURM_JOB_ID beginned on $(hostname) at $(date)"
echo

julia -t auto src/main_print.jl

echo "Running main simulation with job ID $SLURM_JOB_ID"
julia -t auto src/main.jl $SLURM_JOB_ID

echo
echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"