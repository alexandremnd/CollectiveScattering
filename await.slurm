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
echo "Total number of physical cores detected: $TOTAL_CORES"

julia -E "while True; sleep(1); end" &

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"