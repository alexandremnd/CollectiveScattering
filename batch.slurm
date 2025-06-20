#!/bin/bash

#SBATCH -J Δ-1000
#SBATCH -p skylake
#SBATCH -N 1
#SBATCH -n 28
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=alexandre.menard1@universite-paris-saclay.fr
#SBATCH --mail-type=END
#SBATCH --output=logs/%A-%a.out

NUM_SOCKETS=$(egrep '^physical id' /proc/cpuinfo | sort -u | wc -l)
PHYSICAL_CORES_PER_SOCKET=$(egrep '^core id' /proc/cpuinfo | sort -u | wc -l)
TOTAL_CORES=$((NUM_SOCKETS * PHYSICAL_CORES_PER_SOCKET))

if [[ $NUM_SOCKETS -eq 0 || $PHYSICAL_CORES_PER_SOCKET -eq 0 ]]; then
    echo "Error: Unable to determine the number of sockets or physical cores."
    TOTAL_CORES=$(nproc)
fi

export JULIA_NUM_THREADS=$TOTAL_CORES

echo "Job $SLURM_JOB_ID beginned on $(hostname) at $(date)"
echo "Total number of physical cores detected: $TOTAL_CORES"

julia src/main.jl

echo
echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
