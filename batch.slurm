#!/bin/bash

#SBATCH -J Δ-2-6
#SBATCH -p skylake
#SBATCH -N 1
#SBATCH --ntasks-per-core=2
#SBATCH --mail-user=alexandre.menard1@universite-paris-saclay.fr
#SBATCH --mail-type=END
#SBATCH --output=logs/%j.out

NUM_SOCKETS=$(egrep '^physical id' /proc/cpuinfo | sort -u | wc -l)
PHYSICAL_CORES_PER_SOCKET=$(egrep '^core id' /proc/cpuinfo | sort -u | wc -l)
TOTAL_CORES=$((NUM_SOCKETS * PHYSICAL_CORES_PER_SOCKET))

if [[ $NUM_SOCKETS -eq 0 || $PHYSICAL_CORES_PER_SOCKET -eq 0 ]]; then
    echo "Error: Unable to determine the number of sockets or physical cores."
    TOTAL_CORES=$(nproc)
fi

export JULIA_NUM_THREADS=$TOTAL_CORES
export SLURM_JOBID=$SLURM_JOB_ID

echo "Job $SLURM_JOB_ID beginned on $(hostname) at $(date)"
echo "Total number of physical cores detected: $TOTAL_CORES"
echo

julia -t auto src/setup.jl

echo
echo

julia src/main.jl

echo
echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
