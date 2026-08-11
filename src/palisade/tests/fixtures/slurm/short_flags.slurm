#!/bin/bash -l
# Exercises every short-flag #SBATCH directive form the parser must
# normalize: -N, -t, -n, -c, -A, -p, -q, -d, and a -G gpu count.
#SBATCH -N 16
#SBATCH -t 1-12:30:00
#SBATCH -n 64
#SBATCH -c 8
#SBATCH -A constrained-resource
#SBATCH -p debug
#SBATCH -q normal
#SBATCH -d afterok:990011
#SBATCH -G 4

srun hostname
