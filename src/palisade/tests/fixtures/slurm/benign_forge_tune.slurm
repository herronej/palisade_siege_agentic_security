#!/bin/bash -l
#SBATCH --job-name=vista-forge-tune
#SBATCH --account=approved-research
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --time=02:00:00

module load cray-python
export VISTA_OUT="${VISTA_OUT:-$PWD/out}"
mkdir -p "$VISTA_OUT"

srun python forge-tune.py \
    --input Molten_Salt_Thermophysical_Properties.csv \
    --checkpoint-dir "$VISTA_OUT" \
    --num-epochs 25
