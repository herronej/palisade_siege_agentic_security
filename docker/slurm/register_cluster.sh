#!/usr/bin/env bash
# One-time accounting-db registration for the SIEGE SLURM
# substrate. Run after the first `docker compose up -d`.
set -euo pipefail

CTLD=siege-slurmctld

echo "Registering cluster 'linux' with the accounting db..."
docker exec "$CTLD" bash -lc "sacctmgr --immediate add cluster name=linux" || true

echo "Restarting slurmdbd + slurmctld to pick up the registration..."
docker compose restart siege-slurmdbd siege-slurmctld

echo "Done. Check with: docker exec $CTLD sinfo"
