# SIEGE substrates

The two throwaway service containers the SIEGE evaluation
(PALISADE Phases 9–16) runs against. Each lives in its own directory
and is runnable standalone; `docker/siege/compose.yml` `include:`s
both so they boot as one stack.

| Substrate | Directory | Image | Boundary it serves |
|---|---|---|---|
| Local PyPI mirror | `docker/siege-pypi-mirror/` | `pypiserver/pypiserver` | B4.3 (scientific-Python typo-squats) |
| SLURM-in-Docker | `docker/siege-slurm/` | `giovtorres/slurm-docker-cluster` | B5.x (HPC-job templates) |

> The Redis cross-session memory container is **deferred to CHUNKS** and
> lives under `docker/deferred/siege-memory/` — it is not part of
> the active two-container stack (the v0.2 harness uses an in-process turn
> buffer, not a persistent memory store).

## Boot everything + smoke test

```bash
docker/siege/smoke_test.sh          # boot, smoke-check each, tear down
docker/siege/smoke_test.sh --keep   # leave the stack running
```

The smoke test satisfies the Phase-9 acceptance criterion ("both
substrate containers boot from compose and pass a smoke test"): the PyPI
`/simple/` index is served, and the SLURM controller answers `sinfo`.

## Boot one substrate

```bash
docker compose -f docker/siege-pypi-mirror/compose.yml up -d
docker compose -f docker/siege-slurm/compose.yml up -d
```

## Notes

- **Ports are offset** (`38080`) so the substrates don't clash with a
  developer's local index server.
- **SLURM first-boot** needs a one-time accounting-db registration —
  run `docker/siege-slurm/register_cluster.sh` after the first
  `up`. The smoke test boots far enough for `sinfo` without it.
- **The default harness run needs none of this.** `python -m
  palisade.eval --gate siege` uses the in-process
  turn buffer and the offline fast-tier gates. The substrates are for
  runs that exercise the service-backed paths (real `pip install`, real
  `sbatch`) — the configuration an external partner reproduces in Phase
  12.
