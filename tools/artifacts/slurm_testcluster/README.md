# Minimal single-node Slurm test controller (R5-C5)

A live `slurmctld` for the parser differential (`tools.slurm_live_differential`).
The oracle is *submit-then-inspect* — `sbatch` submits, `scontrol show job` reports
the parsed spec — which needs **only the controller, not compute nodes**, so no
cgroup/systemd is required (a job simply pends without nodes, which is harmless).

The giovtorres/slurm-docker-cluster source build fails on arm64 (a dnf/Spack
stage); this installs Ubuntu 24.04's `slurm-wlm` (Slurm 23.11.4) instead, which
has arm64 packages.

```bash
cd backend/src/vista_backend/palisade/tools/artifacts/slurm_testcluster
docker build -t palisade-slurm .
docker run -d --name cslurm --privileged --cgroupns=host palisade-slurm
# then:
cd ../../../../../..           # back to backend/
uv run python -m tools.slurm_live_differential
docker rm -f cslurm            # teardown
```

`--privileged --cgroupns=host` let slurmd start for completeness, but the parser
differential does not use slurmd — only `slurmctld` (which `scontrol ping` confirms).

## Accounting enforcement (R13-10)

By default the controller has **no accounting database**, so it cannot validate
an account, an association or a QoS at all — it can only check whether a named
object exists on the cluster. That is the configuration the paper's original
live-controller cross-tabulation was measured on, and `PALISADE_ACCOUNTING=0`
(the default) reproduces it unchanged.

Set `PALISADE_ACCOUNTING=1` to bring up `slurmdbd` + MariaDB with
`AccountingStorageEnforce=associations,qos`, so the association check runs at
submission. Two variables control what the facility is taken to have
provisioned, which is the experimental variable:

| variable | default | meaning |
|---|---|---|
| `PALISADE_ACCOUNTS` | `msr_thermo` | comma-separated accounts to create. The default is the sole allocation in G5's own `_EVAL_G5_ALLOCATION_POLICY`, so it is derived from the policy object rather than chosen. |
| `PALISADE_QOS` | `normal,high` | comma-separated QoS to create. `premium` must **not** appear — it is the injected value the `b5_11` survivor carries. |

Associations are created for `root`, because the sweep submits over
`docker exec`. `AdminLevel` is deliberately left at `None`: an Operator or
Administrator would bypass the very check under test.

```bash
docker build -t palisade-slurm:acct .
# strict: only the account G5's policy sanctions
docker run -d --name cslurm_acct --privileged --cgroupns=host \
  -e PALISADE_ACCOUNTING=1 -e PALISADE_ACCOUNTS=msr_thermo \
  -e PALISADE_QOS=normal,high palisade-slurm:acct
# permissive: adds the other account the corpus scripts charge to, which
# separates the account effect from the QoS effect
docker run -d --name cslurm_acct2 --privileged --cgroupns=host \
  -e PALISADE_ACCOUNTING=1 -e PALISADE_ACCOUNTS=msr_thermo,approved-research \
  -e PALISADE_QOS=normal,high palisade-slurm:acct

cd ../../../../../..
uv run python -m tools.slurm_b5_live_sweep \
  --accounting-compare "no accounting=cslurm,strict=cslurm_acct,permissive=cslurm_acct2" \
  --out ../docs/palisade/slurm_b5_accounting_r13.md
```

Startup takes ~60 s with accounting on (MariaDB init plus `sacctmgr`
provisioning); `scontrol ping` returning `UP` is the readiness signal, and the
entrypoint prints the resulting association table.
