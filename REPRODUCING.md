# Reproducing the results

Everything runs on **one workstation**. No result in the paper was produced on
facility infrastructure and no attack traffic was directed at a production
system: PALISADE is client-side, so reproduction needs no facility allocation
and no account.

Which module produces which number is in
[`docs/palisade/README.md`](docs/palisade/README.md). This file is about how to
run them.

---

## What reproduces, and how exactly

Results fall into three classes, and they should be checked against different
standards.

| Class | Standard | Needs |
|---|---|---|
| **Offline / deterministic** | Exact equality, seed 42 | Nothing but a checkout |
| **Served-model** | The reported **range**, not point equality | An OpenAI-compatible endpoint |
| **Scheduler** | Exact parse agreement | The local `slurmctld` container |

**The artifact also runs fully offline.** With no endpoint configured, the
deterministic tiers and every fast-tier column reproduce exactly, and the
slow-tier columns degrade to a recorded stub rather than silently failing open.

---

## Setup

Requires **Python 3.14.3** and [uv](https://docs.astral.sh/uv/). Dependencies
are pinned in `uv.lock`; `pyproject.toml` pins the direct ones too, because
`pydantic-ai>=1.90` floats to a 2.x that removes an API G2 depends on.

```bash
uv sync --extra repro
uv run pytest
```

The suite is deterministic and needs no network. Tests requiring the VISTA
reference deployment (`src/palisade/tests/host_integration/`) skip
automatically; nothing the paper reports depends on them.

### Host

The cost figures in Section V-F and Table VI are single-node and
machine-dependent. Ours is **Darwin arm64, 10 cores**. That bears on
reproduction twice: Docker there runs containers inside a Linux virtual
machine, which scopes the process-isolation result to that topology rather
than to a Linux deployment; and Python 3.14 on arm64 is the least-tested rung
of our dependency stack.

---

## Offline results (minutes to low hours)

These reproduce deterministically at seed 42 with no endpoint:

```bash
uv run python -m tools.corpus_datasheet          # Table III
uv run python -m tools.soft_hard_decomposition   # the soft/hard decomposition
uv run python -m tools.benign_fpr                # Section V-F benign rates
uv run python -m tools.overhead                  # Table VI
uv run python -m tools.cluster_stats             # the cluster-robust intervals
uv run python -m tools.leave_one_out_ablation    # leave-one-class-out
uv run python -m tools.trust_oscillation         # Appendix E-B
uv run python -m tools.gate_scaling              # throughput and tail latency
```

Each overwrites its report under `docs/palisade/`. The committed version is
ours at the release tag, so `git diff` after a run is the comparison.

The full ablation (all nine configurations):

```bash
uv run python -m siege.full_ablation
```

Its `+slow` and `+all` columns need an endpoint; without one they degrade to
the recorded stub.

---

## Served-model results

A served **`gpt-oss-120b` at temperature 0** occupies every model-backed role
in this evaluation: the quarantine model, the LLM judge, AgentDojo's planner,
and both adaptive-attack proposers. The paper flags this as a shared-family
confound on the adaptive results; a cross-family proposer condition would
separate them and we do not run one.

The model must be served **unfiltered**. The safety-filtered variant refuses
adversarial inputs and inflates the apparent block rate, so a reproduction
against a filtered endpoint will report a defense that looks better than it is.

```bash
export OPENAI_BASE_URL="https://<your-endpoint>/v1"
export OPENAI_API_KEY="<key>"
export PALISADE_QUARANTINE_ENABLED=true
export PALISADE_QUARANTINE_MODEL="openai:gpt-oss-120b"
```

Ours was served on the American Science Cloud
(`api.i2-core.american-science-cloud.org`), reached over the network from the
workstation — **not** inside a facility enclave. Nothing sensitive transited
that boundary, since payloads are detection-pattern triggers and dual-use
content is surrogate. A facility running the slow tier on live scientific data
would have to satisfy the in-boundary enclave requirement first, and the
latency figures would move with its endpoint.

```bash
uv run python -m tools.optimizer_hardwin      # Table I (PAIR / TAP)
uv run python -m tools.adaptive_tier_table    # by access tier
uv run python -m tools.adaptive_budget_curve  # ASR vs. query budget
uv run python -m tools.agentdojo_e1           # Appendix D, Table V
```

**Seeds and cost.** The `+slow` and `+all` columns pool five live samples per
instance, costing ~1340 endpoint calls; the harness memoizes on the inspected
text, so a large fraction of lookups are served from cache. PAIR and TAP report
a **range over five seeds** rather than a point — five is thin, and a re-run
should be checked against the range, not against point equality. The
objective-preservation verifier runs at temperature 0.

`tools.agentdojo_e1` defaults to the full 949-pair cross-product (~4–6 h). The
paper's 97-pair coverage is:

```bash
uv run python -m tools.agentdojo_e1 --injection-tasks injection_task_1
```

---

## Scheduler results

The B5 sweeps and the parser differential run against a **containerized
single-node `slurmctld` at Slurm 23.11.4**, shipped under `docker/slurm/`. It
is local; no facility access is involved.

```bash
docker compose -f docker/slurm/compose.yml up -d
./docker/slurm/register_cluster.sh

uv run python -m tools.slurm_live_differential   # parse agreement
uv run python -m tools.slurm_b5_live_sweep       # all 55 B5 submissions
uv run python -m tools.slurm_parser_differential
```

All 55 submission instances are driven through the live controller rather than
a mocked scheduler. No scored instance reached a production scheduler, which
leaves prolog and epilog execution, dependency resolution, and allocation
accounting untested.

`docker/pypi-mirror/` serves the sandbox-mediation probe's package fetches
offline.

---

## Known reproduction hazards

- **Do not let dependencies float.** `pydantic-ai` in particular: `>=1.90`
  resolves to a 2.x that removes `MCPServerStreamableHTTP`, which G2's
  descriptor pinning needs. Use `uv sync`, not `pip install -e .`.
- **A filtered endpoint inflates the defense.** See above.
- **The slow-tier medians rest on small escalation counts** (six and two).
  They are order-of-magnitude figures, not point estimates.
- **Both `+all` adaptive runs logged 3 slow-tier errors**, each scored
  fail-closed, so those cells if anything overstate the defense.
- **Semgrep** runs the ruleset bundled under `src/palisade/contracts/semgrep/`
  (six rule files), not one pinned to an upstream registry, so `+semgrep`
  reproduces without network access.
- **MSTDB-TP is not shipped.** The reference-value contract round-trips cited
  values against it; without it those checks run against the surrogate tables
  under `src/siege/oracles/ground_truth_tables/`. Tolerances are **not**
  domain-scientist signed, so the paper reports contract *coverage*, not ground
  truth — and so should a reproduction.
