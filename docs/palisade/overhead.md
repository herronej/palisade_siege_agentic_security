<!-- Generated report. Source module: tools.overhead. Regenerate with `uv run python -m tools.overhead`. -->

# Operational overhead (WI18/WI21)

Measured wall-clock cost of the gate stack on a benign turn (machine-dependent; medians over repetitions). Confirms the operational claim that the fast-only path is well under 100 ms.

## Per-gate fast-tier latency

| gate | median | p95 |
|---|---|---|
| G1 | 12.1 µs | 13.0 µs |
| G2 | 6.5 µs | 8.4 µs |
| G3 | 32.0 µs | 36.9 µs |
| G4 | 149.6 µs | 156.1 µs |
| G5 | 139.7 µs | 144.7 µs |

All fast-tier checks are sub-100 ms (p95): **True** -- deterministic CPU-only checks run in microseconds.

## End-to-end wall-time shift (benign turn)

| config | median per-instance |
|---|---|
| baseline (all gates off) | 0.02 ms |
| full PALISADE (fast tier) | 0.28 ms |
| **overhead** | **0.26 ms** |

## Provenance serialization

Per-decision serialization cost (allow or deny, off the model's critical path): **3.9 µs**.

## Benign escalation rate

Benign prompt actions reaching the G1 slow tier: **6/6** (100%). Escalation is **unconditional on a fast-tier allow** (`gates/base.py`: `check_slow` runs whenever a quarantine agent is wired; the session runner escalates every fast-allow) -- there is no uncertainty gate, so benign escalation is **100%** at every gate with a slow tier deployed, not a narrow residual. A fast-only deployment pays none of the Q-LLM cost; a slow-tier deployment pays it on *all* admitted benign traffic, which is why the fast tier is the reported operating point and the fast/slow split is a real per-gate deployment cost, not a free optimization.

## Slow-tier (Q-LLM) latency and +both overhead

Measured against the served `openai:gpt-oss-120b` endpoint over 10 benign turns (8 real escalations); wall-clock, machine- and network-dependent.

### Per-gate slow-tier latency (one escalation = one Q-LLM round-trip)

| gate | escalations | p50 | p99 |
|---|---|---|---|
| G1 | 6 | 953 ms | 3959 ms |
| G5 | 2 | 6401 ms | 6401 ms |

### Task-completion overhead per benign turn

| config | median per-turn |
|---|---|
| baseline (no gates) | 0.02 ms |
| fast-only (deterministic tier) | 0.27 ms |
| **+both (fast + Q-LLM slow tier)** | **2497 ms** |
| slow-tier overhead (+both − fast-only) | 2496 ms |

The fast-only tier -- the reported operating point -- adds 0.26 ms over the ungated baseline; the slow tier, when deployed, multiplies a benign turn's wall-time by the Q-LLM round-trips it escalates (100% of admitted actions), the dominant deployment cost.

### Throughput under concurrent sessions (+both)

6 benign turns at +both: 4600 ms sequential vs 3753 ms concurrent (**1.2× speedup**) -- the slow tier is I/O-bound on the endpoint and overlaps across sessions rather than serializing.
