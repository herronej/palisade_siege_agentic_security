"""
WI18/WI21 overhead-measurement tests.

Timing is wall-clock (machine-dependent), so these pin the *structure* the
manuscript's operational claim rests on -- the fast tier is sub-100 ms and the
measurements are well-formed -- not exact latencies.
"""

from __future__ import annotations

import asyncio

from tools.overhead import (
    GateLatency,
    OverheadResult,
    SlowGateLatency,
    SlowTierResult,
    benign_escalation_rate,
    per_gate_fast_latency,
    provenance_serialization_cost,
    run_overhead,
)


def test_per_gate_fast_tier_is_sub_100ms():
    lat = asyncio.run(per_gate_fast_latency(reps=30))
    # G2 joined the per-gate table: its fast tier is the one whose cost is not
    # constant in session length (the taint walk scans the registry).
    assert {g.gate for g in lat} == {"G1", "G2", "G3", "G4", "G5"}
    for g in lat:
        assert g.median_us > 0.0
        assert g.p95_us < 100_000.0  # deterministic CPU checks << 100 ms


def test_provenance_serialization_is_cheap():
    us = provenance_serialization_cost(reps=100)
    assert 0.0 < us < 100_000.0  # per-decision serialization is microseconds


def test_benign_escalation_rate_is_measured():
    escalated, total = asyncio.run(benign_escalation_rate())
    assert total > 0
    assert 0 <= escalated <= total


def test_run_overhead_assembles_and_confirms_fast_path():
    result = run_overhead(reps=20, e2e_reps=2)
    assert result.fast_tier_sub_100ms
    assert result.baseline_ms > 0 and result.full_ms > 0
    assert result.slow is None  # no model -> slow tier not measured (offline)
    md = result.to_markdown()
    assert "fast-tier" in md.lower()
    assert "overhead" in md.lower()
    # The escalation framing is fixed to 100% / every-fast-allow, not "narrow".
    assert "100%" in md and "unconditional" in md


def _synthetic_slow() -> SlowTierResult:
    return SlowTierResult(
        model="openai:gpt-oss-120b",
        per_gate=(
            SlowGateLatency("G1", 11, 1405.0, 2429.0),
            SlowGateLatency("G5", 11, 3085.0, 6252.0),
        ),
        benign_turns=24, baseline_ms=0.02, fast_ms=0.30, both_ms=2055.0,
        escalations=22, concurrency=8,
        concurrent_both_ms=2765.0, sequential_both_ms=9656.0,
    )


def test_slow_tier_result_derived_metrics():
    s = _synthetic_slow()
    assert round(s.both_overhead_ms, 0) == 2055
    assert round(s.concurrency_speedup, 1) == 3.5


def test_slow_tier_markdown_renders_per_gate_and_overhead():
    result = OverheadResult(
        gate_latencies=(GateLatency("G1", 12.0, 20.0),),
        baseline_ms=0.02, full_ms=0.30, provenance_us=5.0,
        escalated=11, escalation_total=11, slow=_synthetic_slow(),
    )
    md = result.to_markdown()
    assert "Slow-tier (Q-LLM) latency" in md
    assert "p50" in md and "p99" in md
    assert "3085 ms" in md  # G5 p50
    assert "2.1×" in md or "3.5×" in md  # concurrency speedup rendered
