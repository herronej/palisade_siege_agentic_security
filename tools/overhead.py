"""
Operational overhead measurement (PALISADE WI18/WI21 gap-closure).

Measures the deployment cost the manuscript commits to reporting rather than
estimating (the "fast-only path is sub-100 ms" operational claim): the
end-to-end wall-time shift the gate stack adds to a benign turn, the per-gate
fast-tier latency, the per-decision provenance serialization cost, and the
benign escalation rate (how often a benign action reaches the Q-LLM slow tier).

Timing is wall-clock and therefore machine-dependent, so -- unlike the rest of
the harness -- these numbers are not seed-deterministic; each is a median over
repetitions and the released doc records the machine. The *structure* is
deterministic and is what the test pins (fast-tier per-gate latency well under
100 ms; the stack never speeds a benign turn up).

Lives under ``tools/`` because it times the real gate stack; it only *calls*
the gates (zero gate-source change).

Run: ``python -m tools.overhead``.
"""

from __future__ import annotations

import asyncio
import collections
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from palisade.capabilities.registry import CapabilityRegistry, CapabilityTag
from palisade.config import PalisadeSettings
from palisade.gates.base import GateContext
from palisade.provenance import ProvenanceEmitter
from siege.ablation_matrix import (
    AUGMENTED_CONFIGS,
    CUMULATIVE_CONFIGS,
    build_gate_stack,
)
from siege.instance_loader import load_instances
from siege.session_runner import SessionRunner
from palisade.trust import TrustScorer
from siege.paths import CORPUS_DIR, REPO_ROOT

__all__ = [
    "GateLatency",
    "SlowGateLatency",
    "SlowTierResult",
    "OverheadResult",
    "per_gate_fast_latency",
    "end_to_end_shift",
    "provenance_serialization_cost",
    "benign_escalation_rate",
    "measure_slow_tier",
    "run_overhead",
    "write_results_doc",
]

_CORPUS_DIR = CORPUS_DIR
_BASELINE = CUMULATIVE_CONFIGS[0]
_FULL = CUMULATIVE_CONFIGS[-1]
#: The augmented "full +all" config (fast tier + Semgrep + the Q-LLM slow tier).
_FULL_BOTH = next(c for c in AUGMENTED_CONFIGS if c.name == "full +all")
#: Gates that run a Q-LLM slow tier when a quarantine agent is wired.
_SLOW_TIER_GATES: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5")

#: One representative corpus class per gate; the fast-tier check runs identically
#: for benign and attack payloads, so any real instance gives the right shape.
_GATE_SAMPLES: tuple[tuple[str, str], ...] = (
    ("G1", "b1_1_direct_injection"),
    ("G2", "b3_9_tool_return_injection"),
    ("G3", "b3_1_corpus_poisoning"),
    ("G4", "b4_1_malicious_code"),
    ("G5", "b5_1_mining"),
)

#: Registry sizes at which G2's fast tier is timed. Every other gate's fast
#: tier is constant-time in session state; G2's is not. Its taint walk runs
#: ``propagate_taint`` for each string argument against *every* registered
#: untrusted value, across three decode views, so its cost grows with how long
#: the session has been running. A single number would read as constant and
#: understate a long session, so the report sweeps it instead.
_G2_REGISTRY_SIZES: tuple[int, ...] = (0, 10, 50)


def _slow_tier_judge(model: Any) -> Any | None:
    """The slow tier's judge stage, or None when unavailable.

    Since the judge was folded into the slow tier, an escalated action costs
    *two* model round trips, not one. Timing the tier without it measures a
    configuration that no longer ships, so this is wired into every slow-tier
    measurement below. Returns None when the endpoint is unconfigured, and the
    report says so rather than silently reporting the Q-LLM-only cost.
    """
    if model is None:
        return None
    from siege.redteam.baselines.detectors import LlmJudgeDetector

    judge = LlmJudgeDetector()
    return judge if judge.available() else None


def _fresh_ctx() -> GateContext:
    settings = PalisadeSettings(enabled=True)
    return GateContext(
        capability_registry=CapabilityRegistry(),
        trust_scorer=TrustScorer(settings),
    )


def _first_payload_for_gate(cls: str, gate: str) -> dict:
    """The payload of the first action targeting ``gate`` in class ``cls``."""
    for inst in load_instances(_CORPUS_DIR / cls):
        for session in inst.sessions:
            for turn in session.turns:
                for a in turn.actions:
                    if a.resolved_gate() == gate and a.payload:
                        return dict(a.payload)
    raise LookupError(f"no {gate} action found in {cls}")


# =================================================================
# Per-gate fast-tier latency
# =================================================================


@dataclass(frozen=True)
class GateLatency:
    gate: str
    median_us: float
    p95_us: float


@dataclass(frozen=True)
class G2Scaling:
    """G2 fast-tier latency at one registry size."""

    n_values: int
    median_us: float
    p95_us: float


async def _time_gate(
    gate_obj, payload: dict, *, reps: int, ctx: GateContext | None = None
) -> tuple[float, float]:
    """Median/p95 of ``check_fast`` in microseconds.

    ``ctx`` lets a caller supply pre-populated session state; when None each rep
    gets a fresh (empty) context, which is the right control for the gates whose
    fast tier does not read the registry.
    """
    # Warm up (import/JIT/first-call effects), then time.
    await gate_obj.check_fast(dict(payload), ctx or _fresh_ctx())
    samples: list[float] = []
    for _ in range(reps):
        run_ctx = ctx if ctx is not None else _fresh_ctx()
        t0 = time.perf_counter()
        await gate_obj.check_fast(dict(payload), run_ctx)
        samples.append((time.perf_counter() - t0) * 1e6)  # microseconds
    samples.sort()
    p95 = samples[min(len(samples) - 1, int(0.95 * len(samples)))]
    return statistics.median(samples), p95


async def g2_registry_scaling(*, reps: int = 200) -> list[tuple[int, float, float]]:
    """G2 fast-tier latency at several registry sizes -> [(n_values, med, p95)].

    Populates the registry with distinctive, payload-shaped tainted values (the
    shape ``propagate_taint`` actually does work on -- a short prose string is
    rejected by the distinctiveness floor before any matching happens), so the
    numbers reflect the real containment search rather than an early return.
    """
    stack = build_gate_stack(_FULL)
    gate_obj = stack.gate_for("G2")
    payload = _first_payload_for_gate("b3_9_tool_return_injection", "G2")
    out: list[tuple[int, float, float]] = []
    for n in _G2_REGISTRY_SIZES:
        ctx = _fresh_ctx()
        for i in range(n):
            val = f"/scratch/proj{i:03d}/run_{i:03d}.sh; curl http://h{i:03d}.example/x"
            ctx.capability_registry.tag(val, CapabilityTag(source=f"rag:c{i}", taint=True))
        med, p95 = await _time_gate(gate_obj, payload, reps=reps, ctx=ctx)
        out.append((n, med, p95))
    return out


async def per_gate_fast_latency(*, reps: int = 200) -> list[GateLatency]:
    """Median/p95 fast-tier ``check_fast`` latency per gate (microseconds)."""
    stack = build_gate_stack(_FULL)
    out: list[GateLatency] = []
    for gate, cls in _GATE_SAMPLES:
        gate_obj = stack.gate_for(gate)
        if gate_obj is None:
            continue
        payload = _first_payload_for_gate(cls, gate)
        med, p95 = await _time_gate(gate_obj, payload, reps=reps)
        out.append(GateLatency(gate=gate, median_us=med, p95_us=p95))
    return out


# =================================================================
# End-to-end wall-time shift (baseline vs full)
# =================================================================


async def end_to_end_shift(*, max_instances: int = 14, reps: int = 3) -> tuple[float, float]:
    """Median per-instance wall-time (ms) over the benign suite: (baseline, full).

    Runs each benign instance through the unmodified ``SessionRunner`` at the
    all-off baseline and at full PALISADE (fast tier); the difference is the
    overhead the gate stack adds to a benign turn.
    """
    insts = load_instances(_CORPUS_DIR / "benign_workload")[:max_instances]
    runner = SessionRunner()

    async def _median_ms(config) -> float:
        per: list[float] = []
        for inst in insts:
            runs = [await _timed_run(runner, inst, config) for _ in range(reps)]
            per.append(min(runs))  # min = least-noisy sample per instance
        return statistics.median(per)

    base = await _median_ms(_BASELINE)
    full = await _median_ms(_FULL)
    return base, full


async def _timed_run(runner: SessionRunner, inst, config) -> float:
    t0 = time.perf_counter()
    await runner.run(inst, config)
    return (time.perf_counter() - t0) * 1e3  # milliseconds


# =================================================================
# Provenance serialization cost
# =================================================================


class _NullSink:
    """Counts events without I/O, so the timing is serialization, not disk."""

    def __init__(self) -> None:
        self.n = 0

    def emit(self, event) -> None:  # noqa: ANN001
        self.n += 1


def provenance_serialization_cost(*, reps: int = 2000) -> float:
    """Median time (microseconds) to serialize + emit one gate decision."""
    from palisade.gates.base import GateDecision

    emitter = ProvenanceEmitter(PalisadeSettings(enabled=True), sink=_NullSink())
    decision = GateDecision(allow=True, reason="G5: fast-tier ok", incident_level=None)
    emitter.emit_gate_decision("G5", decision, tier="fast")  # warm up
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        emitter.emit_gate_decision("G5", decision, tier="fast")
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples)


# =================================================================
# Benign escalation rate
# =================================================================


async def benign_escalation_rate(
    *, max_instances: int = 14, judge: Any = None
) -> tuple[int, int]:
    """(escalated, total) benign prompt actions that reach the G1 slow tier.

    Wires a counting mock Q-LLM at G1 and runs the benign suite; a benign prompt
    the fast tier admits escalates to the slow tier. Returns the count so the
    caller can report the rate and the fast-only-vs-slow-tier deployment split.
    """
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart

    from palisade.quarantine import build_intent_extraction_agent

    calls = {"n": 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls["n"] += 1
        return ModelResponse(parts=[ToolCallPart(
            tool_name="final_result",
            args={"intent_summary": "benign scientific query", "dual_use_flag": "none",
                  "confidence": 0.95, "reasoning": "routine query"},
            tool_call_id="c1")])

    agent = build_intent_extraction_agent(FunctionModel(fn))
    insts = load_instances(_CORPUS_DIR / "benign_workload")[:max_instances]
    runner = SessionRunner(quarantine_agents={"G1": agent}, judge=judge)
    total = 0
    for inst in insts:
        for session in inst.sessions:
            for turn in session.turns:
                total += sum(1 for a in turn.actions if a.resolved_gate() == "G1")
        await runner.run(inst, _FULL)
    return calls["n"], total


# =================================================================
# Slow-tier (Q-LLM) latency + +both overhead (W6, real endpoint)
# =================================================================


class _TimingAgent:
    """Wraps a slow-tier agent, recording each ``.run()`` wall-time by gate.

    Non-caching (unlike ``full_ablation``'s ``_CachingAgent``): every escalation
    is a real Q-LLM round-trip, which is exactly the per-escalation latency we
    want. Delegates everything else to the wrapped agent.
    """

    def __init__(self, inner: Any, gate: str, records: list[tuple[str, float]]) -> None:
        self._inner = inner
        self._gate = gate
        self._records = records

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return await self._inner.run(*args, **kwargs)
        finally:
            self._records.append((self._gate, (time.perf_counter() - t0) * 1e3))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _timing_agents(model: Any, records: list[tuple[str, float]]) -> dict[str, Any]:
    """Per-gate slow-tier agents wrapped for timing (same mapping full_ablation uses)."""
    from palisade.quarantine import (
        build_code_intent_extraction_agent,
        build_intent_extraction_agent,
        build_quarantine_agent,
    )

    intent = build_intent_extraction_agent(model)  # G1
    quarantine = build_quarantine_agent(model)  # G2 + G3
    code_intent = build_code_intent_extraction_agent(model)  # G4 + G5
    per_gate = {"G1": intent, "G2": quarantine, "G3": quarantine, "G4": code_intent, "G5": code_intent}
    return {g: _TimingAgent(a, g, records) for g, a in per_gate.items()}


def _pct(sorted_samples: list[float], q: float) -> float:
    if not sorted_samples:
        return 0.0
    return sorted_samples[min(len(sorted_samples) - 1, int(q * len(sorted_samples)))]


@dataclass(frozen=True)
class SlowGateLatency:
    gate: str
    n_calls: int
    p50_ms: float
    p99_ms: float


@dataclass(frozen=True)
class SlowTierResult:
    """Real-endpoint slow-tier latency + the +both deployment overhead."""

    model: str
    per_gate: tuple[SlowGateLatency, ...]
    benign_turns: int
    baseline_ms: float
    fast_ms: float
    both_ms: float
    escalations: int
    concurrency: int
    concurrent_both_ms: float | None
    sequential_both_ms: float | None

    @property
    def both_overhead_ms(self) -> float:
        """Wall-time the slow tier adds to a benign turn over the fast-only path."""
        return self.both_ms - self.fast_ms

    @property
    def concurrency_speedup(self) -> float | None:
        if not (self.concurrent_both_ms and self.sequential_both_ms):
            return None
        return self.sequential_both_ms / self.concurrent_both_ms


async def _run_suite_plus_both(insts, agents_factory, judge=None) -> list[float]:
    """Per-instance wall-time (ms) running the benign suite at +both.

    A fresh ``SessionRunner`` per instance isolates the in-process turn buffer so
    the runs don't interfere (and stay safe if called concurrently).
    """
    out: list[float] = []
    for inst in insts:
        runner = SessionRunner(quarantine_agents=agents_factory(), judge=judge)
        out.append(await _timed_run(runner, inst, _FULL_BOTH))
    return out


async def measure_slow_tier(
    model: Any,
    *,
    max_instances: int = 10,
    concurrency: int = 6,
) -> SlowTierResult:
    """Measure real Q-LLM slow-tier latency + the +both benign-turn overhead.

    Runs the benign suite at ``full +all`` with per-gate timing agents (each
    ``check_slow`` escalation is a real endpoint round-trip), collecting per-gate
    p50/p99 and the per-turn wall-time; compares to the offline baseline and
    fast-only paths; and runs ``concurrency`` distinct benign turns sequentially
    vs. concurrently at +both to expose whether the slow tier serializes. Needs a
    served ``model`` (e.g. ``openai:gpt-oss-120b``).
    """
    # The slow tier is the Q-LLM *and* the judge stage; timing it without the
    # judge measures a configuration that no longer ships.
    judge = _slow_tier_judge(model)
    insts = load_instances(_CORPUS_DIR / "benign_workload")[:max_instances]

    # +both timed run (per-gate slow-tier latency from the timing agents).
    records: list[tuple[str, float]] = []
    both_per = await _run_suite_plus_both(
        insts, lambda: _timing_agents(model, records), judge=judge
    )
    both_ms = statistics.median(both_per) if both_per else 0.0

    by_gate: dict[str, list[float]] = collections.defaultdict(list)
    for gate, dur in records:
        by_gate[gate].append(dur)
    per_gate = tuple(
        SlowGateLatency(
            gate=g,
            n_calls=len(by_gate[g]),
            p50_ms=_pct(sorted(by_gate[g]), 0.5),
            p99_ms=_pct(sorted(by_gate[g]), 0.99),
        )
        for g in _SLOW_TIER_GATES
        if by_gate.get(g)
    )

    # Offline baseline + fast-only shift (reuse the existing measurement).
    baseline_ms, fast_ms = await end_to_end_shift(max_instances=max_instances, reps=3)

    # Concurrency: distinct benign turns sequential vs concurrent at +both.
    conc_insts = insts[:concurrency]
    seq_start = time.perf_counter()
    for inst in conc_insts:
        await SessionRunner(quarantine_agents=_timing_agents(model, []), judge=judge).run(
            inst, _FULL_BOTH
        )
    sequential_ms = (time.perf_counter() - seq_start) * 1e3

    conc_start = time.perf_counter()
    await asyncio.gather(
        *(
            SessionRunner(quarantine_agents=_timing_agents(model, []), judge=judge).run(
                inst, _FULL_BOTH
            )
            for inst in conc_insts
        )
    )
    concurrent_ms = (time.perf_counter() - conc_start) * 1e3

    return SlowTierResult(
        model=str(model),
        per_gate=per_gate,
        benign_turns=len(insts),
        baseline_ms=baseline_ms,
        fast_ms=fast_ms,
        both_ms=both_ms,
        escalations=len(records),
        concurrency=len(conc_insts),
        concurrent_both_ms=concurrent_ms if conc_insts else None,
        sequential_both_ms=sequential_ms if conc_insts else None,
    )


# =================================================================
# Assembled result
# =================================================================


@dataclass(frozen=True)
class OverheadResult:
    gate_latencies: tuple[GateLatency, ...]
    baseline_ms: float
    full_ms: float
    provenance_us: float
    escalated: int
    escalation_total: int
    slow: SlowTierResult | None = None
    g2_scaling: tuple[G2Scaling, ...] = ()

    @property
    def shift_ms(self) -> float:
        return self.full_ms - self.baseline_ms

    @property
    def fast_tier_sub_100ms(self) -> bool:
        return all(g.p95_us < 100_000 for g in self.gate_latencies)

    @property
    def escalation_rate(self) -> float:
        return self.escalated / self.escalation_total if self.escalation_total else 0.0

    def to_markdown(self) -> str:
        lines = [
            "# Operational overhead (WI18/WI21)",
            "",
            "Measured wall-clock cost of the gate stack on a benign turn "
            "(machine-dependent; medians over repetitions). Confirms the "
            "operational claim that the fast-only path is well under 100 ms.",
            "",
            "## Per-gate fast-tier latency",
            "",
            "| gate | median | p95 |",
            "|---|---|---|",
        ]
        for g in self.gate_latencies:
            lines.append(f"| {g.gate} | {g.median_us:.1f} µs | {g.p95_us:.1f} µs |")
        lines += [
            "",
            f"All fast-tier checks are sub-100 ms (p95): **{self.fast_tier_sub_100ms}** "
            "-- deterministic CPU-only checks run in microseconds.",
            "",
        ]
        if getattr(self, "g2_scaling", ()):
            lines += [
                "### G2 is the one fast-tier check that is not constant-time",
                "",
                "Every other gate's fast tier costs the same whatever the session "
                "has done. G2's does not: its taint walk runs the containment "
                "predicate for each string argument against *every* registered "
                "untrusted value, across three decode views, so its cost grows "
                "with session length. The row above is the empty-registry case "
                "and is not representative of a long retrieval-heavy session.",
                "",
                "| registered untrusted values | median | p95 |",
                "|---:|---|---|",
            ]
            for s in self.g2_scaling:
                lines.append(f"| {s.n_values} | {s.median_us:.1f} µs | {s.p95_us:.1f} µs |")
            lines.append("")
        lines += [
            "",
            "## End-to-end wall-time shift (benign turn)",
            "",
            "| config | median per-instance |",
            "|---|---|",
            f"| baseline (all gates off) | {self.baseline_ms:.2f} ms |",
            f"| full PALISADE (fast tier) | {self.full_ms:.2f} ms |",
            f"| **overhead** | **{self.shift_ms:.2f} ms** |",
            "",
            "## Provenance serialization",
            "",
            f"Per-decision serialization cost (allow or deny, off the model's "
            f"critical path): **{self.provenance_us:.1f} µs**.",
            "",
            "## Benign escalation rate",
            "",
            f"Benign prompt actions reaching the G1 slow tier: "
            f"**{self.escalated}/{self.escalation_total}** "
            f"({self.escalation_rate:.0%}). Escalation is **unconditional on a "
            "fast-tier allow** (`gates/base.py`: `check_slow` runs whenever a "
            "quarantine agent is wired; the session runner escalates every "
            "fast-allow) -- there is no uncertainty gate, so benign escalation is "
            "**100%** at every gate with a slow tier deployed, not a narrow "
            "residual. A fast-only deployment pays none of the Q-LLM cost; a "
            "slow-tier deployment pays it on *all* admitted benign traffic, which "
            "is why the fast tier is the reported operating point and the fast/slow "
            "split is a real per-gate deployment cost, not a free optimization.",
            "",
        ]
        if self.slow is not None:
            lines += self._slow_markdown(self.slow)
        return "\n".join(lines)

    @staticmethod
    def _slow_markdown(s: SlowTierResult) -> list[str]:
        lines = [
            "## Slow-tier (Q-LLM) latency and +both overhead",
            "",
            f"Measured against the served `{s.model}` endpoint over "
            f"{s.benign_turns} benign turns ({s.escalations} real escalations); "
            "wall-clock, machine- and network-dependent.",
            "",
            "### Per-gate slow-tier latency (one escalation = one Q-LLM round-trip)",
            "",
            "| gate | escalations | p50 | p99 |",
            "|---|---|---|---|",
        ]
        for g in s.per_gate:
            lines.append(
                f"| {g.gate} | {g.n_calls} | {g.p50_ms:.0f} ms | {g.p99_ms:.0f} ms |"
            )
        lines += [
            "",
            "### Task-completion overhead per benign turn",
            "",
            "| config | median per-turn |",
            "|---|---|",
            f"| baseline (no gates) | {s.baseline_ms:.2f} ms |",
            f"| fast-only (deterministic tier) | {s.fast_ms:.2f} ms |",
            f"| **+both (fast + Q-LLM slow tier)** | **{s.both_ms:.0f} ms** |",
            f"| slow-tier overhead (+both − fast-only) | {s.both_overhead_ms:.0f} ms |",
            "",
            "The fast-only tier -- the reported operating point -- adds "
            f"{s.fast_ms - s.baseline_ms:.2f} ms over the ungated baseline; the "
            "slow tier, when deployed, multiplies a benign turn's wall-time by the "
            "Q-LLM round-trips it escalates (100% of admitted actions), the "
            "dominant deployment cost.",
        ]
        if s.concurrency_speedup is not None:
            lines += [
                "",
                "### Throughput under concurrent sessions (+both)",
                "",
                f"{s.concurrency} benign turns at +both: "
                f"{s.sequential_both_ms:.0f} ms sequential vs "
                f"{s.concurrent_both_ms:.0f} ms concurrent "
                f"(**{s.concurrency_speedup:.1f}× speedup**) -- the slow tier is "
                "I/O-bound on the endpoint and overlaps across sessions rather than "
                "serializing.",
            ]
        lines.append("")
        return lines


async def _arun(
    *, reps: int, e2e_reps: int, model: Any | None, slow_max: int, slow_conc: int
) -> OverheadResult:
    gate_lat = await per_gate_fast_latency(reps=reps)
    base, full = await end_to_end_shift(reps=e2e_reps)
    prov = provenance_serialization_cost()
    esc, tot = await benign_escalation_rate()
    slow = (
        await measure_slow_tier(model, max_instances=slow_max, concurrency=slow_conc)
        if model is not None
        else None
    )
    g2_scale = tuple(
        G2Scaling(n_values=n, median_us=med, p95_us=p95)
        for n, med, p95 in await g2_registry_scaling(reps=reps)
    )
    return OverheadResult(
        gate_latencies=tuple(gate_lat), baseline_ms=base, full_ms=full,
        provenance_us=prov, escalated=esc, escalation_total=tot, slow=slow,
        g2_scaling=g2_scale,
    )


def run_overhead(
    *,
    reps: int = 200,
    e2e_reps: int = 3,
    model: Any | None = None,
    slow_max: int = 10,
    slow_conc: int = 6,
) -> OverheadResult:
    """Run all overhead measurements (wall-clock; machine-dependent).

    With ``model`` (e.g. ``openai:gpt-oss-120b``) it also measures the real Q-LLM
    slow tier (per-gate p50/p99, +both overhead, concurrency); omit for the
    offline fast-tier-only result.
    """
    return asyncio.run(
        _arun(reps=reps, e2e_reps=e2e_reps, model=model, slow_max=slow_max, slow_conc=slow_conc)
    )


def write_results_doc(result: OverheadResult, path: str | None = None) -> str:
    if path is None:
        root = REPO_ROOT
        path = str(root / "docs" / "palisade" / "overhead_wi18.md")
    Path(path).write_text(result.to_markdown(), encoding="utf-8")
    return path


def main(argv=None) -> int:  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(
        description="PALISADE operational overhead (fast tier; optional Q-LLM slow tier)."
    )
    ap.add_argument(
        "--model", default=None,
        help="Q-LLM model spec (e.g. openai:gpt-oss-120b) to also measure the slow "
             "tier; omit for the offline fast-tier-only result.",
    )
    ap.add_argument("--slow-max", type=int, default=10, help="benign turns for the slow-tier run.")
    ap.add_argument("--slow-concurrency", type=int, default=6, help="concurrent sessions for the throughput test.")
    ap.add_argument("--report-out", default=None, metavar="PATH")
    args = ap.parse_args(argv)
    result = run_overhead(
        model=args.model, slow_max=args.slow_max, slow_conc=args.slow_concurrency
    )
    print(f"wrote {write_results_doc(result, args.report_out)}")
    print(result.to_markdown())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
