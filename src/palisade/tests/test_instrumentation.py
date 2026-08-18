"""The gate-timing hook, and the property that makes it safe to ship.

A host wants per-gate timing; PALISADE must not import the host to provide it.
These tests pin both halves: that an injected recorder actually receives the
tiers, and that the default costs nothing and changes nothing.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from palisade.capabilities.registry import CapabilityRegistry
from palisade.gates.base import Gate, GateContext, GateDecision
from palisade.config import PalisadeSettings
from palisade.trust import TrustScorer
from palisade.instrumentation import (
    GateRecorder,
    NullGateRecorder,
    get_gate_recorder,
    set_gate_recorder,
)


class _Recorder:
    """A host-shaped recorder: not a subclass of anything of ours."""

    def __init__(self, *, live: bool = True) -> None:
        self.live = live
        self.calls: list[dict[str, Any]] = []
        self.active_args: list[tuple] = []

    def active(self, min_level: Any, override: str | None = None) -> bool:
        self.active_args.append((min_level, override))
        return self.live

    def gate(self, **kw: Any) -> None:
        self.calls.append(kw)


class _Gate(Gate):
    name = "GT"

    def __init__(self, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)

    async def _check_fast_when_enabled(self, payload, ctx) -> GateDecision:
        return GateDecision(allow=True, reason="ok")

    async def _check_slow_when_enabled(self, payload, ctx, decision) -> GateDecision:
        return decision


@pytest.fixture(autouse=True)
def _restore_recorder():
    yield
    set_gate_recorder(None)


def _ctx() -> GateContext:
    return GateContext(
        capability_registry=CapabilityRegistry(),
        trust_scorer=TrustScorer(PalisadeSettings()),
    )


def test_default_recorder_is_inactive_and_records_nothing() -> None:
    """The uninstrumented path: no host, no clock, no payload."""
    r = get_gate_recorder()
    assert isinstance(r, NullGateRecorder)
    assert r.active("perf", "gate_timing") is False
    assert r.gate(gate="G1", tier="fast", duration_ms=1.0, allow=True) is None


def test_injected_recorder_receives_both_tiers() -> None:
    rec = _Recorder(live=True)
    set_gate_recorder(rec)
    gate = _Gate()
    asyncio.run(gate.check_fast({}, _ctx()))
    asyncio.run(gate.check_slow({}, _ctx(), GateDecision(allow=True, reason="ok")))

    tiers = [c["tier"] for c in rec.calls]
    assert tiers == ["fast", "slow"], rec.calls
    for c in rec.calls:
        assert c["gate"] == "GT"
        assert c["allow"] is True
        assert isinstance(c["duration_ms"], float) and c["duration_ms"] >= 0.0


def test_inactive_recorder_is_never_asked_for_timing() -> None:
    """`active()` gates the clock: a recorder that says no gets no `gate()`."""
    rec = _Recorder(live=False)
    set_gate_recorder(rec)
    asyncio.run(_Gate().check_fast({}, _ctx()))
    assert rec.active_args, "the dispatch must consult active()"
    assert rec.calls == [], "no timing may be recorded when the probe is off"


def test_a_foreign_recorder_satisfies_the_contract() -> None:
    """Structural, like the host contract: no base class to inherit."""
    assert isinstance(_Recorder(), GateRecorder)


def test_disabled_gate_takes_no_timing() -> None:
    """A disabled gate short-circuits before the probe, so instrumentation
    cannot make a flag-off build differ from an uninstrumented one."""
    rec = _Recorder(live=True)
    set_gate_recorder(rec)
    asyncio.run(_Gate(enabled=False).check_fast({}, _ctx()))
    assert rec.calls == []
    assert rec.active_args == []


def test_set_none_restores_the_no_op() -> None:
    set_gate_recorder(_Recorder())
    set_gate_recorder(None)
    assert isinstance(get_gate_recorder(), NullGateRecorder)
