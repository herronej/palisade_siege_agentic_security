"""
Unit tests for `ProvenanceEmitter`.

Covers the disabled-path (master flag off), the JSONL-write path
(master flag on, Flowcept disabled), the fail-fast construction
contract, the schema of an emitted event, and the Flowcept-broker sink
(the `FlowceptTask`-over-MQ publish path and its endpoint-env mapping).
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

from palisade.capabilities import (
    CapabilityTag,
    DualUseMarker,
    SensitivityTier,
)
from palisade.config import PalisadeSettings
from palisade.gates.base import GateDecision
from palisade.incidents import IncidentRecord
from palisade.provenance import (
    EVENT_GATE_DECISION,
    EVENT_INCIDENT,
    ProvenanceEmitter,
)


# -----------------------------------------------------------------
# Construction-time validation
# -----------------------------------------------------------------


def test_constructor_raises_when_flowcept_enabled_without_endpoint():
    """Fail-fast: flowcept_enabled=True with no endpoint -> ValueError."""
    settings = PalisadeSettings(
        enabled=True,
        flowcept_enabled=True,
        flowcept_endpoint=None,
    )
    with pytest.raises(ValueError, match="flowcept_endpoint"):
        ProvenanceEmitter(settings)


def test_constructor_accepts_flowcept_enabled_with_endpoint():
    """flowcept_enabled=True + endpoint set -> constructs fine."""
    settings = PalisadeSettings(
        enabled=True,
        flowcept_enabled=True,
        flowcept_endpoint="redis://broker.local:6379",
    )
    # Should not raise; emission path is exercised separately.
    ProvenanceEmitter(settings)


def test_constructor_accepts_flowcept_disabled_default():
    """flowcept_enabled=False (default) -> constructs without endpoint."""
    settings = PalisadeSettings(enabled=True)
    ProvenanceEmitter(settings)


def test_constructor_generates_session_id_when_unset():
    """When session_id is None the emitter generates a stable UUID4 hex."""
    settings = PalisadeSettings(enabled=True)
    emitter = ProvenanceEmitter(settings)
    assert emitter.session_id
    # uuid4().hex is 32 chars; this is a smoke check, not a regex on UUID.
    assert len(emitter.session_id) == 32


def test_constructor_preserves_explicit_session_id():
    settings = PalisadeSettings(enabled=True)
    emitter = ProvenanceEmitter(settings, session_id="test-session")
    assert emitter.session_id == "test-session"


# -----------------------------------------------------------------
# Master-flag no-op contract
# -----------------------------------------------------------------


def test_emit_gate_decision_is_noop_when_master_flag_off(tmp_path):
    """When settings.enabled is False, emit_gate_decision returns None and writes nothing."""
    log_path = tmp_path / "provenance.jsonl"
    settings = PalisadeSettings(
        enabled=False,
        provenance_log_path=str(log_path),
    )
    emitter = ProvenanceEmitter(settings)

    result = emitter.emit_gate_decision(
        gate="G2",
        decision=GateDecision(allow=True, reason="ok"),
    )

    assert result is None
    assert emitter.last_event is None
    assert not log_path.exists()


def test_emit_incident_is_noop_when_master_flag_off(tmp_path):
    """When settings.enabled is False, emit_incident returns None and writes nothing."""
    log_path = tmp_path / "provenance.jsonl"
    settings = PalisadeSettings(
        enabled=False,
        provenance_log_path=str(log_path),
    )
    emitter = ProvenanceEmitter(settings)

    record = IncidentRecord(
        level=2,
        severity_name="SEV2",
        gate="G2",
        reason="suspicious",
    )
    result = emitter.emit_incident(record)

    assert result is None
    assert emitter.last_event is None
    assert not log_path.exists()


# -----------------------------------------------------------------
# JSONL emission (flowcept disabled, master flag on)
# -----------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_emit_gate_decision_writes_jsonl(tmp_path):
    log_path = tmp_path / "provenance.jsonl"
    settings = PalisadeSettings(
        enabled=True,
        provenance_log_path=str(log_path),
    )
    emitter = ProvenanceEmitter(settings, session_id="sess-1")

    tag = CapabilityTag(
        source="tool:foo",
        sensitivity=SensitivityTier.INTERNAL,
        dual_use=DualUseMarker.NONE,
        taint=True,
        provenance_chain=("origin",),
        metadata={"k": "v"},
    )
    decision = GateDecision(
        allow=False,
        reason="blocked",
        capability_tag=tag,
        incident_level=2,
        rewritten_args={"sanitized": "..."},
    )
    event = emitter.emit_gate_decision(gate="G2", decision=decision)

    assert event is not None
    assert event.event_type == EVENT_GATE_DECISION
    assert event.session_id == "sess-1"

    records = _read_jsonl(log_path)
    assert len(records) == 1
    rec = records[0]
    # Schema: timestamp, gate name, decision summary, capability-tag
    # snapshot, session ID.
    assert rec["event_type"] == EVENT_GATE_DECISION
    assert "timestamp" in rec
    assert rec["session_id"] == "sess-1"
    assert rec["gate"] == "G2"
    assert rec["decision"] == {
        "allow": False,
        "reason": "blocked",
        "incident_level": 2,
        "has_rewritten_args": True,
    }
    # Capability tag is preserved verbatim under its own top-level key.
    assert rec["capability_tag"]["source"] == "tool:foo"
    assert rec["capability_tag"]["sensitivity"] == "internal"
    assert rec["capability_tag"]["taint"] is True
    assert rec["capability_tag"]["provenance_chain"] == ["origin"]
    assert rec["capability_tag"]["metadata"] == {"k": "v"}


def test_emit_gate_decision_omits_capability_tag_when_absent(tmp_path):
    log_path = tmp_path / "provenance.jsonl"
    settings = PalisadeSettings(
        enabled=True,
        provenance_log_path=str(log_path),
    )
    emitter = ProvenanceEmitter(settings)

    emitter.emit_gate_decision(
        gate="G2",
        decision=GateDecision(allow=True, reason="ok"),
    )

    rec = _read_jsonl(log_path)[0]
    assert "capability_tag" not in rec
    assert rec["decision"]["has_rewritten_args"] is False


def test_emit_incident_writes_jsonl(tmp_path):
    log_path = tmp_path / "provenance.jsonl"
    settings = PalisadeSettings(
        enabled=True,
        provenance_log_path=str(log_path),
    )
    emitter = ProvenanceEmitter(settings, session_id="sess-2")

    tag = CapabilityTag(source="user:alice")
    record = IncidentRecord(
        level=1,
        severity_name="SEV1",
        gate="G5",
        reason="above ceiling",
        capability_tag=tag,
        metadata={"tool": "submit_hpc_job"},
    )
    event = emitter.emit_incident(record)

    assert event is not None
    assert event.event_type == EVENT_INCIDENT

    rec = _read_jsonl(log_path)[0]
    assert rec["event_type"] == EVENT_INCIDENT
    assert rec["session_id"] == "sess-2"
    assert rec["level"] == 1
    assert rec["severity_name"] == "SEV1"
    assert rec["gate"] == "G5"
    assert rec["reason"] == "above ceiling"
    assert rec["capability_tag"]["source"] == "user:alice"
    assert rec["metadata"] == {"tool": "submit_hpc_job"}


def test_emit_appends_multiple_events_to_same_file(tmp_path):
    log_path = tmp_path / "nested" / "dir" / "provenance.jsonl"
    settings = PalisadeSettings(
        enabled=True,
        provenance_log_path=str(log_path),
    )
    emitter = ProvenanceEmitter(settings)

    emitter.emit_gate_decision("G1", GateDecision(allow=True))
    emitter.emit_gate_decision("G2", GateDecision(allow=False, reason="x"))
    emitter.emit_incident(
        IncidentRecord(level=3, severity_name="SEV3", gate="G3", reason="info")
    )

    records = _read_jsonl(log_path)
    assert len(records) == 3
    assert [r["event_type"] for r in records] == [
        EVENT_GATE_DECISION,
        EVENT_GATE_DECISION,
        EVENT_INCIDENT,
    ]


def test_last_event_tracks_most_recent_emit(tmp_path):
    log_path = tmp_path / "provenance.jsonl"
    settings = PalisadeSettings(
        enabled=True,
        provenance_log_path=str(log_path),
    )
    emitter = ProvenanceEmitter(settings)

    assert emitter.last_event is None

    emitter.emit_gate_decision("G1", GateDecision(allow=True))
    assert emitter.last_event is not None
    assert emitter.last_event.event_type == EVENT_GATE_DECISION

    emitter.emit_incident(
        IncidentRecord(level=2, severity_name="SEV2", gate="G2", reason="x")
    )
    assert emitter.last_event.event_type == EVENT_INCIDENT


def test_explicit_log_path_overrides_settings(tmp_path):
    settings_path = tmp_path / "from_settings.jsonl"
    override_path = tmp_path / "from_override.jsonl"
    settings = PalisadeSettings(
        enabled=True,
        provenance_log_path=str(settings_path),
    )
    emitter = ProvenanceEmitter(settings, log_path=override_path)

    emitter.emit_gate_decision("G1", GateDecision(allow=True))

    assert not settings_path.exists()
    assert override_path.exists()


def test_emit_logs_via_logger_when_no_log_path(caplog):
    """With no path configured, events go through the module logger."""
    settings = PalisadeSettings(enabled=True)
    emitter = ProvenanceEmitter(settings, session_id="sess-3")

    with caplog.at_level("INFO", logger="palisade.provenance"):
        emitter.emit_gate_decision("G1", GateDecision(allow=True, reason="ok"))

    assert any(
        "PALISADE provenance" in rec.getMessage() for rec in caplog.records
    )


# -----------------------------------------------------------------
# Flowcept broker path
# -----------------------------------------------------------------


class _FakeSink:
    """Records events; optionally fails to exercise the fallback path."""

    def __init__(self, *, fail: bool = False) -> None:
        self.events: list = []
        self.closed = False
        self._fail = fail

    def emit(self, event) -> None:
        if self._fail:
            raise RuntimeError("broker unreachable")
        self.events.append(event)

    def close(self) -> None:
        self.closed = True


def _flowcept_settings(**kw):
    return PalisadeSettings(
        enabled=True, flowcept_enabled=True,
        flowcept_endpoint="redis://broker.local:6379", **kw,
    )


def test_flowcept_enabled_routes_to_sink_not_jsonl(tmp_path):
    """With Flowcept enabled, emission goes to the broker sink and the
    JSONL file is not written."""
    sink = _FakeSink()
    log = tmp_path / "prov.jsonl"
    emitter = ProvenanceEmitter(_flowcept_settings(), sink=sink, log_path=log)

    event = emitter.emit_incident(
        IncidentRecord(level=1, severity_name="SEV1", gate="G2", reason="boom")
    )

    assert event is not None
    assert len(sink.events) == 1
    assert sink.events[0].payload["gate"] == "G2"
    assert emitter.last_event is event
    assert not log.exists()  # broker path bypasses the JSONL log


def test_flowcept_sink_failure_falls_back_to_jsonl(tmp_path, caplog):
    """A broker failure degrades to the durable JSONL log (and logs once)."""
    sink = _FakeSink(fail=True)
    log = tmp_path / "prov.jsonl"
    emitter = ProvenanceEmitter(_flowcept_settings(), sink=sink, log_path=log)

    with caplog.at_level("ERROR"):
        emitter.emit_gate_decision("G2", GateDecision(allow=False, reason="x"))

    assert log.exists()
    record = json.loads(log.read_text().strip())
    assert record["event_type"] == EVENT_GATE_DECISION
    assert any("Flowcept emission failed" in r.message for r in caplog.records)

    # Subsequent emits skip the broker entirely and go straight to JSONL.
    emitter.emit_gate_decision("G1", GateDecision(allow=True))
    assert len(log.read_text().strip().splitlines()) == 2
    assert len(sink.events) == 0


def test_default_sink_is_flowcept_sink_when_enabled():
    from palisade.provenance import FlowceptSink

    emitter = ProvenanceEmitter(_flowcept_settings())
    assert isinstance(emitter._sink, FlowceptSink)


def test_no_sink_when_flowcept_disabled(tmp_path):
    emitter = ProvenanceEmitter(
        PalisadeSettings(enabled=True), log_path=tmp_path / "p.jsonl"
    )
    assert emitter._sink is None


def test_flowcept_sink_missing_package_raises_importerror():
    """The Flowcept adapter imports lazily; absent the package, the first
    emit raises a clear, actionable ImportError."""
    from palisade.provenance import EVENT_INCIDENT, FlowceptSink, ProvenanceEvent

    sink = FlowceptSink("redis://broker.local:6379", session_id="s1")
    event = ProvenanceEvent(
        event_type=EVENT_INCIDENT, timestamp="2026-01-01T00:00:00.000Z",
        session_id="s1", payload={"gate": "G2"},
    )
    with pytest.raises(ImportError, match="flowcept"):
        sink.emit(event)


def test_emitter_degrades_to_jsonl_when_flowcept_missing(tmp_path):
    """End-to-end with the real (default) FlowceptSink: since flowcept is
    not installed, the first emit's ImportError degrades to JSONL."""
    log = tmp_path / "prov.jsonl"
    emitter = ProvenanceEmitter(_flowcept_settings(), log_path=log)
    emitter.emit_incident(
        IncidentRecord(level=2, severity_name="SEV2", gate="G3", reason="contract")
    )
    assert log.exists()
    assert json.loads(log.read_text().strip())["gate"] == "G3"


def test_close_flushes_sink_and_is_safe_without_one():
    sink = _FakeSink()
    emitter = ProvenanceEmitter(_flowcept_settings(), sink=sink)
    emitter.close()
    assert sink.closed is True
    # No sink configured -> close() is a no-op, not an error.
    ProvenanceEmitter(PalisadeSettings(enabled=True)).close()


# -----------------------------------------------------------------
# FlowceptSink: documented MQ publish path
# -----------------------------------------------------------------


def _install_fake_flowcept(monkeypatch):
    """Inject a fake `flowcept` module so FlowceptSink._ensure_started
    imports our stand-ins instead of the real (uninstalled) package.
    Returns a dict that records the controller/task interactions."""
    calls: dict = {"tasks": []}

    class FakeFlowcept:
        def __init__(self, **kw):
            calls["controller_kwargs"] = kw

        def start(self):
            calls["started"] = True

        def stop(self):
            calls["stopped"] = True

    class FakeFlowceptTask:
        def __init__(self, **kw):
            self.init_kwargs = kw
            self.end_kwargs = None
            calls["tasks"].append(self)

        def end(self, **kw):
            self.end_kwargs = kw

    fake_mod = types.ModuleType("flowcept")
    fake_mod.Flowcept = FakeFlowcept
    fake_mod.FlowceptTask = FakeFlowceptTask
    monkeypatch.setitem(sys.modules, "flowcept", fake_mod)
    return calls


def test_flowcept_sink_publishes_via_flowcept_task(monkeypatch):
    """The sink starts a workflow controller keyed on the session id and
    publishes each event as a FlowceptTask finalized with end(generated=...)
    -- Flowcept's documented MQ capture path, not a direct DB write."""
    from palisade.provenance import EVENT_GATE_DECISION, FlowceptSink, ProvenanceEvent

    for var in ("MQ_URI", "KVDB_URI", "MQ_ENABLED", "MONGO_ENABLED"):
        monkeypatch.delenv(var, raising=False)
    calls = _install_fake_flowcept(monkeypatch)

    sink = FlowceptSink(
        "redis://broker.local:6379", session_id="sess-1", start_persistence=True
    )
    sink.emit(
        ProvenanceEvent(
            event_type=EVENT_GATE_DECISION,
            timestamp="2026-06-30T00:00:00.000Z",
            session_id="sess-1",
            payload={"gate": "G2", "tier": "fast", "decision": {"allow": False}},
        )
    )

    # Endpoint mapped onto Flowcept's documented MQ env knobs before import.
    assert os.environ["MQ_URI"] == "redis://broker.local:6379"
    assert os.environ["MQ_ENABLED"] == "true"
    # DB persistence is left to Flowcept's own config (no MONGO_ENABLED
    # forcing, which would conflict with offline-flush settings files).
    assert os.environ.get("MONGO_ENABLED") is None

    # Controller grouped under the session id and started once.
    assert calls["controller_kwargs"]["workflow_id"] == "sess-1"
    assert calls["controller_kwargs"]["start_persistence"] is True
    assert calls["started"] is True

    # Exactly one task, published via end(generated=...) with our payload.
    assert len(calls["tasks"]) == 1
    task = calls["tasks"][0]
    assert task.init_kwargs["activity_id"] == EVENT_GATE_DECISION
    assert task.init_kwargs["used"]["session_id"] == "sess-1"
    assert "palisade" in task.init_kwargs["tags"]
    assert task.init_kwargs["custom_metadata"]["palisade_timestamp"] == (
        "2026-06-30T00:00:00.000Z"
    )
    assert task.end_kwargs["generated"]["gate"] == "G2"


def test_flowcept_sink_does_not_override_operator_env(monkeypatch):
    """An operator who has already configured Flowcept's MQ wins: the sink
    fills in only unset routing vars."""
    from palisade.provenance import EVENT_INCIDENT, FlowceptSink, ProvenanceEvent

    monkeypatch.setenv("MQ_URI", "redis://operator-broker:6379")
    monkeypatch.delenv("MQ_ENABLED", raising=False)
    _install_fake_flowcept(monkeypatch)

    sink = FlowceptSink("redis://palisade-default:6379", session_id="s")
    sink.emit(
        ProvenanceEvent(
            event_type=EVENT_INCIDENT, timestamp="t", session_id="s", payload={}
        )
    )
    assert os.environ["MQ_URI"] == "redis://operator-broker:6379"


# -----------------------------------------------------------------
# Gate dispatch emits decisions to the provenance bus
# -----------------------------------------------------------------


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_gate_dispatch_emits_fast_and_slow_decisions(tmp_path):
    """Every enabled gate's check_fast / check_slow dispatch emits the
    decision (allow included) to the provenance emitter on the context,
    tagged with the producing tier -- so the audit bundle records gate
    decisions, not just incidents."""
    from palisade.capabilities import CapabilityRegistry
    from palisade.gates.base import GateContext, PassThroughGate
    from palisade.trust import TrustScorer

    settings = PalisadeSettings(enabled=True)
    log = tmp_path / "prov.jsonl"
    emitter = ProvenanceEmitter(settings, log_path=log)
    ctx = GateContext(
        capability_registry=CapabilityRegistry(),
        trust_scorer=TrustScorer(settings),
        provenance=emitter,
    )
    gate = PassThroughGate()

    fast = await gate.check_fast({"x": 1}, ctx)
    await gate.check_slow({"x": 1}, ctx, fast)

    records = [json.loads(line) for line in log.read_text().strip().splitlines()]
    assert [r["tier"] for r in records] == ["fast", "slow"]
    assert all(r["event_type"] == EVENT_GATE_DECISION for r in records)
    assert all(r["gate"] == "PassThrough" for r in records)
    assert all(r["decision"]["allow"] is True for r in records)


@pytest.mark.anyio
async def test_disabled_gate_emits_no_decision(tmp_path):
    """A disabled gate's pass-through is not provenance-worthy."""
    from palisade.capabilities import CapabilityRegistry
    from palisade.gates.base import GateContext, PassThroughGate
    from palisade.trust import TrustScorer

    settings = PalisadeSettings(enabled=True)
    log = tmp_path / "prov.jsonl"
    emitter = ProvenanceEmitter(settings, log_path=log)
    ctx = GateContext(
        capability_registry=CapabilityRegistry(),
        trust_scorer=TrustScorer(settings),
        provenance=emitter,
    )
    gate = PassThroughGate(enabled=False)

    await gate.check_fast({"x": 1}, ctx)

    assert not log.exists()
