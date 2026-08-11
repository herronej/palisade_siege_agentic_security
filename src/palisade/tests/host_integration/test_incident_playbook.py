"""
Tests for the incident playbook.

The playbook wires `IncidentManager` actions through `TrustScorer`:

  * SEV1 terminates the session (tier pinned TERMINATED; the agent loop
    then refuses every further request);
  * SEV2 elevates the session tier and sets the re-auth-required flag;
  * SEV3 logs only -- it nudges the Bayesian score but takes no discrete
    playbook action (no termination, no elevation, no re-auth flag).

The scorer-level tests pin the playbook semantics; the integration tests
drive a real `ProjectAgent.run_stream` (with a stub model) to prove that
a SEV1 raised from a G2 gate denies the rest of the session.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

pytestmark = pytest.mark.anyio

from pydantic_ai.models.function import FunctionModel

from vista_backend.agents.agents import ProjectAgent, ProjectAgentResultEvent
from palisade.host import HostProject, HostUser
from palisade.config import PalisadeSettings
from palisade.incidents import IncidentManager
from palisade.capabilities.registry import TrustTier
from palisade.trust import TrustScorer


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# -----------------------------------------------------------------
# Scorer-level playbook semantics
# -----------------------------------------------------------------


def _scorer() -> TrustScorer:
    return TrustScorer(PalisadeSettings(enabled=True))


def test_sev1_terminates_the_session() -> None:
    scorer = _scorer()
    scorer.notify_incident(level=1, gate="G2")
    assert scorer.terminated is True
    assert scorer.current_tier() is TrustTier.TERMINATED
    assert scorer.current_tier_for("G2") is TrustTier.TERMINATED
    # Every capability is forced TERMINATED while the session is dead.
    assert scorer.current_tier_for("G3") is TrustTier.TERMINATED


def test_sev2_elevates_tier_and_sets_reauth_flag() -> None:
    scorer = _scorer()
    scorer.notify_incident(level=2, gate="G2")
    assert scorer.terminated is False
    assert scorer.reauth_required is True
    # The session floor is at least ELEVATED for every capability.
    assert scorer.current_tier() is not TrustTier.NORMAL
    assert scorer.current_tier_for("G3") is not TrustTier.NORMAL


def test_sev3_logs_without_behavioral_change() -> None:
    scorer = _scorer()
    scorer.notify_incident(level=3, gate="G2")
    assert scorer.terminated is False
    assert scorer.reauth_required is False
    # No tier transition: SEV3 only nudges the score.
    assert scorer.current_tier() is TrustTier.NORMAL


def test_termination_is_permanent() -> None:
    """Clean calls and re-auth cannot revive a terminated session."""
    scorer = _scorer()
    scorer.notify_incident(level=1, gate="G2")
    for _ in range(50):
        scorer.record_clean_call("G2")
    assert scorer.current_tier() is TrustTier.TERMINATED
    assert scorer.reauthenticate() == ()
    assert scorer.terminated is True
    assert scorer.current_tier() is TrustTier.TERMINATED


def test_reauth_clears_sev2_escalation() -> None:
    scorer = _scorer()
    scorer.notify_incident(level=2, gate="G2")
    assert scorer.reauth_required is True
    assert scorer.current_tier() is not TrustTier.NORMAL

    scorer.reauthenticate()
    assert scorer.reauth_required is False
    assert scorer.current_tier() is TrustTier.NORMAL


def test_reset_clears_playbook_state() -> None:
    scorer = _scorer()
    scorer.notify_incident(level=1, gate="G2")
    scorer.reset()
    assert scorer.terminated is False
    assert scorer.reauth_required is False
    assert scorer.current_tier() is TrustTier.NORMAL


def test_incident_manager_record_drives_the_playbook() -> None:
    """The wiring point: IncidentManager.record -> notify_incident."""
    settings = PalisadeSettings(enabled=True)
    scorer = TrustScorer(settings)
    mgr = IncidentManager(settings, trust_scorer=scorer)

    mgr.record(level=2, gate="G2", reason="warning")
    assert scorer.reauth_required is True
    assert scorer.terminated is False

    mgr.record(level=1, gate="G2", reason="severe")
    assert scorer.terminated is True


def test_disabled_master_flag_makes_record_a_noop() -> None:
    settings = PalisadeSettings(enabled=False)
    scorer = TrustScorer(settings)
    mgr = IncidentManager(settings, trust_scorer=scorer)
    mgr.record(level=1, gate="G2", reason="severe")
    assert scorer.terminated is False
    assert scorer.current_tier() is TrustTier.NORMAL


# -----------------------------------------------------------------
# Integration: SEV1 from a G2 gate denies the rest of the session
# -----------------------------------------------------------------


def _make_project() -> HostProject:
    return HostProject(
        id=uuid.uuid4(), name="playbook", description=None, system_prompt=None,
        skills=[], knowledge_bases=[], tools=[], usage_limits={},
    )


def _make_user() -> HostUser:
    return HostUser(
        id=uuid.uuid4(), email="tester@example.com", is_admin=False,
    )


class _ModelSpy:
    """Stub streaming model that counts how often it is invoked."""

    def __init__(self) -> None:
        self.calls = 0

    def as_model(self) -> FunctionModel:
        async def stream_fn(messages, info):  # noqa: ANN001 - framework signature
            self.calls += 1
            yield "ok"

        return FunctionModel(stream_function=stream_fn)


async def _run(agent: ProjectAgent, prompt: str, spy: _ModelSpy) -> list[Any]:
    events: list[Any] = []
    with agent.agent.override(model=spy.as_model(), toolsets=[]):
        async for event in agent.run_stream(prompt, enable_elicitation=False):
            events.append(event)
    return events


def _enable(monkeypatch: pytest.MonkeyPatch, **flags: bool) -> None:
    from vista_backend.agents import agents as agents_module

    monkeypatch.setattr(agents_module.settings.palisade, "enabled", True)
    monkeypatch.setattr(agents_module.settings.palisade, "quarantine_enabled", False)
    for gate in ("g1", "g2", "g3", "g4", "g5"):
        monkeypatch.setattr(
            agents_module.settings.palisade, f"{gate}_enabled",
            flags.get(f"{gate}_enabled", False),
        )


def _terminals(events: list[Any]) -> list[ProjectAgentResultEvent]:
    return [e for e in events if isinstance(e, ProjectAgentResultEvent)]


async def test_sev1_from_g2_gate_denies_rest_of_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(monkeypatch, g1_enabled=True)
    agent = ProjectAgent(_make_project(), _make_user())

    # Before the incident, a benign request reaches the model.
    spy_before = _ModelSpy()
    await _run(agent, "FLiBe density at 873 K", spy_before)
    assert spy_before.calls == 1
    assert agent.sidecar.trust_scorer.terminated is False

    # A G2 gate trips a SEV1 (e.g. credential exfiltration in a tool call).
    agent.sidecar.incident_manager.record(
        level=1, gate="G2", reason="credential exfiltration",
    )
    assert agent.sidecar.trust_scorer.terminated is True
    assert agent.sidecar.trust_scorer.current_tier() is TrustTier.TERMINATED

    # Every subsequent request is refused without touching the model.
    spy_after = _ModelSpy()
    events = await _run(agent, "another benign question", spy_after)
    assert spy_after.calls == 0
    terminals = _terminals(events)
    assert len(terminals) == 1
    assert terminals[0].result.new_messages == []


async def test_termination_is_enforced_even_with_g1_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run_stream guard is gate-agnostic: termination holds even when
    G1's before-run hook is not in the chain."""
    _enable(monkeypatch)  # master on, every gate off
    agent = ProjectAgent(_make_project(), _make_user())

    agent.sidecar.incident_manager.record(level=1, gate="G2", reason="severe")
    assert agent.sidecar.trust_scorer.terminated is True

    spy = _ModelSpy()
    events = await _run(agent, "benign", spy)
    assert spy.calls == 0
    assert _terminals(events)[0].result.new_messages == []


async def test_sev2_elevates_but_session_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(monkeypatch, g1_enabled=True)
    agent = ProjectAgent(_make_project(), _make_user())

    agent.sidecar.incident_manager.record(level=2, gate="G2", reason="warning")
    ts = agent.sidecar.trust_scorer
    assert ts.terminated is False
    assert ts.reauth_required is True
    assert ts.current_tier() is not TrustTier.NORMAL

    # SEV2 does not stop the session: the next request still reaches the model.
    spy = _ModelSpy()
    await _run(agent, "benign", spy)
    assert spy.calls == 1
