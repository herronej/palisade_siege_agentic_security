"""
Unit tests for `G4CodeCapability` (issue R2).

These exercise the capability's hook surface against a real `Agent` and
a `TestModel` (no real LLM, no real Semgrep -- the gate's
``_run_semgrep`` / ``check_fast`` is stubbed):

- ``before_tool_execute`` runs only for code-bearing tools, denies on an
  ERROR finding (`SkipToolExecution` + SEV2), logs a WARNING-only
  finding (SEV3) while letting the call proceed, and default-denies on a
  Semgrep timeout.
- ``prepare_tools`` drops a code-bearing tool whose per-tool override is
  malformed (Bell-LaPadula) and keeps tools with valid/absent overrides.
- A disabled capability is a no-op.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

pytestmark = pytest.mark.anyio

from pydantic_ai import Agent
from pydantic_ai.tools import ToolDefinition

from palisade.host import HostProjectModel
from palisade.config import PalisadeSettings
from palisade.gates.g4_code import G4CodeGate, SemgrepFinding
from palisade.sidecar import PalisadeSidecar
from palisade.capabilities.g4_code import G4CodeCapability


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# -----------------------------------------------------------------
# Fixtures / helpers
# -----------------------------------------------------------------


def _make_project() -> HostProject:
    return HostProjectModel(
        id=uuid.uuid4(),
        name="g4-capability",
        description=None,
        system_prompt=None,
        skills=[],
        knowledge_bases=[],
        tools=[],
        usage_limits={},
    )


def _make_sidecar(*, enabled: bool = True, g4_enabled: bool = True) -> PalisadeSidecar:
    return PalisadeSidecar(
        PalisadeSettings(enabled=enabled, g4_enabled=g4_enabled), _make_project()
    )


def _finding(severity: str) -> SemgrepFinding:
    return SemgrepFinding(
        check_id=f"vista-{severity.lower()}",
        severity=severity,
        message=f"{severity} finding",
        start_line=1,
        end_line=1,
    )


def _gate_with_findings(
    findings: list[SemgrepFinding] | None = None,
    *,
    per_tool_overrides: dict | None = None,
) -> G4CodeGate:
    gate = G4CodeGate(
        enabled=True,
        semgrep_enabled=True,
        per_tool_overrides=per_tool_overrides,
    )

    async def fake_semgrep(code: str, language: str) -> list[SemgrepFinding]:
        return list(findings or [])

    gate._run_semgrep = fake_semgrep  # type: ignore[method-assign]
    return gate


def _agent_with_run_bash(
    sidecar: PalisadeSidecar, gate: G4CodeGate
) -> tuple[Agent, dict]:
    """Agent whose only tool is `run_bash`, driven by TestModel."""
    from pydantic_ai.models.test import TestModel

    state = {"ran": False}
    cap = G4CodeCapability(sidecar, gate, sidecar.settings)
    agent = Agent(TestModel(), capabilities=[cap])

    @agent.tool_plain
    def run_bash(command: str) -> str:
        "Run a bash command in the sandbox."
        state["ran"] = True
        return "ran ok"

    return agent, state


# -----------------------------------------------------------------
# before_tool_execute: severity routing
# -----------------------------------------------------------------


async def test_error_finding_skips_execution_and_records_sev2() -> None:
    """AC2: an ERROR finding raises SkipToolExecution (tool not run)
    and records a SEV2 incident."""
    sidecar = _make_sidecar()
    gate = _gate_with_findings([_finding("ERROR")])
    agent, state = _agent_with_run_bash(sidecar, gate)

    result = await agent.run("run something")

    assert state["ran"] is False
    incident = sidecar.incident_manager.last_incident
    assert incident is not None and incident.level == 2 and incident.gate == "G4"
    # The deny message is surfaced as the tool result the model sees.
    assert any(
        "PALISADE G4 denied" in str(getattr(p, "content", ""))
        for m in result.all_messages()
        for p in getattr(m, "parts", [])
    )


async def test_warning_finding_records_sev3_and_proceeds() -> None:
    """AC2: a WARNING-only finding records SEV3 and lets the call run."""
    sidecar = _make_sidecar()
    gate = _gate_with_findings([_finding("WARNING")])
    agent, state = _agent_with_run_bash(sidecar, gate)

    await agent.run("run something")

    assert state["ran"] is True
    incident = sidecar.incident_manager.last_incident
    assert incident is not None and incident.level == 3 and incident.gate == "G4"


async def test_clean_scan_proceeds_without_incident() -> None:
    """No findings -> the call runs and no incident is recorded."""
    sidecar = _make_sidecar()
    gate = _gate_with_findings([])
    agent, state = _agent_with_run_bash(sidecar, gate)

    await agent.run("run something")

    assert state["ran"] is True
    assert sidecar.incident_manager.last_incident is None


async def test_timeout_default_denies_sev2() -> None:
    """AC4: a Semgrep scan exceeding the timeout default-denies SEV2.

    The timeout is bounded at the hook (anyio.fail_after), not inside
    the runner.
    """
    sidecar = _make_sidecar()
    gate = _gate_with_findings([])
    gate._semgrep_timeout = 0.05  # type: ignore[attr-defined]

    async def slow_check_fast(payload, ctx):
        await asyncio.sleep(5)
        raise AssertionError("should have timed out")

    gate.check_fast = slow_check_fast  # type: ignore[method-assign]
    agent, state = _agent_with_run_bash(sidecar, gate)

    await agent.run("run something")

    assert state["ran"] is False
    incident = sidecar.incident_manager.last_incident
    assert incident is not None and incident.level == 2
    assert "timed out" in incident.reason


# -----------------------------------------------------------------
# Tool filter
# -----------------------------------------------------------------


async def test_non_code_tool_is_not_scanned() -> None:
    """A tool outside scan_tools bypasses the gate entirely."""
    from pydantic_ai.models.test import TestModel

    sidecar = _make_sidecar()
    gate = _gate_with_findings([_finding("ERROR")])

    async def boom(payload, ctx):
        raise AssertionError("check_fast must not run for non-code tools")

    gate.check_fast = boom  # type: ignore[method-assign]

    state = {"ran": False}
    cap = G4CodeCapability(sidecar, gate, sidecar.settings)
    agent = Agent(TestModel(), capabilities=[cap])

    @agent.tool_plain
    def summarize(text: str) -> str:
        "Summarize text."
        state["ran"] = True
        return "summary"

    await agent.run("summarize this")
    assert state["ran"] is True


# -----------------------------------------------------------------
# prepare_tools: per-tool override filtering (Bell-LaPadula)
# -----------------------------------------------------------------


async def test_prepare_tools_drops_tool_with_malformed_override() -> None:
    """AC3: a code-bearing tool with a malformed per-tool override is
    dropped from the toolset; others are kept."""
    sidecar = _make_sidecar()
    # `run_bash` override names a non-`vista-*` rule -> malformed.
    gate = _gate_with_findings(
        per_tool_overrides={"run_bash": ["p/security-audit.some-rule"]}
    )
    cap = G4CodeCapability(sidecar, gate, sidecar.settings)

    tool_defs = [
        ToolDefinition(name="run_bash"),
        ToolDefinition(name="create_file"),
        ToolDefinition(name="summarize"),
    ]
    kept = await cap.prepare_tools(_ctx(), tool_defs)
    names = {td.name for td in kept}

    assert "run_bash" not in names  # dropped (Bell-LaPadula)
    assert names == {"create_file", "summarize"}


async def test_prepare_tools_keeps_tool_with_valid_override() -> None:
    """A well-formed `vista-*` override keeps the tool available."""
    sidecar = _make_sidecar()
    gate = _gate_with_findings(
        per_tool_overrides={"run_bash": ["vista-revshell"]}
    )
    cap = G4CodeCapability(sidecar, gate, sidecar.settings)

    tool_defs = [ToolDefinition(name="run_bash"), ToolDefinition(name="create_file")]
    kept = await cap.prepare_tools(_ctx(), tool_defs)

    assert {td.name for td in kept} == {"run_bash", "create_file"}


# -----------------------------------------------------------------
# Disabled capability is a no-op
# -----------------------------------------------------------------


async def test_disabled_capability_does_not_scan() -> None:
    """With the master flag off the capability is inert: an ERROR
    finding does not stop the call."""
    sidecar = _make_sidecar(enabled=False, g4_enabled=True)
    gate = _gate_with_findings([_finding("ERROR")])
    agent, state = _agent_with_run_bash(sidecar, gate)

    await agent.run("run something")

    assert state["ran"] is True
    assert sidecar.incident_manager.last_incident is None


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------


def _ctx():
    """A throwaway RunContext stand-in: `prepare_tools` ignores ctx, so
    `None` is sufficient for the direct-call tests."""
    return None  # type: ignore[return-value]
