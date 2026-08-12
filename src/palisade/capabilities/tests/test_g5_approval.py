"""
Unit tests for the G5 <-> approval-capability integration.

G5 gates ``submit_hpc_job`` behind human approval using PydanticAI's
native deferred-tool-calls mechanism: after the fast + slow tiers pass,
``G5HpcCapability.wrap_tool_execute`` stashes the fast-tier decision
metadata on the sidecar and raises ``ApprovalRequired``;
``PalisadeApprovalCapability`` resolves it via a human callback.

Acceptance criteria covered:

- The approval flow uses deferred_tool_calls (not a bespoke path): a
  ``submit_hpc_job`` call defers and is resolved by the approval
  capability.
- The approval request carries the resolved SLURM script + G5 policy
  check results (the sidecar side-channel).
- Declining produces a clean denial (the tool does not run; no
  exception escapes), and increments the sticky-on-decline counter
  without firing an incident.
- Backward compatibility: with G5 disabled the capability is inert and
  no deferral happens.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

pytestmark = pytest.mark.anyio

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

from palisade.host import HostProjectModel
from palisade.config import PalisadeSettings
from palisade.gates.g5_hpc import G5HpcJobGate
from palisade.sidecar import PalisadeSidecar
from palisade.capabilities import ApprovalOutcome, PalisadeApprovalCapability
from palisade.capabilities.g5_hpc import G5HpcCapability


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_sidecar(*, enabled: bool = True, g5_enabled: bool = True) -> PalisadeSidecar:
    project = HostProjectModel(
        id=uuid.uuid4(), name="g5-approval", description=None, system_prompt=None,
        skills=[], knowledge_bases=[], tools=[], usage_limits={},
    )
    return PalisadeSidecar(
        PalisadeSettings(enabled=enabled, g5_enabled=g5_enabled), project
    )


_CLEAN_SCRIPT = (
    "#SBATCH --account=approved-research\n#SBATCH --partition=gpu\n"
    "#SBATCH --nodes=4\nsrun python forge-tune.py\n"
)


def _run_submit(
    sidecar: PalisadeSidecar,
    *,
    approve: bool,
    gate: G5HpcJobGate | None = None,
    script: str = _CLEAN_SCRIPT,
):
    """
    Drive a ``submit_hpc_job`` call through a real Agent with the G5 +
    approval capabilities. Returns (tool_ran, captured_metadata).
    """
    gate = gate or G5HpcJobGate(enabled=True)
    g5 = G5HpcCapability(sidecar, gate, sidecar.settings)

    captured: dict[str, Any] = {}

    async def request_approval(*, tool_name: str, tool_call_id: str, args: Any):
        # Mirror agents.py's emitter: read the stashed metadata, then
        # record the outcome (clears metadata + sticky-on-decline).
        captured["metadata"] = sidecar.pending_approval_metadata.get(tool_call_id)
        captured["tool_name"] = tool_name
        sidecar.note_approval_outcome(tool_call_id, approved=approve)
        return ApprovalOutcome(
            approved=approve, message="Declined by user." if not approve else ""
        )

    approval = PalisadeApprovalCapability(request_approval=request_approval)
    state = {"ran": False}
    agent = Agent(TestModel(), capabilities=[g5, approval])

    @agent.tool_plain
    def submit_hpc_job(slurm_script: str) -> str:
        "Submit a SLURM job."
        state["ran"] = True
        return "job-1"

    return agent, state, captured


# -----------------------------------------------------------------
# Deferral + approval
# -----------------------------------------------------------------


async def test_submit_defers_for_approval_and_runs_on_accept() -> None:
    sidecar = _make_sidecar()
    agent, state, captured = _run_submit(sidecar, approve=True)
    await agent.run("submit the job")
    # The approval capability was consulted for submit_hpc_job...
    assert captured.get("tool_name") == "submit_hpc_job"
    # ...and on accept the tool executed.
    assert state["ran"] is True
    assert sidecar.sticky_decline_count == 0


async def test_approval_request_carries_g5_decision_metadata() -> None:
    # Invoke wrap_tool_execute directly so the submitted script is
    # controlled (TestModel would synthesize a placeholder arg).
    from pydantic_ai.exceptions import ApprovalRequired

    sidecar = _make_sidecar()
    gate = G5HpcJobGate(enabled=True)
    g5 = G5HpcCapability(sidecar, gate, sidecar.settings)

    class _Ctx:
        tool_call_approved = False

    class _Call:
        tool_call_id = "tc-1"

    async def handler(_args: Any) -> str:
        raise AssertionError("handler must not run before approval")

    with pytest.raises(ApprovalRequired):
        await g5.wrap_tool_execute(
            _Ctx(),  # type: ignore[arg-type]
            call=_Call(),  # type: ignore[arg-type]
            tool_def=ToolDefinition(name="submit_hpc_job"),
            args={"slurm_script": _CLEAN_SCRIPT},
            handler=handler,
        )
    meta = sidecar.pending_approval_metadata.get("tc-1")
    assert meta is not None
    assert meta["gate"] == "G5"
    assert "forge-tune.py" in meta["resolved_script"]
    assert meta["resource_ceiling_passed"] is True
    assert meta["binary_denylist_match"] is None
    assert "allocation" in meta["checks_passed"]
    assert meta["requested_nodes"] == 4


async def test_decline_blocks_tool_cleanly_and_marks_sticky() -> None:
    sidecar = _make_sidecar()
    agent, state, captured = _run_submit(sidecar, approve=False)
    # A clean run (no exception escapes); the tool simply doesn't run.
    result = await agent.run("submit the job")
    assert state["ran"] is False
    assert result is not None
    # Sticky-on-decline recorded WITHOUT an incident.
    assert sidecar.sticky_decline_count == 1
    assert sidecar.incident_manager.last_incident is None
    # A sticky decline mark was written to the registry.
    sticky = [
        tag
        for _vid, tag in sidecar.capability_registry.find(
            source_pattern="tool:submit_hpc_job"
        )
        if tag.metadata.get("g5_check") == "user_decline"
    ]
    assert len(sticky) == 1
    assert sticky[0].metadata.get("sticky") is True


async def test_decline_message_surfaced_to_model() -> None:
    sidecar = _make_sidecar()
    agent, state, captured = _run_submit(sidecar, approve=False)
    result = await agent.run("submit the job")
    assert any(
        "Declined by user" in str(getattr(p, "content", ""))
        for m in result.all_messages()
        for p in getattr(m, "parts", [])
    )


# -----------------------------------------------------------------
# Backward compatibility / inertness
# -----------------------------------------------------------------


async def test_g5_disabled_does_not_defer() -> None:
    # With G5 off the capability is inert: no metadata stashed, the
    # approval callback is never consulted, the tool runs directly (the
    # legacy MCP elicitation path remains responsible for confirmation).
    sidecar = _make_sidecar(enabled=False, g5_enabled=True)
    agent, state, captured = _run_submit(sidecar, approve=False)
    await agent.run("submit the job")
    assert state["ran"] is True
    assert captured == {}
    assert sidecar.sticky_decline_count == 0


async def test_require_submit_approval_false_skips_deferral() -> None:
    sidecar = _make_sidecar()
    gate = G5HpcJobGate(enabled=True)
    g5 = G5HpcCapability(
        sidecar, gate, sidecar.settings, require_submit_approval=False
    )
    consulted = {"v": False}

    async def request_approval(**kwargs: Any):
        consulted["v"] = True
        return ApprovalOutcome(approved=False)

    approval = PalisadeApprovalCapability(request_approval=request_approval)
    state = {"ran": False}
    agent = Agent(TestModel(), capabilities=[g5, approval])

    @agent.tool_plain
    def submit_hpc_job(slurm_script: str) -> str:
        "Submit a SLURM job."
        state["ran"] = True
        return "job-1"

    await agent.run("submit", message_history=[])
    assert state["ran"] is True
    assert consulted["v"] is False


# -----------------------------------------------------------------
# A denied fast/slow tier never reaches the approval step
# -----------------------------------------------------------------


async def test_mining_submission_denied_before_approval() -> None:
    # A mining script is denied by the fast tier (before_tool_execute);
    # the approval step is never reached.
    sidecar = _make_sidecar()
    gate = G5HpcJobGate(enabled=True)
    g5 = G5HpcCapability(sidecar, gate, sidecar.settings)
    consulted = {"v": False}

    async def request_approval(**kwargs: Any):
        consulted["v"] = True
        return ApprovalOutcome(approved=True)

    approval = PalisadeApprovalCapability(request_approval=request_approval)
    state = {"ran": False}
    agent = Agent(TestModel(), capabilities=[g5, approval])

    @agent.tool_plain
    def submit_hpc_job(slurm_script: str) -> str:
        "Submit a SLURM job."
        state["ran"] = True
        return "job-1"

    # before_tool_execute runs the gate's check_fast on TestModel's
    # generated args, which won't be a mining script; to make this
    # deterministic, stub the fast tier to deny.
    from palisade.gates.base import GateDecision

    async def deny(payload, ctx):
        return GateDecision(allow=False, reason="mining IOC", incident_level=1)

    gate.check_fast = deny  # type: ignore[method-assign]
    await agent.run("submit the job")
    assert state["ran"] is False
    assert consulted["v"] is False  # approval never reached
