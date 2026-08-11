"""
Tests for `PalisadeApprovalCapability` (issue R6).

Two levels:

- Unit: the capability's ``handle_deferred_tool_calls`` maps approval
  outcomes to `ToolApproved` / `ToolDenied`, fails closed on a callback
  error, and no-ops when there are no approval requests.
- Integration: a `ProjectAgent` with a ``requires_approval=True`` tool
  surfaces an `McpToolApprovalEvent` and is resolved through the
  existing ``resolve_elicitation(action, content)`` surface (AC2).
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.anyio

from pydantic_ai import DeferredToolRequests, ToolApproved, ToolDenied
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel

from vista_backend.agents.agents import (
    McpToolApprovalEvent,
    ProjectAgent,
    ProjectAgentResultEvent,
)
from palisade.host import HostProject, HostUser
from palisade.capabilities.approval import ApprovalOutcome, PalisadeApprovalCapability


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# -----------------------------------------------------------------
# Unit: handle_deferred_tool_calls
# -----------------------------------------------------------------


def _requests(*tool_call_ids: str) -> DeferredToolRequests:
    return DeferredToolRequests(
        approvals=[
            ToolCallPart(tool_name="submit_hpc_job", args={"script": "x"}, tool_call_id=tcid)
            for tcid in tool_call_ids
        ]
    )


async def test_no_approvals_returns_none() -> None:
    async def never(**_: object) -> ApprovalOutcome:  # pragma: no cover
        raise AssertionError("should not be called")

    cap = PalisadeApprovalCapability(request_approval=never)
    assert await cap.handle_deferred_tool_calls(None, DeferredToolRequests()) is None


async def test_approved_call_maps_to_tool_approved() -> None:
    async def approve(**_: object) -> ApprovalOutcome:
        return ApprovalOutcome(approved=True, override_args={"script": "clean"})

    cap = PalisadeApprovalCapability(request_approval=approve)
    results = await cap.handle_deferred_tool_calls(None, _requests("c1"))

    assert isinstance(results.approvals["c1"], ToolApproved)
    assert results.approvals["c1"].override_args == {"script": "clean"}


async def test_denied_call_maps_to_tool_denied() -> None:
    async def deny(**_: object) -> ApprovalOutcome:
        return ApprovalOutcome(approved=False, message="nope")

    cap = PalisadeApprovalCapability(request_approval=deny)
    results = await cap.handle_deferred_tool_calls(None, _requests("c1"))

    assert isinstance(results.approvals["c1"], ToolDenied)
    assert results.approvals["c1"].message == "nope"


async def test_callback_error_fails_closed() -> None:
    async def boom(**_: object) -> ApprovalOutcome:
        raise RuntimeError("approval channel exploded")

    cap = PalisadeApprovalCapability(request_approval=boom)
    results = await cap.handle_deferred_tool_calls(None, _requests("c1"))

    assert isinstance(results.approvals["c1"], ToolDenied)


async def test_resolves_each_request_independently() -> None:
    async def by_id(*, tool_call_id: str, **_: object) -> ApprovalOutcome:
        return ApprovalOutcome(approved=(tool_call_id == "c1"))

    cap = PalisadeApprovalCapability(request_approval=by_id)
    results = await cap.handle_deferred_tool_calls(None, _requests("c1", "c2"))

    assert isinstance(results.approvals["c1"], ToolApproved)
    assert isinstance(results.approvals["c2"], ToolDenied)


# -----------------------------------------------------------------
# Integration: ProjectAgent + requires_approval + resolve_elicitation
# -----------------------------------------------------------------


def _make_project() -> HostProject:
    return HostProject(
        id=uuid.uuid4(),
        name="approval",
        description=None,
        system_prompt=None,
        skills=[],
        knowledge_bases=[],
        tools=[],
        usage_limits={},
    )


def _make_user() -> HostUser:
    return HostUser(id=uuid.uuid4(), email="t@e.com", is_admin=False)


async def _run_with_approval(decision: str, *, enable_elicitation: bool = True):
    pa = ProjectAgent(_make_project(), _make_user())
    state = {"ran": False}

    @pa.agent.tool_plain(requires_approval=True)
    def submit_hpc_job(script: str) -> str:
        "Submit an HPC job to the cluster."
        state["ran"] = True
        return "job-42"

    events = []
    with pa.agent.override(model=TestModel(), toolsets=[]):
        async for ev in pa.run_stream("submit a job", enable_elicitation=enable_elicitation):
            events.append(ev)
            if isinstance(ev, McpToolApprovalEvent):
                pa.resolve_elicitation(ev.elicitation_id, decision)
    return state["ran"], events


async def test_requires_approval_accept_runs_tool() -> None:
    ran, events = await _run_with_approval("accept")
    approvals = [e for e in events if isinstance(e, McpToolApprovalEvent)]
    assert len(approvals) == 1
    assert approvals[0].tool_name == "submit_hpc_job"
    assert ran is True
    assert any(isinstance(e, ProjectAgentResultEvent) for e in events)


async def test_requires_approval_decline_blocks_tool() -> None:
    ran, events = await _run_with_approval("decline")
    assert [e for e in events if isinstance(e, McpToolApprovalEvent)]
    assert ran is False


async def test_no_approval_channel_fails_closed() -> None:
    """With elicitation disabled there is no approval channel, so a
    requires_approval tool is denied (fail closed) -- no approval event
    is emitted."""
    ran, events = await _run_with_approval("accept", enable_elicitation=False)
    assert not [e for e in events if isinstance(e, McpToolApprovalEvent)]
    assert ran is False
