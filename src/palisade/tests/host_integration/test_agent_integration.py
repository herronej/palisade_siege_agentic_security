"""
Integration test for PalisadeSidecar wiring into ProjectAgent.

Verifies the structural contract that survived the capability
migration: `ProjectAgent.__init__` instantiates a `PalisadeSidecar`
wired to the project and the live `settings.palisade` sub-model.

(The earlier `_compose_process_tool_call` composition tests were removed
when PALISADE migrated onto PydanticAI's `AbstractCapability` + Hooks
API; the capability-hook behavior that replaced it is covered by
`test_g1_integration.py` and the per-capability tests.)
"""

from __future__ import annotations

import uuid

from vista_backend.agents.agents import ProjectAgent
from palisade.host import HostProjectModel, HostUserModel
from palisade.sidecar import PalisadeSidecar


def _make_project(name: str = "agent-integration") -> HostProject:
    """Construct a minimal valid `HostProject`."""
    return HostProjectModel(
        id=uuid.uuid4(),
        name=name,
        description=None,
        system_prompt=None,
        skills=[],
        knowledge_bases=[],
        tools=[],
        usage_limits={},
    )


def _make_user() -> HostUser:
    return HostUserModel(
        id=uuid.uuid4(), email="tester@example.com", is_admin=False,
    )


def test_project_agent_owns_a_sidecar():
    project = _make_project()
    agent = ProjectAgent(project, _make_user())

    assert isinstance(agent._sidecar, PalisadeSidecar)
    assert agent._sidecar.project is project


def test_sidecar_reads_live_palisade_settings():
    """The sidecar's settings come from `settings.palisade`."""
    from vista_backend.agents import agents as agents_module

    agent = ProjectAgent(_make_project(), _make_user())
    assert agent._sidecar.settings is agents_module.settings.palisade
