"""
Unit tests for `PalisadeSidecar`.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

# `anyio`'s pytest plugin (auto-loaded with the `anyio` dependency)
# requires async tests to be marked. A module-level `pytestmark`
# applies the marker to every test in the file; the few sync tests
# inherit it harmlessly.
pytestmark = pytest.mark.anyio

from palisade.host import HostProjectModel
from palisade.capabilities import CapabilityRegistry
from palisade.config import PalisadeSettings
from palisade.gates.base import PassThroughGate
from palisade.incidents import IncidentManager
from palisade.provenance import ProvenanceEmitter
from palisade.sidecar import PalisadeSidecar
from palisade.trust import TrustScorer


# -----------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------


def _make_project(name: str = "test-project") -> HostProject:
    """Construct a minimal valid `HostProject` for sidecar tests."""
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


# -----------------------------------------------------------------
# Construction
# -----------------------------------------------------------------


def test_constructs_when_master_flag_off():
    """Acceptance: constructs without error when enabled=False."""
    settings = PalisadeSettings(enabled=False)
    sidecar = PalisadeSidecar(settings, _make_project())

    # Collaborators are wired even with the master flag off; the
    # individual components are responsible for their own no-op
    # behavior.
    assert isinstance(sidecar.capability_registry, CapabilityRegistry)
    assert isinstance(sidecar.trust_scorer, TrustScorer)
    assert isinstance(sidecar.incident_manager, IncidentManager)
    assert isinstance(sidecar.provenance, ProvenanceEmitter)


def test_constructs_when_master_flag_on():
    settings = PalisadeSettings(enabled=True)
    sidecar = PalisadeSidecar(settings, _make_project())
    assert sidecar.settings is settings
    assert sidecar.project.name == "test-project"


def test_constructor_propagates_flowcept_misconfiguration():
    """ProvenanceEmitter's fail-fast bubbles up through the sidecar."""
    settings = PalisadeSettings(
        enabled=True,
        flowcept_enabled=True,
        flowcept_endpoint=None,
    )
    with pytest.raises(ValueError, match="flowcept_endpoint"):
        PalisadeSidecar(settings, _make_project())


def test_default_gate_set_is_empty_with_all_flags_off():
    """
    `_build_gates()` returns an empty dict when every per-gate
    flag is False, regardless of the master flag.

    Replaces the original "always empty" test now that
    G2 ships -- the modularity contract is now "no gates
    *unless* their flag is on," not "no gates at all."
    """
    settings = PalisadeSettings(enabled=True)  # all g{N}_enabled default False
    sidecar = PalisadeSidecar(settings, _make_project())
    assert dict(sidecar.gates) == {}


def test_g2_gate_is_built_when_g2_enabled():
    """`g2_enabled=True` populates `gates["G2"]`."""
    settings = PalisadeSettings(enabled=True, g2_enabled=True)
    sidecar = PalisadeSidecar(settings, _make_project())
    assert "G2" in sidecar.gates
    assert sidecar.is_gate_enabled("G2") is True


def test_self_consistency_samples_wire_to_every_slow_gate():
    """`quarantine_self_consistency_samples` reaches G1 *and* G2/G3/G4/G5.

    Regression guard: the setting was previously threaded only into G1, so the
    G2/G3 sanitize and G4/G5 code-intent slow tiers silently ran single-sample
    regardless of the configured value.
    """
    settings = PalisadeSettings(
        enabled=True,
        g1_enabled=True,
        g2_enabled=True,
        g3_enabled=True,
        g4_enabled=True,
        g5_enabled=True,
        quarantine_self_consistency_samples=3,
    )
    gates = PalisadeSidecar(settings, _make_project()).gates
    assert gates["G1"]._intent_self_consistency_samples == 3
    assert gates["G2"]._self_consistency_samples == 3
    assert gates["G3"]._self_consistency_samples == 3
    assert gates["G4"]._code_intent_self_consistency_samples == 3
    assert gates["G5"]._code_intent_self_consistency_samples == 3


def test_collaborators_are_distinct_instances_per_sidecar():
    """Two sidecars get independent state -- session-scoped contract."""
    settings = PalisadeSettings(enabled=True)
    s1 = PalisadeSidecar(settings, _make_project())
    s2 = PalisadeSidecar(settings, _make_project())

    assert s1.capability_registry is not s2.capability_registry
    assert s1.trust_scorer is not s2.trust_scorer
    assert s1.provenance is not s2.provenance


# -----------------------------------------------------------------
# is_active() predicate
# -----------------------------------------------------------------


def test_is_active_false_when_master_flag_off():
    settings = PalisadeSettings(enabled=False)
    sidecar = PalisadeSidecar(settings, _make_project())
    assert sidecar.is_active() is False


def test_is_active_false_when_on_with_no_gates():
    """Master flag on but gate set empty -> still not active."""
    settings = PalisadeSettings(enabled=True)
    sidecar = PalisadeSidecar(settings, _make_project())
    assert sidecar.is_active() is False


def test_is_active_true_when_on_with_at_least_one_gate():
    """Inject a gate to simulate active behavior."""
    settings = PalisadeSettings(enabled=True)
    sidecar = PalisadeSidecar(settings, _make_project())
    # Direct injection: once gates exist, `_build_gates()` will populate
    # this dict from the flags; here we simulate that to verify the
    # predicate without depending on a not-yet-written gate class.
    sidecar._gates["G2"] = PassThroughGate(enabled=True)
    assert sidecar.is_active() is True


def test_is_active_false_when_off_even_with_gates():
    """Master flag off dominates over a populated gate set."""
    settings = PalisadeSettings(enabled=False)
    sidecar = PalisadeSidecar(settings, _make_project())
    sidecar._gates["G2"] = PassThroughGate(enabled=True)
    assert sidecar.is_active() is False


# -----------------------------------------------------------------
# is_gate_enabled(name) predicate
# -----------------------------------------------------------------


def test_all_gates_ship_when_enabled():
    """Every gate G1..G6 builds when its flag is on. G4 is the merged
    sandbox/code gate (the former G7 is folded into it), so there is no
    separate G7 gate."""
    settings = PalisadeSettings(
        enabled=True,
        g1_enabled=True,
        g2_enabled=True,
        g3_enabled=True,
        g4_enabled=True,
        g5_enabled=True,
        g6_enabled=True,
    )
    sidecar = PalisadeSidecar(settings, _make_project())
    for name in ("G1", "G2", "G3", "G4", "G5", "G6"):
        assert sidecar.is_gate_enabled(name) is True
    # The retired G7 gate no longer exists as a separate gate.
    assert sidecar.is_gate_enabled("G7") is False


def test_g7_flag_is_deprecated_alias_for_g4():
    """The retired ``g7_enabled`` flag folds into G4 so the deterministic
    sandbox checks are never silently dropped on upgrade."""
    settings = PalisadeSettings(enabled=True, g7_enabled=True)
    sidecar = PalisadeSidecar(settings, _make_project())
    assert sidecar.is_gate_enabled("G4") is True
    assert sidecar.is_gate_enabled("G7") is False


def test_is_gate_enabled_true_for_g3_when_enabled():
    """g3_enabled=True populates `gates["G3"]`."""
    settings = PalisadeSettings(enabled=True, g3_enabled=True)
    sidecar = PalisadeSidecar(settings, _make_project())
    assert sidecar.is_gate_enabled("G3") is True
    assert sidecar.is_active() is True


def test_is_gate_enabled_true_for_g5_when_enabled():
    """g5_enabled=True populates `gates["G5"]`."""
    settings = PalisadeSettings(enabled=True, g5_enabled=True)
    sidecar = PalisadeSidecar(settings, _make_project())
    assert sidecar.is_gate_enabled("G5") is True
    assert sidecar.is_active() is True


def test_is_gate_enabled_true_when_gate_injected():
    settings = PalisadeSettings(enabled=True)
    sidecar = PalisadeSidecar(settings, _make_project())
    sidecar._gates["G2"] = PassThroughGate(enabled=True)
    assert sidecar.is_gate_enabled("G2") is True
    assert sidecar.is_gate_enabled("G1") is False


def test_is_gate_enabled_false_for_unknown_name():
    """Unknown names return False rather than raising."""
    settings = PalisadeSettings(enabled=True)
    sidecar = PalisadeSidecar(settings, _make_project())
    assert sidecar.is_gate_enabled("not-a-gate") is False


# -----------------------------------------------------------------
# build_capabilities factory
# -----------------------------------------------------------------


def test_build_capabilities_empty_when_flag_off():
    """Flag-off -> no gates -> empty capability list (byte-identical)."""
    settings = PalisadeSettings(enabled=False)
    sidecar = PalisadeSidecar(settings, _make_project())
    assert sidecar.build_capabilities() == []


def test_build_capabilities_returns_g1_and_g3_when_enabled():
    """G1 and G3 gates -> their capabilities; the Q-LLM is stored."""
    from palisade.capabilities import G1PromptCapability, G3RagCapability

    settings = PalisadeSettings(enabled=True, g1_enabled=True, g3_enabled=True)
    sidecar = PalisadeSidecar(settings, _make_project())

    sentinel = object()
    caps = sidecar.build_capabilities(quarantine_agent=sentinel)

    assert sidecar.quarantine_agent is sentinel
    kinds = {type(c) for c in caps}
    assert G1PromptCapability in kinds
    assert G3RagCapability in kinds


def test_g5_submit_approval_autonomous_by_default():
    from palisade.capabilities import G5HpcCapability

    settings = PalisadeSettings(enabled=True, g5_enabled=True)
    sidecar = PalisadeSidecar(settings, _make_project())
    g5 = next(
        c for c in sidecar.build_capabilities() if isinstance(c, G5HpcCapability)
    )
    # No human approver in this deployment: submit_hpc_job is gated
    # autonomously by the policy tiers, not held for elicitation.
    assert g5._require_submit_approval is False


def test_g5_submit_approval_opt_in():
    from palisade.capabilities import G5HpcCapability

    settings = PalisadeSettings(
        enabled=True, g5_enabled=True, g5_require_submit_approval=True
    )
    sidecar = PalisadeSidecar(settings, _make_project())
    g5 = next(
        c for c in sidecar.build_capabilities() if isinstance(c, G5HpcCapability)
    )
    assert g5._require_submit_approval is True


def test_g2_tool_gate_is_wired_when_enabled():
    from palisade.capabilities import G2ToolCapability

    settings = PalisadeSettings(enabled=True, g2_enabled=True)
    sidecar = PalisadeSidecar(settings, _make_project())
    assert sidecar.is_gate_enabled("G2") is True
    assert any(
        isinstance(c, G2ToolCapability) for c in sidecar.build_capabilities()
    )


def test_g2_absent_when_flag_off():
    from palisade.capabilities import G2ToolCapability

    settings = PalisadeSettings(enabled=True)  # g2 default off
    sidecar = PalisadeSidecar(settings, _make_project())
    assert not any(
        isinstance(c, G2ToolCapability) for c in sidecar.build_capabilities()
    )


# -----------------------------------------------------------------
# Anyio backend selection -- the async tests above default to
# trio + asyncio; pin to asyncio so a missing trio dep doesn't
# cause spurious failures.
# -----------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"
