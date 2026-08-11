"""
Unit tests for `PalisadeCapability` (issue R0).

This module lands only the shared base class -- no gate hooks -- so these
tests cover the three things the base class is responsible for:

1. It is a genuine `pydantic_ai.capabilities.AbstractCapability` and an
   empty subclass works when handed to an `Agent` (the "smoke test").
2. Its constructor stores the (sidecar, gate, settings) collaborators.
3. `is_enabled()` AND-s the master flag with the per-gate flag.
"""

from __future__ import annotations

import uuid

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel

from palisade.host import HostProject
from palisade.config import PalisadeSettings
from palisade.gates.base import Gate, GateContext, GateDecision, PassThroughGate
from palisade.sidecar import PalisadeSidecar
from palisade.capabilities.base import PalisadeCapability


# -----------------------------------------------------------------
# Fixtures / helpers
# -----------------------------------------------------------------


def _make_project(name: str = "test-project") -> HostProject:
    """Construct a minimal valid `HostProject` for sidecar construction."""
    return HostProject(
        id=uuid.uuid4(),
        name=name,
        description=None,
        system_prompt=None,
        skills=[],
        knowledge_bases=[],
        tools=[],
        usage_limits={},
    )


def _make_sidecar(settings: PalisadeSettings) -> PalisadeSidecar:
    return PalisadeSidecar(settings, _make_project())


class _G1Gate(Gate):
    """A minimal gate named ``G1`` so the derived flag name resolves to
    ``g1_enabled``."""

    name = "G1"

    async def _check_fast_when_enabled(
        self, payload: object, ctx: GateContext
    ) -> GateDecision:  # pragma: no cover - not exercised in R0
        return GateDecision(allow=True)


class _EmptyCapability(PalisadeCapability):
    """An otherwise-empty subclass -- registers no hooks."""


# -----------------------------------------------------------------
# AbstractCapability conformance / smoke test
# -----------------------------------------------------------------


def test_subclasses_abstract_capability():
    """Acceptance: PalisadeCapability subclasses AbstractCapability."""
    assert issubclass(PalisadeCapability, AbstractCapability)


def test_empty_subclass_smoke():
    """
    Acceptance: an empty subclass passes the AbstractCapability smoke
    test -- it can be handed to an `Agent` and the run completes.

    A no-op capability registers no hooks, so the run must be
    byte-identical to a run with no capability at all.
    """
    settings = PalisadeSettings(enabled=True, g1_enabled=True)
    cap = _EmptyCapability(_make_sidecar(settings), _G1Gate(), settings)

    agent = Agent(TestModel(), capabilities=[cap])
    result = agent.run_sync("hello")

    assert isinstance(cap, AbstractCapability)
    assert result.output == "success (no tool calls)"


# -----------------------------------------------------------------
# Constructor
# -----------------------------------------------------------------


def test_constructor_stores_collaborators():
    """Acceptance: constructor takes (sidecar, gate, settings) and exposes them."""
    settings = PalisadeSettings(enabled=True, g1_enabled=True)
    sidecar = _make_sidecar(settings)
    gate = _G1Gate()

    cap = _EmptyCapability(sidecar, gate, settings)

    assert cap.sidecar is sidecar
    assert cap.gate is gate
    assert cap.settings is settings


# -----------------------------------------------------------------
# is_enabled()
# -----------------------------------------------------------------


def test_is_enabled_false_when_master_flag_off():
    """Master flag off -> disabled even if the per-gate flag is on."""
    settings = PalisadeSettings(enabled=False, g1_enabled=True)
    cap = _EmptyCapability(_make_sidecar(settings), _G1Gate(), settings)
    assert cap.is_enabled() is False


def test_is_enabled_false_when_gate_flag_off():
    """Per-gate flag off -> disabled even when the master flag is on."""
    settings = PalisadeSettings(enabled=True, g1_enabled=False)
    cap = _EmptyCapability(_make_sidecar(settings), _G1Gate(), settings)
    assert cap.is_enabled() is False


def test_is_enabled_true_when_both_on():
    """Master flag and per-gate flag both on -> enabled."""
    settings = PalisadeSettings(enabled=True, g1_enabled=True)
    cap = _EmptyCapability(_make_sidecar(settings), _G1Gate(), settings)
    assert cap.is_enabled() is True


def test_per_gate_flag_derived_from_gate_name():
    """
    With no explicit `gate_flag`, the flag name is derived from the
    gate's `name`: a gate named ``G1`` is gated by ``g1_enabled``.
    """
    settings = PalisadeSettings(enabled=True, g1_enabled=True, g2_enabled=False)
    cap = _EmptyCapability(_make_sidecar(settings), _G1Gate(), settings)
    assert cap._gate_flag_name() == "g1_enabled"
    assert cap.is_enabled() is True


def test_explicit_gate_flag_overrides_derivation():
    """An explicit `gate_flag` class attribute wins over name derivation."""

    class _ExplicitCapability(PalisadeCapability):
        gate_flag = "g2_enabled"

    settings = PalisadeSettings(enabled=True, g1_enabled=True, g2_enabled=False)
    # Gate is named "G1" but the capability pins g2_enabled, which is off.
    cap = _ExplicitCapability(_make_sidecar(settings), _G1Gate(), settings)
    assert cap._gate_flag_name() == "g2_enabled"
    assert cap.is_enabled() is False


def test_unknown_gate_name_treated_as_disabled():
    """
    A gate whose name has no matching settings field (e.g. the
    `PassThroughGate` test fixture) resolves to a missing flag and is
    treated as disabled rather than raising.
    """
    settings = PalisadeSettings(enabled=True, g1_enabled=True)
    cap = _EmptyCapability(_make_sidecar(settings), PassThroughGate(), settings)
    assert cap._gate_flag_name() == "passthrough_enabled"
    assert cap.is_enabled() is False
