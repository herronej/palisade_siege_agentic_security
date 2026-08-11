"""
Unit tests for `G2ToolGate._check_fast_when_enabled`.

Covers all six acceptance criteria of the work item
`Implement G2 fast-tier: schema validation, allow-list,
capability propagation`:

1. `G2ToolGate.check_fast(payload, ctx)` where
   `payload = {"tool_name": str, "args": dict}` returns a
   `GateDecision`.
2. Schema validation: malformed args -> deny with
   `incident_level=2`.
3. Allow-list check: tool name not in project's `tools` patterns
   -> deny with `incident_level=3`.
4. Capability propagation: tool returns get tagged
   `tool-output:<tool_name>, taint=True, source=tool:<tool_name>`.
5. High-stakes guard: any argument with `taint=True` going into a
   high-stakes tool -> deny with `incident_level=2`.
6. Unit tests cover each path.

The tests build the gate directly (no sidecar) so each AC has a
focused fixture. A separate suite (`test_g2_annotations.py`,
`test_sidecar.py`) covers the sidecar's `_build_gates` wiring;
that's not duplicated here.
"""

from __future__ import annotations

from typing import Any

import pytest

# anyio's pytest plugin requires async tests to be marked. A
# module-level pytestmark applies to every async test below;
# sync tests inherit the marker harmlessly.
pytestmark = pytest.mark.anyio

from palisade.capabilities import (
    CapabilityRegistry,
    CapabilityTag,
    DualUseMarker,
    SensitivityTier,
)
from palisade.config import PalisadeSettings
from palisade.gates.base import GateContext
from palisade.gates.g2_tool import (
    HIGH_STAKES_FALLBACK,
    G2ToolGate,
    _iter_string_values,
    _tool_allowed,
)
from palisade.trust import TrustScorer


# -----------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------


def _ctx(registry: CapabilityRegistry | None = None) -> GateContext:
    """Build a minimal `GateContext` for fast-tier tests."""
    return GateContext(
        capability_registry=registry or CapabilityRegistry(),
        trust_scorer=TrustScorer(PalisadeSettings(enabled=True)),
    )


def _gate(
    *,
    enabled: bool = True,
    allow_patterns: list[str] | None = None,
    high_stakes: frozenset[str] = HIGH_STAKES_FALLBACK,
    schemas: dict[str, dict[str, Any]] | None = None,
) -> G2ToolGate:
    """
    Build a G2 gate with explicit dependencies.

    Defaults: enabled, allow-list `["*"]` (so the allow-list check
    is a no-op unless overridden), the standard fallback
    high-stakes set, no schemas.
    """
    return G2ToolGate(
        enabled=enabled,
        allow_patterns=allow_patterns if allow_patterns is not None else ["*"],
        high_stakes=high_stakes,
        schemas=schemas or {},
    )


# Schema fixtures used by multiple tests; expressed inline so the
# tests themselves remain self-contained.

_RUN_BASH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {"type": "string"},
        "timeout": {"type": "integer", "minimum": 1},
    },
    "required": ["command"],
}

_RAG_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kb_slug": {"type": "string"},
        "query": {"type": "string"},
        "n_results": {"type": "integer"},
    },
    "required": ["kb_slug", "query"],
}


# -----------------------------------------------------------------
# AC 1: shape -- payload is {"tool_name", "args"}; returns
# GateDecision
# -----------------------------------------------------------------


async def test_check_fast_returns_gate_decision_for_valid_payload() -> None:
    """AC1: the public surface is a `GateDecision` for the
    well-formed payload shape."""
    gate = _gate()
    decision = await gate.check_fast(
        {"tool_name": "rag_search", "args": {"q": "hi"}}, _ctx()
    )
    assert decision.allow is True
    assert decision.reason  # populated on allow for the audit trail


async def test_check_fast_raises_on_bad_payload_shape() -> None:
    """
    A malformed payload is a *programming* error from the
    sidecar, not a runtime policy denial. Raise so it surfaces
    early; the runtime should never construct this shape.
    """
    gate = _gate()
    ctx = _ctx()

    with pytest.raises(TypeError, match="dict"):
        await gate.check_fast("not a dict", ctx)
    with pytest.raises(TypeError, match="tool_name"):
        await gate.check_fast({"tool_name": 42, "args": {}}, ctx)
    with pytest.raises(TypeError, match="args"):
        await gate.check_fast({"tool_name": "rag_search", "args": []}, ctx)


async def test_disabled_gate_allows_without_running_logic() -> None:
    """
    The base-class dispatch guarantees a disabled gate returns
    `allow=True` without ever calling
    `_check_fast_when_enabled`. Verifying here pins the
    modularity contract -- a disabled G2 cannot crash the agent
    even with a fully malformed payload.
    """
    gate = _gate(enabled=False, allow_patterns=["never_match_anything"])
    decision = await gate.check_fast(
        {"tool_name": "run_bash", "args": {"command": "rm -rf /"}}, _ctx()
    )
    assert decision.allow is True
    assert "disabled" in decision.reason


# -----------------------------------------------------------------
# AC 3: allow-list -- check first because it's the cheapest
# (deterministic predicate, no taint walk or schema parse)
# -----------------------------------------------------------------


async def test_allow_list_denies_unlisted_tool_with_sev3() -> None:
    """
    AC3: a tool name not matched by the project's patterns is
    denied with `incident_level=3`. SEV3 because PydanticAI's
    filter is the primary defense; a hit here is belt-and-
    suspenders, not a session-terminating event.
    """
    gate = _gate(allow_patterns=["rag_search", "display_file"])
    decision = await gate.check_fast(
        {"tool_name": "run_bash", "args": {"command": "ls"}}, _ctx()
    )
    assert decision.allow is False
    assert decision.incident_level == 3
    assert "allow-list" in decision.reason
    assert "run_bash" in decision.reason


async def test_allow_list_honors_deny_patterns() -> None:
    """
    Deny patterns (`!name`) override the implicit `*` allow.
    Mirrors the augmentation `agents.py` already applies for
    projects with no knowledge bases (`!rag_search`).
    """
    gate = _gate(allow_patterns=["*", "!rag_search"])
    decision = await gate.check_fast(
        {"tool_name": "rag_search", "args": {"kb_slug": "x", "query": "q"}},
        _ctx(),
    )
    assert decision.allow is False
    assert decision.incident_level == 3


async def test_allow_list_glob_patterns_match() -> None:
    """fnmatch glob patterns are honored (matches `_tool_allowed`)."""
    gate = _gate(allow_patterns=["rag_*", "display_*"])
    decision = await gate.check_fast(
        {"tool_name": "rag_search", "args": {"kb_slug": "x", "query": "q"}},
        _ctx(),
    )
    assert decision.allow is True


async def test_empty_allow_list_means_allow_all() -> None:
    """
    Matches `_tool_allowed`'s contract: an empty allow-list
    implies the implicit `*` allow. Without this, a fresh project
    with no `tools` field would refuse every call.
    """
    gate = _gate(allow_patterns=[])
    decision = await gate.check_fast(
        {"tool_name": "anything", "args": {}}, _ctx()
    )
    assert decision.allow is True


def test_vendored_tool_allowed_matches_agents_module() -> None:
    """
    Vendored `_tool_allowed` MUST be semantically identical to
    `vista_backend.agents.agents._tool_allowed` (the function we
    avoid importing to dodge a circular import). Pin the
    equivalence so a future change in either copy fires here.
    """
    pytest.importorskip(
        "vista_backend",
        reason="cross-checks a symbol owned by the reference host application",
    )
    from vista_backend.agents.agents import _tool_allowed as agent_tool_allowed

    cases = [
        ("rag_search", ["rag_search"], True),
        ("rag_search", ["display_*"], False),
        ("rag_search", ["*", "!rag_search"], False),
        ("rag_search", [], True),  # empty -> implicit '*'
        ("rag_search", ["!rag_*"], False),
        ("display_file", ["display_*", "!display_secret"], True),
        ("display_secret", ["display_*", "!display_secret"], False),
    ]
    for name, patterns, expected in cases:
        assert (
            _tool_allowed(name, patterns) is expected
        ), f"_tool_allowed({name!r}, {patterns!r}) drift"
        assert (
            agent_tool_allowed(name, patterns) is expected
        ), f"agents._tool_allowed({name!r}, {patterns!r}) drift"


# -----------------------------------------------------------------
# AC 2: schema validation -- malformed args -> SEV2
# -----------------------------------------------------------------


async def test_schema_validates_well_formed_args() -> None:
    """Args that match the schema produce an allow decision."""
    gate = _gate(schemas={"run_bash": _RUN_BASH_SCHEMA})
    decision = await gate.check_fast(
        {"tool_name": "run_bash", "args": {"command": "ls", "timeout": 30}},
        _ctx(),
    )
    assert decision.allow is True


async def test_schema_denies_missing_required_field() -> None:
    """AC2: missing required field -> SEV2 deny."""
    gate = _gate(schemas={"run_bash": _RUN_BASH_SCHEMA})
    decision = await gate.check_fast(
        {"tool_name": "run_bash", "args": {"timeout": 30}}, _ctx()
    )
    assert decision.allow is False
    assert decision.incident_level == 2
    assert "schema" in decision.reason.lower()
    assert "command" in decision.reason


async def test_schema_denies_wrong_type() -> None:
    """AC2: type mismatch -> SEV2 deny."""
    gate = _gate(schemas={"run_bash": _RUN_BASH_SCHEMA})
    decision = await gate.check_fast(
        {"tool_name": "run_bash", "args": {"command": "ls", "timeout": "thirty"}},
        _ctx(),
    )
    assert decision.allow is False
    assert decision.incident_level == 2
    assert "timeout" in decision.reason


async def test_schema_denies_constraint_violation() -> None:
    """`minimum: 1` constraint catches out-of-range integers."""
    gate = _gate(schemas={"run_bash": _RUN_BASH_SCHEMA})
    decision = await gate.check_fast(
        {"tool_name": "run_bash", "args": {"command": "ls", "timeout": 0}},
        _ctx(),
    )
    assert decision.allow is False
    assert decision.incident_level == 2


async def test_no_cached_schema_falls_back_to_structural_floor() -> None:
    """
    A tool with no cached schema (discovery failed or
    third-party tool) passes through with only the structural
    floor: args must be a dict with string keys.
    """
    gate = _gate(schemas={})  # no schema for `unknown_tool`
    decision = await gate.check_fast(
        {"tool_name": "unknown_tool", "args": {"a": 1, "b": [2, 3]}},
        _ctx(),
    )
    assert decision.allow is True


async def test_schema_floor_catches_non_string_keys() -> None:
    """
    Even with no cached schema, non-string keys are denied. JSON
    Schema requires string keys on the wire; accepting non-string
    keys would let an attacker smuggle structural anomalies
    through unannotated tools.
    """
    gate = _gate(schemas={})
    decision = await gate.check_fast(
        {"tool_name": "unknown", "args": {42: "value"}}, _ctx()
    )
    assert decision.allow is False
    assert decision.incident_level == 2
    assert "non-string" in decision.reason


async def test_schema_lists_multiple_errors_in_reason() -> None:
    """Multiple errors are collected so the audit trail captures all."""
    gate = _gate(schemas={"run_bash": _RUN_BASH_SCHEMA})
    decision = await gate.check_fast(
        # missing `command`, wrong type for `timeout`
        {"tool_name": "run_bash", "args": {"timeout": "x"}},
        _ctx(),
    )
    assert decision.allow is False
    assert "2 validation error(s)" in decision.reason


async def test_invalid_cached_schema_denies_with_clear_reason() -> None:
    """
    A malformed cached schema should not crash -- deny with an
    explicit operator-facing message. Default-deny is the right
    posture if the schema itself can't be trusted.
    """
    # Invalid: "type" must be a string or list of strings.
    bad_schema = {"type": 123}
    gate = _gate(schemas={"bad_tool": bad_schema})
    decision = await gate.check_fast(
        {"tool_name": "bad_tool", "args": {}}, _ctx()
    )
    assert decision.allow is False
    assert decision.incident_level == 2
    assert "schema" in decision.reason.lower()


# -----------------------------------------------------------------
# AC 5: high-stakes guard -- tainted args + high-stakes tool ->
# SEV2 deny
# -----------------------------------------------------------------


async def test_tainted_arg_in_high_stakes_tool_denies_sev2() -> None:
    """
    AC5: G3-style tainted argument flowing into a high-stakes
    tool -> SEV2 deny. Models the cross-boundary attack vector:
    a poisoned RAG chunk text reused as a `run_bash` command.
    """
    registry = CapabilityRegistry()
    # The "G3 tagged this chunk as tainted" convention: value_id
    # is the string content.
    poisoned = "echo POISONED && rm -rf /"
    registry.tag(poisoned, CapabilityTag(source="rag:salt-papers", taint=True))

    gate = _gate(high_stakes=frozenset({"run_bash"}))
    decision = await gate.check_fast(
        {"tool_name": "run_bash", "args": {"command": poisoned}},
        _ctx(registry),
    )

    assert decision.allow is False
    assert decision.incident_level == 2
    assert "high-stakes" in decision.reason


async def test_tainted_arg_in_safe_tool_allows() -> None:
    """
    The taint walk runs for every tool, but only the
    high-stakes-tool branch denies. A tainted arg going into a
    *read-only* tool (rag_search) passes -- the gate's
    capability_tag still records the taint for downstream gates.
    """
    registry = CapabilityRegistry()
    chunk = "tainted but harmless string"
    registry.tag(chunk, CapabilityTag(source="rag:x", taint=True))

    gate = _gate(high_stakes=frozenset({"run_bash"}))
    decision = await gate.check_fast(
        {"tool_name": "rag_search", "args": {"kb_slug": "x", "query": chunk}},
        _ctx(registry),
    )
    assert decision.allow is True
    # The capability_tag on the *return* records that the call
    # consumed tainted inputs, so downstream gates see the
    # provenance.
    assert decision.capability_tag is not None
    assert decision.capability_tag.metadata["tainted_input_count"] == 1


async def test_clean_arg_in_high_stakes_tool_allows() -> None:
    """
    A high-stakes tool with no tainted inputs passes the guard.
    The capability_tag's `destructive=True` flag is set so
    downstream gates (trust escalation) can route on it.
    """
    gate = _gate(high_stakes=frozenset({"run_bash"}))
    decision = await gate.check_fast(
        {"tool_name": "run_bash", "args": {"command": "ls"}},
        _ctx(),
    )
    assert decision.allow is True
    assert decision.capability_tag is not None
    assert decision.capability_tag.metadata["destructive"] is True


async def test_taint_walk_recurses_into_nested_args() -> None:
    """
    Taint check walks into nested dicts and lists. Models the
    scenario where the LLM wraps a tainted string inside an
    `options` object or a list parameter.
    """
    registry = CapabilityRegistry()
    poisoned = "naughty-string"
    registry.tag(poisoned, CapabilityTag(source="rag:x", taint=True))

    gate = _gate(high_stakes=frozenset({"run_bash"}))
    decision = await gate.check_fast(
        {
            "tool_name": "run_bash",
            "args": {"command": "ls", "env": {"FOO": [poisoned, "bar"]}},
        },
        _ctx(registry),
    )
    assert decision.allow is False
    assert decision.incident_level == 2


async def test_tag_with_taint_false_does_not_trigger_guard() -> None:
    """
    Only `taint=True` tags trigger the high-stakes guard. A tag
    with `taint=False` (sanitized by a slow-tier check, for
    example) is treated as clean.
    """
    registry = CapabilityRegistry()
    cleaned = "previously-tainted-now-clean"
    registry.tag(
        cleaned,
        CapabilityTag(source="rag:x", taint=False),
    )

    gate = _gate(high_stakes=frozenset({"run_bash"}))
    decision = await gate.check_fast(
        {"tool_name": "run_bash", "args": {"command": cleaned}},
        _ctx(registry),
    )
    assert decision.allow is True


# -----------------------------------------------------------------
# AC 4: capability propagation -- tool returns tagged
# `source=tool:<name>, taint=True`
# -----------------------------------------------------------------


async def test_allow_decision_carries_capability_tag_for_return() -> None:
    """
    AC4: on allow, `decision.capability_tag` is the tag the
    sidecar will attach to the tool's *return value*.

    Pins source format (`tool:<name>`), default-deny taint
    (`True`), and the destructive flag from the high-stakes
    check.
    """
    gate = _gate(high_stakes=frozenset({"submit_hpc_job"}))
    decision = await gate.check_fast(
        {"tool_name": "submit_hpc_job", "args": {"template": "small"}},
        _ctx(),
    )

    assert decision.allow is True
    tag = decision.capability_tag
    assert tag is not None
    assert tag.source == "tool:submit_hpc_job"
    assert tag.taint is True
    assert tag.metadata["destructive"] is True


async def test_allow_decision_tag_records_clean_provenance() -> None:
    """A clean (no tainted-input) call records `clean_inputs` in
    its provenance chain so audit can distinguish clean from
    tainted-input tool returns."""
    gate = _gate(high_stakes=frozenset())  # nothing high-stakes
    decision = await gate.check_fast(
        {"tool_name": "rag_search", "args": {"kb_slug": "x", "query": "q"}},
        _ctx(),
    )
    assert decision.allow is True
    tag = decision.capability_tag
    assert tag is not None
    assert any(
        "clean_inputs" in step for step in tag.provenance_chain
    ), f"expected 'clean_inputs' marker in {tag.provenance_chain}"


async def test_allow_decision_tag_records_tainted_inputs() -> None:
    """
    Tainted-input value_ids appear in the return tag's
    `provenance_chain` so downstream consumers (the sidecar's
    apply-tag step, slow-tier checks) can trace the dependency.
    """
    registry = CapabilityRegistry()
    chunk = "tainted-snippet-A"
    registry.tag(chunk, CapabilityTag(source="rag:x", taint=True))

    gate = _gate(high_stakes=frozenset())  # not high-stakes so we allow
    decision = await gate.check_fast(
        {"tool_name": "summarize", "args": {"text": chunk}},
        _ctx(registry),
    )
    assert decision.allow is True
    tag = decision.capability_tag
    assert tag is not None
    assert tag.metadata["tainted_input_count"] == 1
    assert any(
        "tainted_inputs" in step and chunk in step
        for step in tag.provenance_chain
    )


async def test_default_deny_taint_on_return_tag() -> None:
    """
    Even when inputs are clean and the tool is read-only, the
    return tag has `taint=True`. The slow-tier Minimize-and-
    Sanitize check (follow-on issue) is responsible for
    clearing taint; the fast tier never produces a clean tag.
    """
    gate = _gate(high_stakes=frozenset())
    decision = await gate.check_fast(
        {"tool_name": "rag_search", "args": {"kb_slug": "x", "query": "q"}},
        _ctx(),
    )
    assert decision.allow is True
    assert decision.capability_tag is not None
    assert decision.capability_tag.taint is True


async def test_deny_decision_has_no_capability_tag() -> None:
    """
    A denial decision does not produce a capability_tag (no
    tool-call happens, so there's no return to tag). Documents
    the contract the sidecar relies on at the apply-tag step.
    """
    gate = _gate(allow_patterns=["nothing_else"])
    decision = await gate.check_fast(
        {"tool_name": "run_bash", "args": {"command": "ls"}}, _ctx()
    )
    assert decision.allow is False
    assert decision.capability_tag is None


# -----------------------------------------------------------------
# Ordering -- denial precedence (cheapest check first)
# -----------------------------------------------------------------


async def test_allow_list_check_runs_before_schema() -> None:
    """
    An unlisted tool with malformed args denies on the allow-list
    (SEV3), not on the schema (SEV2). The cheapest-first ordering
    is part of the gate's documented behavior so denials are
    routed to the right SEV bucket.
    """
    gate = _gate(
        allow_patterns=["only_this_tool"],
        schemas={"run_bash": _RUN_BASH_SCHEMA},
    )
    decision = await gate.check_fast(
        {"tool_name": "run_bash", "args": {"junk": True}},
        _ctx(),
    )
    assert decision.allow is False
    assert decision.incident_level == 3


async def test_schema_check_runs_before_taint_walk() -> None:
    """
    Malformed args for a high-stakes tool fall to schema-SEV2,
    not the taint guard. The taint check would never get a
    chance to run on garbage args anyway.
    """
    registry = CapabilityRegistry()
    chunk = "tainted"
    registry.tag(chunk, CapabilityTag(source="rag:x", taint=True))

    gate = _gate(
        schemas={"run_bash": _RUN_BASH_SCHEMA},
        high_stakes=frozenset({"run_bash"}),
    )
    decision = await gate.check_fast(
        {"tool_name": "run_bash", "args": {}},  # missing `command`
        _ctx(registry),
    )
    assert decision.allow is False
    assert decision.incident_level == 2
    assert "schema" in decision.reason.lower()


# -----------------------------------------------------------------
# rebind_metadata -- live update from sidecar populate
# -----------------------------------------------------------------


async def test_rebind_metadata_updates_high_stakes_and_schemas() -> None:
    """
    The sidecar's `populate_tool_metadata` rebinds the gate's
    snapshot via `G2ToolGate.rebind_metadata`. Verify the new
    metadata takes effect immediately on subsequent checks.
    """
    gate = _gate(high_stakes=frozenset(), schemas={})
    # Initially: `run_bash` is not high-stakes; tainted arg passes.
    registry = CapabilityRegistry()
    chunk = "tainted"
    registry.tag(chunk, CapabilityTag(source="rag:x", taint=True))

    before = await gate.check_fast(
        {"tool_name": "run_bash", "args": {"command": chunk}},
        _ctx(registry),
    )
    assert before.allow is True

    # Rebind to a high-stakes set that includes `run_bash`.
    gate.rebind_metadata(
        high_stakes=frozenset({"run_bash"}),
        schemas={"run_bash": _RUN_BASH_SCHEMA},
    )

    after = await gate.check_fast(
        {"tool_name": "run_bash", "args": {"command": chunk}},
        _ctx(registry),
    )
    assert after.allow is False
    assert after.incident_level == 2


# -----------------------------------------------------------------
# _iter_string_values -- helper unit tests so the taint walk has
# clean coverage
# -----------------------------------------------------------------


def test_iter_string_values_walks_nested_structure() -> None:
    args = {
        "a": "alpha",
        "b": {"c": "beta", "d": [1, "gamma", {"e": "delta"}]},
        "f": ["epsilon"],
        "g": 42,  # not a string -- skipped
        "h": None,  # skipped
    }
    found = set(_iter_string_values(args))
    assert found == {"alpha", "beta", "gamma", "delta", "epsilon"}


def test_iter_string_values_handles_empty_inputs() -> None:
    assert list(_iter_string_values({})) == []
    assert list(_iter_string_values([])) == []
    assert list(_iter_string_values("")) == [""]  # the empty string is a string
    assert list(_iter_string_values(None)) == []


# -----------------------------------------------------------------
# Settings ignored
# -----------------------------------------------------------------
# Defensive: the gate doesn't consult SensitivityTier / DualUseMarker
# in the fast tier; importing them and the trust scorer above
# makes sure the imports still work without forcing us to use the
# values in every test.
_UNUSED_IMPORTS_ANCHOR = (SensitivityTier, DualUseMarker)


# -----------------------------------------------------------------
# Anyio backend selection (matches the other palisade tests)
# -----------------------------------------------------------------


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
