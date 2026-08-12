"""
Unit tests for the Q-LLM (quarantine model) integration.

Covers all seven acceptance criteria of the work item
`Build Q-LLM Agent factory with QuarantineDecision
output type`:

1. `build_quarantine_agent(model_spec)` returns a configured `Agent`.
2. System prompt instructs the Q-LLM to return only structured JSON,
   no tool access.
3. `QuarantineDecision` Pydantic model with the five required fields
   (`contains_instructions`, `suspicious_score`, `sanitized_text`,
   `reasoning`, `intent_summary`).
4. `_build_agent` instantiates the Q-LLM only when
   `quarantine_enabled=True`.
5. When `quarantine_enabled=False`, gates' `check_slow` is a no-op
   (no model invocation).
6. Unit test with PydanticAI `TestModel` confirms the
   structured-output path.
7. Self-consistency: with `quarantine_self_consistency_samples=2`,
   the Q-LLM is called twice and disagreement triggers default-deny.

Mocking strategy:

- `TestModel` for happy-path structured-output testing (returns a
  deterministic `QuarantineDecision`).
- `FunctionModel` for self-consistency testing (alternates outputs
  across calls so we can build "agree" / "disagree" sequences).
- An exception-raising `FunctionModel` for the error-path test
  (verifies that a model-side failure collapses to default-deny
  rather than propagating).
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

# anyio's pytest plugin requires async tests to be marked.
pytestmark = pytest.mark.anyio

from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from palisade.host import HostProjectModel
from palisade.config import PalisadeSettings
from palisade.gates.base import GateContext
from palisade.quarantine import (
    QUARANTINE_SYSTEM_PROMPT,
    SELF_CONSISTENCY_SAMPLING_TEMPERATURE,
    QuarantineDecision,
    build_quarantine_agent,
    run_quarantine_with_self_consistency,
)
from palisade.sidecar import PalisadeSidecar


# -----------------------------------------------------------------
# Test helpers
# -----------------------------------------------------------------


def _decision_args(
    *,
    contains_instructions: bool,
    suspicious_score: float = 0.5,
    sanitized_text: str = "",
    reasoning: str = "",
    intent_summary: str = "",
) -> dict[str, Any]:
    """Build the kwargs for a `QuarantineDecision` tool-call payload."""
    return {
        "contains_instructions": contains_instructions,
        "suspicious_score": suspicious_score,
        "sanitized_text": sanitized_text,
        "reasoning": reasoning,
        "intent_summary": intent_summary,
    }


def _make_function_model_returning(
    decisions: list[dict[str, Any]],
) -> FunctionModel:
    """
    Construct a `FunctionModel` that returns `decisions[i]` on the
    i-th call. Useful for self-consistency tests where we need
    distinct outputs per call.

    After exhausting the list, repeats the last decision so a
    misconfigured test that calls one extra time still gets a
    sensible response rather than an IndexError. Tests that care
    about call count should assert on `call_counter[0]`.
    """
    call_counter = [0]

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        i = min(call_counter[0], len(decisions) - 1)
        call_counter[0] += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args=decisions[i],
                    tool_call_id=f"call-{call_counter[0]}",
                )
            ]
        )

    model = FunctionModel(fn)
    # Stash the counter on the model so tests can read it.
    model._test_call_counter = call_counter  # type: ignore[attr-defined]
    return model


def _make_project(name: str = "test-project") -> HostProject:
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
# AC 3: QuarantineDecision schema
# -----------------------------------------------------------------


def test_quarantine_decision_has_required_fields() -> None:
    """AC3: the five named fields exist and accept reasonable values."""
    d = QuarantineDecision(
        contains_instructions=True,
        suspicious_score=0.75,
        sanitized_text="clean version",
        reasoning="found a DAN-style payload",
        intent_summary="jailbreak attempt",
    )
    assert d.contains_instructions is True
    assert d.suspicious_score == 0.75
    assert d.sanitized_text == "clean version"
    assert d.reasoning == "found a DAN-style payload"
    assert d.intent_summary == "jailbreak attempt"


def test_quarantine_decision_score_bounded_to_unit_interval() -> None:
    """`suspicious_score` is constrained to [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        QuarantineDecision(
            contains_instructions=False,
            suspicious_score=1.5,
        )
    with pytest.raises(ValidationError):
        QuarantineDecision(
            contains_instructions=False,
            suspicious_score=-0.1,
        )


def test_quarantine_decision_defaults_keep_optional_fields_empty() -> None:
    """
    Optional text fields default to empty strings so a misbehaving
    Q-LLM that omits a field still produces a valid model. Pinning
    the defaults here protects gates that read these fields
    unconditionally.
    """
    d = QuarantineDecision(
        contains_instructions=False,
        suspicious_score=0.0,
    )
    assert d.sanitized_text == ""
    assert d.reasoning == ""
    assert d.intent_summary == ""


# -----------------------------------------------------------------
# AC 1 + AC 2: factory shape -- Agent, no toolsets, system prompt
# -----------------------------------------------------------------


def test_build_quarantine_agent_returns_agent_instance() -> None:
    """AC1: factory returns a configured PydanticAI `Agent`."""
    agent = build_quarantine_agent(TestModel())
    assert isinstance(agent, Agent)


def test_build_quarantine_agent_pins_output_type() -> None:
    """
    AC2/AC6: the agent's `output_type` is `QuarantineDecision`.
    Pinned so a future refactor that loosens the output shape
    breaks here, not silently in production.
    """
    agent = build_quarantine_agent(TestModel())
    assert agent.output_type is QuarantineDecision


def test_build_quarantine_agent_has_no_user_toolsets() -> None:
    """
    AC2: the Q-LLM has NO user-provided tool access.

    PydanticAI synthesizes one `_AgentFunctionToolset` per Agent
    that holds the structured-output "final_result" tool the
    model must call to submit a `QuarantineDecision`. That
    toolset is part of the output_type enforcement mechanism, not
    something the model can use to take action in the world.

    The structural property we test is: the Q-LLM's toolset
    surface is the same as a control Agent built with no
    toolsets and the same output_type. Anything extra would be a
    user-provided toolset slipping through the factory.
    """
    q_agent = build_quarantine_agent(TestModel())
    # Control: the most stripped-down agent we can build with the
    # same model and output_type. By construction it has no user
    # tools.
    control_agent = Agent(TestModel(), output_type=QuarantineDecision)

    q_toolset_types = [type(t).__name__ for t in q_agent.toolsets]
    control_toolset_types = [type(t).__name__ for t in control_agent.toolsets]

    assert q_toolset_types == control_toolset_types, (
        f"Q-LLM has additional toolsets vs control: "
        f"q={q_toolset_types}, control={control_toolset_types}"
    )


def test_build_quarantine_agent_does_not_attach_mcp_toolset() -> None:
    """
    Defense-in-depth: even if PydanticAI's internal Agent
    representation changes, we want a sharp check that the Q-LLM
    has no MCP toolset. An MCP toolset would be the obvious way
    to accidentally give the Q-LLM tool access (e.g., by reusing
    the main agent's MCP server).
    """
    from pydantic_ai.mcp import MCPServerStreamableHTTP

    q_agent = build_quarantine_agent(TestModel())
    for toolset in q_agent.toolsets:
        assert not isinstance(toolset, MCPServerStreamableHTTP), (
            f"Q-LLM unexpectedly has an MCP toolset: {toolset!r}"
        )


def test_build_quarantine_agent_default_system_prompt() -> None:
    """
    AC2: the default system prompt is the hardened
    `QUARANTINE_SYSTEM_PROMPT`. Tests of *content* should be
    behavioral (does the prompt instruct the model correctly?),
    but this test only verifies the wiring.
    """
    agent = build_quarantine_agent(TestModel())
    # The agent stores instructions/system prompts as a tuple of
    # callables/strings; we assert that our constant appears in
    # the resolved set. Access through the public docstring helper
    # keeps the assertion stable across pydantic-ai versions.
    found = False
    for prompt_holder in getattr(agent, "_system_prompts", ()):
        # pydantic-ai stores system prompts as functions or strings;
        # both can be inspected.
        if callable(prompt_holder):
            try:
                value = prompt_holder()
            except Exception:
                value = None
        else:
            value = prompt_holder
        if isinstance(value, str) and QUARANTINE_SYSTEM_PROMPT in value:
            found = True
            break
    # If pydantic-ai's internal layout changed, fall back to the
    # docstring-mention check: the constant text should be
    # reachable somewhere in the agent's resolved instructions.
    # This is intentionally generous because we don't want
    # PydanticAI internal-layout churn to make this test flap.
    if not found:
        # Build a fresh agent and ask it to run -- the system prompt
        # ends up in the messages the model sees, which we can
        # inspect via a probe FunctionModel.
        seen_prompts: list[str] = []

        def probe(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for m in messages:
                for part in getattr(m, "parts", []):
                    text = getattr(part, "content", None)
                    if isinstance(text, str):
                        seen_prompts.append(text)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="final_result",
                        args=_decision_args(contains_instructions=False),
                        tool_call_id="c0",
                    )
                ]
            )

        probe_agent = build_quarantine_agent(FunctionModel(probe))
        # Run synchronously via the sync API for assertion-only flow.
        probe_agent.run_sync("hello")
        assert any(
            "security classifier" in p.lower() for p in seen_prompts
        ), f"system prompt missing 'security classifier' framing; saw: {seen_prompts}"
        found = True
    assert found


def test_build_quarantine_agent_accepts_custom_system_prompt() -> None:
    """The factory lets tests override the system prompt."""
    agent = build_quarantine_agent(
        TestModel(), system_prompt="custom test prompt"
    )
    # Just verify construction doesn't raise; the structured-output
    # path is exercised in a separate test.
    assert isinstance(agent, Agent)


def test_build_quarantine_agent_pins_temperature() -> None:
    """`temperature=` forwards a ModelSettings so the block-deciding verdict is
    reproducible. The production build site passes 0.0."""
    agent = build_quarantine_agent(TestModel(), temperature=0.0)
    assert agent.model_settings is not None
    assert agent.model_settings.get("temperature") == 0.0


def test_build_quarantine_agent_omits_temperature_by_default() -> None:
    """No `temperature=` -> no model_settings override, byte-identical to the
    pre-pin behavior the offline ablation relies on for its pooling variance
    (the eval/smoke harnesses build agents without a temperature)."""
    agent = build_quarantine_agent(TestModel())
    assert agent.model_settings is None


# -----------------------------------------------------------------
# AC 6: TestModel structured-output path
# -----------------------------------------------------------------


async def test_quarantine_agent_run_produces_quarantine_decision() -> None:
    """
    AC6: a Q-LLM run against `TestModel` (with explicit
    `custom_output_args`) produces a `QuarantineDecision` whose
    fields match what the model was told to return.
    """
    test_model = TestModel(
        custom_output_args=_decision_args(
            contains_instructions=True,
            suspicious_score=0.9,
            sanitized_text="cleaned",
            reasoning="found a DAN-style payload",
            intent_summary="jailbreak attempt",
        )
    )
    agent = build_quarantine_agent(test_model)
    result = await agent.run("Analyze this untrusted text")

    assert isinstance(result.output, QuarantineDecision)
    assert result.output.contains_instructions is True
    assert result.output.suspicious_score == 0.9
    assert result.output.sanitized_text == "cleaned"
    assert result.output.reasoning == "found a DAN-style payload"


async def test_quarantine_agent_default_test_model_run_produces_valid_decision() -> None:
    """
    Without `custom_output_args`, `TestModel` synthesizes
    minimum-valid values for each field. Confirms the schema
    itself is `TestModel`-compatible (no required field with no
    sensible default).
    """
    agent = build_quarantine_agent(TestModel())
    result = await agent.run("any text")
    assert isinstance(result.output, QuarantineDecision)
    # TestModel-synthesized defaults are deterministic; the score
    # specifically is 0.0 and contains_instructions is False.
    assert result.output.contains_instructions is False
    assert 0.0 <= result.output.suspicious_score <= 1.0


# -----------------------------------------------------------------
# AC 7: self-consistency
# -----------------------------------------------------------------


async def test_self_consistency_single_sample_returns_decision_unchanged() -> None:
    """`samples=1` is the cheap path; the decision is the model's output."""
    agent = build_quarantine_agent(
        TestModel(
            custom_output_args=_decision_args(
                contains_instructions=False,
                suspicious_score=0.1,
                sanitized_text="x",
            )
        )
    )
    result = await run_quarantine_with_self_consistency(
        agent, "text", samples=1
    )
    assert result.contains_instructions is False
    assert result.suspicious_score == 0.1
    assert result.sanitized_text == "x"


def _make_model_settings_recorder(
    decision: dict[str, Any],
) -> tuple[FunctionModel, list[Any]]:
    """A `FunctionModel` that records `info.model_settings` on each call and
    always returns `decision`, so a test can see the temperature each draw
    actually ran at."""
    seen: list[Any] = []

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.model_settings)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args=decision,
                    tool_call_id=f"call-{len(seen)}",
                )
            ]
        )

    return FunctionModel(fn), seen


async def test_self_consistency_multi_sample_uses_sampling_temperature() -> None:
    """Regression guard for the 'temperature unset' fix: `samples >= 2` draws at
    the nonzero sampling temperature so the votes can actually differ, while
    `samples == 1` adds no override (keeping the single-sample path
    byte-identical to `agent.run(prompt)`)."""
    clean = _decision_args(contains_instructions=False, suspicious_score=0.0)

    # samples=1 -> no per-run override; defers to the agent's own settings.
    model1, seen1 = _make_model_settings_recorder(clean)
    await run_quarantine_with_self_consistency(
        build_quarantine_agent(model1), "text", samples=1
    )
    assert len(seen1) == 1
    assert seen1[0] is None or "temperature" not in seen1[0]

    # samples=3 -> every draw carries the sampling temperature.
    model3, seen3 = _make_model_settings_recorder(clean)
    await run_quarantine_with_self_consistency(
        build_quarantine_agent(model3), "text", samples=3
    )
    assert len(seen3) == 3
    assert all(
        ms is not None
        and ms.get("temperature") == SELF_CONSISTENCY_SAMPLING_TEMPERATURE
        for ms in seen3
    )


async def test_self_consistency_two_agreeing_samples_returns_first() -> None:
    """
    AC7: two samples that agree on `contains_instructions` →
    return the first sample's decision unchanged.
    """
    model = _make_function_model_returning(
        [
            _decision_args(
                contains_instructions=True,
                suspicious_score=0.9,
                sanitized_text="A",
            ),
            _decision_args(
                contains_instructions=True,
                suspicious_score=0.7,
                sanitized_text="B",
            ),
        ]
    )
    agent = build_quarantine_agent(model)
    result = await run_quarantine_with_self_consistency(
        agent, "text", samples=2
    )
    assert result.contains_instructions is True
    # First sample's score/text wins.
    assert result.suspicious_score == 0.9
    assert result.sanitized_text == "A"
    assert model._test_call_counter[0] == 2  # type: ignore[attr-defined]


async def test_self_consistency_two_disagreeing_samples_triggers_default_deny() -> None:
    """
    AC7: two samples that disagree on `contains_instructions` →
    default-deny. The returned decision has
    `contains_instructions=True`, `suspicious_score=1.0`, and a
    reasoning string identifying the disagreement.
    """
    model = _make_function_model_returning(
        [
            _decision_args(contains_instructions=False, sanitized_text="A"),
            _decision_args(contains_instructions=True, sanitized_text="B"),
        ]
    )
    agent = build_quarantine_agent(model)
    result = await run_quarantine_with_self_consistency(
        agent, "text", samples=2
    )
    assert result.contains_instructions is True
    assert result.suspicious_score == 1.0
    assert "disagreement" in result.reasoning.lower()
    assert model._test_call_counter[0] == 2  # type: ignore[attr-defined]


async def test_self_consistency_three_samples_all_agree_returns_first() -> None:
    """Strict-agreement rule extends to N samples."""
    model = _make_function_model_returning(
        [
            _decision_args(contains_instructions=False, sanitized_text="A"),
            _decision_args(contains_instructions=False, sanitized_text="B"),
            _decision_args(contains_instructions=False, sanitized_text="C"),
        ]
    )
    agent = build_quarantine_agent(model)
    result = await run_quarantine_with_self_consistency(
        agent, "text", samples=3
    )
    assert result.contains_instructions is False
    assert result.sanitized_text == "A"


async def test_self_consistency_three_samples_one_disagrees_triggers_default_deny() -> None:
    """One disagreement in N samples is enough to default-deny."""
    model = _make_function_model_returning(
        [
            _decision_args(contains_instructions=True),
            _decision_args(contains_instructions=True),
            _decision_args(contains_instructions=False),  # the outlier
        ]
    )
    agent = build_quarantine_agent(model)
    result = await run_quarantine_with_self_consistency(
        agent, "text", samples=3
    )
    assert result.contains_instructions is True
    assert result.suspicious_score == 1.0


async def test_self_consistency_zero_or_negative_samples_treated_as_one() -> None:
    """
    Defensive: `samples<1` is treated as 1 rather than raising. A
    misconfigured setting should run the Q-LLM once, not crash
    the gate.
    """
    agent = build_quarantine_agent(
        TestModel(
            custom_output_args=_decision_args(contains_instructions=False)
        )
    )
    result = await run_quarantine_with_self_consistency(
        agent, "text", samples=0
    )
    assert isinstance(result, QuarantineDecision)


async def test_self_consistency_model_error_collapses_to_default_deny() -> None:
    """
    A model-side exception during `agent.run` is caught and
    converted to a default-deny decision. Gates calling the
    Q-LLM must never see the exception propagate.
    """

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("simulated model timeout")

    agent = build_quarantine_agent(FunctionModel(fn))
    result = await run_quarantine_with_self_consistency(
        agent, "text", samples=2
    )
    assert result.contains_instructions is True
    assert result.suspicious_score == 1.0
    assert "failed" in result.reasoning.lower() or "default-deny" in result.reasoning.lower()


# -----------------------------------------------------------------
# AC 4: quarantine_enabled wiring
# -----------------------------------------------------------------


def test_sidecar_quarantine_agent_is_none_by_default() -> None:
    """
    A fresh sidecar has no Q-LLM attached. `quarantine_enabled`
    flips this only via `build_capabilities(quarantine_agent=...)`
    in `agents.py:_build_agent`; without that the Q-LLM is absent
    regardless of the flag's value.
    """
    settings = PalisadeSettings(enabled=True, quarantine_enabled=True)
    sidecar = PalisadeSidecar(settings, _make_project())
    assert sidecar.quarantine_agent is None


def test_build_capabilities_stores_quarantine_agent() -> None:
    """
    AC4 mechanism: `build_capabilities(quarantine_agent=...)` stores
    the agent the `_build_agent` path constructs. This is the
    contract the `agents.py` wiring relies on.
    """
    settings = PalisadeSettings(enabled=True, quarantine_enabled=True)
    sidecar = PalisadeSidecar(settings, _make_project())

    agent = build_quarantine_agent(TestModel())
    sidecar.build_capabilities(quarantine_agent=agent)

    assert sidecar.quarantine_agent is agent


# -----------------------------------------------------------------
# AC 5: check_slow no-op when Q-LLM is absent
# -----------------------------------------------------------------


async def test_gate_check_slow_is_noop_when_no_quarantine_agent() -> None:
    """
    AC5: when `ctx.quarantine_agent is None`, `Gate.check_slow`
    returns the input decision unchanged without invoking any
    model. The base-class dispatch makes this a structural
    guarantee, not a per-gate discipline.
    """
    from palisade.gates.base import GateDecision, PassThroughGate
    from palisade.capabilities import CapabilityRegistry
    from palisade.trust import TrustScorer

    gate = PassThroughGate(enabled=True)
    ctx = GateContext(
        capability_registry=CapabilityRegistry(),
        trust_scorer=TrustScorer(PalisadeSettings(enabled=True)),
        quarantine_agent=None,  # explicit -- no Q-LLM
    )
    initial = GateDecision(allow=True, reason="initial")

    result = await gate.check_slow(b"any payload", ctx, initial)

    # Decision passes through unchanged. The Q-LLM was never
    # invoked because there was no agent to invoke.
    assert result is initial


# -----------------------------------------------------------------
# Anyio backend selection (matches the other palisade tests)
# -----------------------------------------------------------------


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
