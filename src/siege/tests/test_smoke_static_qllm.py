"""
Static SIEGE Q-LLM smoketest (CI guard).

Runs the offline (``TestModel``) variant of the slow-tier smoketest over a
small subset and asserts the slow tier actually fired -- regression-guarding
the per-gate Q-LLM wiring (incl. the G4/G5 code-intent injection in
``build_gate_stack``).
"""

from __future__ import annotations

import asyncio

from pydantic_ai.models.test import TestModel

from siege.smoke_static_qllm import build_qllm_agents, run_smoke


def test_static_smoke_wires_all_five_slow_tier_agents():
    counter: list[int] = [0]
    agents = build_qllm_agents(TestModel(), counter)
    assert set(agents) == {"G1", "G2", "G3", "G4", "G5"}


def test_static_smoke_qllm_slow_tier_fires():
    # Offline TestModel: the slow-tier code path runs without a served model,
    # and (unlike a real unreachable model) every call succeeds.
    result, attempted, succeeded = asyncio.run(run_smoke(TestModel(), max_instances=24))
    assert len(result.cells) > 0, "no cells produced"
    assert attempted > 0, "Q-LLM slow tier never fired -- wiring is broken"
    assert succeeded == attempted, "TestModel calls should all succeed"


def test_build_gate_stack_wires_g4_g5_code_intent():
    # Directly pin the build_gate_stack extension: G4/G5 receive the
    # code-intent agent (the corpus prefix in the smoke only reaches B1).
    from siege.ablation_matrix import CUMULATIVE_CONFIGS, build_gate_stack

    full = CUMULATIVE_CONFIGS[-1]
    sentinel_g4, sentinel_g5 = object(), object()
    stack = build_gate_stack(
        full, quarantine_agents={"G4": sentinel_g4, "G5": sentinel_g5}
    )
    assert stack.gate_for("G4").code_intent_extraction_agent is sentinel_g4
    assert stack.gate_for("G5").code_intent_extraction_agent is sentinel_g5
    # Default (no agents) leaves the slow tier off -- backward compatible.
    bare = build_gate_stack(full)
    assert bare.gate_for("G4").code_intent_extraction_agent is None
    assert bare.gate_for("G5").code_intent_extraction_agent is None
