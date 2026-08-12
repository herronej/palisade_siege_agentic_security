"""
WI17 live-agent harness adapter tests (no network: mocked agent + mocked Q-LLM).
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.anyio

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from siege.ablation_matrix import CUMULATIVE_CONFIGS
from siege.instance_loader import bundled_instances_dir, load_instance
from siege.live_session_runner import (
    AgentTurn,
    LiveSessionRunner,
    ScriptedAgentDriver,
    build_live_instance,
)
from siege.scorer import score_trace
from siege.session_runner import SessionRunner
from siege.templates import b1_2_credentialing
from siege.trace_recorder import Trace
from palisade.quarantine import build_intent_extraction_agent

_FULL = CUMULATIVE_CONFIGS[-1]


def _b11():
    return load_instance(bundled_instances_dir() / "b1_1_direct_injection_example.yaml")


def _flagging_g1_agent():
    """A Q-LLM intent-extraction agent that flags a high-confidence dual-use
    intent -> G1 slow tier denies (mirrors test_g1_slow)."""

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={
                        "intent_summary": "request for a restricted dual-use procedure",
                        "dual_use_flag": "nuclear",
                        "confidence": 0.95,
                        "reasoning": "credential-laundered withheld procedure",
                    },
                    tool_call_id="c1",
                )
            ]
        )

    return build_intent_extraction_agent(FunctionModel(fn))


# ---------------------------------------------------------------- round-trip
async def test_b1_round_trips_through_live_agent_to_scorer():
    inst = _b11()
    driver = ScriptedAgentDriver(
        default=AgentTurn(
            tool_calls=(("create_file", {"path": "x.py", "content": "print(1)"}),),
            response_text="done",
        )
    )
    trace = await LiveSessionRunner(agent_driver=driver).run(inst, _FULL)
    assert isinstance(trace, Trace) and trace.actions
    score = score_trace(trace, inst)
    # The authored injection is still blocked by G1; the round-trip produced a
    # scorable trace.
    assert score.attack_success is False


async def test_live_instance_appends_agent_actions_and_response():
    inst = _b11()
    driver = ScriptedAgentDriver(
        default=AgentTurn(
            tool_calls=(("submit_hpc_job", {"slurm_script": "#!/bin/bash\n"}),),
            response_text="ok",
        )
    )
    live = await build_live_instance(inst, driver)
    kinds = [a.kind.value for a in live.sessions[0].turns[0].actions]
    assert "tool_call" in kinds and "response" in kinds


# ---------------------------------------------------------------- Q-LLM fires
# The FunctionModel Q-LLM mock requires a running asyncio loop (it errors under
# trio); pin to asyncio like the slow-tier unit tests. Under fail-open this also
# matters: the deny must come from the *confident* dual-use verdict (delivered
# only when the mock actually runs), not from an exception-path default-deny.
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_qllm_slow_tier_lowers_asr_in_live_mode(anyio_backend):
    cred = b1_2_credentialing.build()[0]
    driver = ScriptedAgentDriver()  # benign refusal

    # Authored / offline, fast tier only: credentialing evades the regex.
    offline = await SessionRunner().run(cred, _FULL)
    assert score_trace(offline, cred).attack_success is True

    # Live, with the Q-LLM slow tier wired on G1: the intent flag denies.
    live = await LiveSessionRunner(
        agent_driver=driver, quarantine_agents={"G1": _flagging_g1_agent()}
    ).run(cred, _FULL)
    assert score_trace(live, cred).attack_success is False


# ---------------------------------------------------------------- gating + audit
async def test_agent_tool_call_is_gated_and_recorded():
    inst = _b11()
    driver = ScriptedAgentDriver(
        default=AgentTurn(
            tool_calls=(("create_file", {"path": "p.py", "content": "x=1"}),)
        )
    )
    trace = await LiveSessionRunner(agent_driver=driver).run(inst, _FULL)
    agent_calls = [a for a in trace.actions if a.label.startswith("agent:")]
    assert agent_calls and agent_calls[0].gate == "G4"


async def test_live_path_admits_no_false_hard_win():
    from siege.redteam.reward import WinKind, discriminate

    inst = _b11()
    driver = ScriptedAgentDriver(default=AgentTurn(response_text="no"))
    trace = await LiveSessionRunner(agent_driver=driver).run(inst, _FULL)
    # No false hard-win enters via the live path: the capability-model
    # discriminator finds no tainted value reaching a sink on this benign run.
    # (The standalone bounds.py hard-win audit guard is WI16 and out of scope
    # on this branch; the kept 13a discriminator is the check.)
    assert discriminate(trace) is not WinKind.HARD


# ---------------------------------------------------------------- cost guard
async def test_max_instances_cost_guard():
    from siege.eval.siege_runner import run_siege_evaluation

    result = await run_siege_evaluation(max_instances=2, configs=(_FULL,))
    assert result.n_instances == 2


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_parity_report_diverges_with_qllm(anyio_backend):
    from pathlib import Path

    from siege.eval.siege_runner import run_parity

    corpus = Path(__file__).resolve().parents[1] / "corpus" / "b1_2_credentialing"
    report = await run_parity(
        instances_dir=corpus,
        configs=(_FULL,),
        agent_driver=ScriptedAgentDriver(),
        quarantine_agents={"G1": _flagging_g1_agent()},
    )
    assert report.cells
    # The Q-LLM slow tier moves ASR down on the credentialing class.
    assert report.divergent()
    assert "authored ASR" in report.to_markdown()
