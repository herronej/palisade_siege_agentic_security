"""E1a: the prompt-only monitor baseline.

The baseline is only meaningful if it is what it claims to be: a model reading
the policy, seeing no label, with nothing of PALISADE's in the loop. These
tests pin each of those properties with stub models, so no served call is
spent finding a wiring mistake.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from siege import SessionRunner, load_instances
from siege.ablation_matrix import (
    MONITOR_GATE_IDS,
    PROMPT_MONITOR_CONFIG,
    AblationConfig,
    GateLayer,
    MonitorGateStack,
    build_gate_stack,
)
from siege.paths import CORPUS_DIR
from siege.redteam.baselines.prompt_monitor import (
    MonitorVerdict,
    PromptOnlyMonitorGate,
    render_action,
)
from siege.scorer import score_trace


@dataclass
class _Result:
    output: MonitorVerdict


@dataclass
class _StubMonitor:
    """Answers a fixed decision and records every prompt it is shown."""

    decision: str = "allow"
    fail: bool = False
    prompts: list[str] = field(default_factory=list)

    async def run(self, prompt: str) -> _Result:
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("endpoint down")
        return _Result(MonitorVerdict(decision=self.decision, reason="stub"))


def _agents(stub: _StubMonitor) -> dict[str, _StubMonitor]:
    return {gid: stub for gid in MONITOR_GATE_IDS}


@pytest.fixture(scope="module")
def corpus():
    return {i.instance_id: i for i in load_instances(CORPUS_DIR)}


def _first(corpus, template: str):
    return next(i for i in corpus.values() if i.template == template)


def _run(instance, stub: _StubMonitor):
    runner = SessionRunner(quarantine_agents=_agents(stub), bound="production")
    return asyncio.run(runner.run(instance, PROMPT_MONITOR_CONFIG))


def _n_actions(instance) -> int:
    return sum(len(t.actions) for s in instance.sessions for t in s.turns)


# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------


def test_config_runs_nothing_of_palisade() -> None:
    cfg = PROMPT_MONITOR_CONFIG
    assert cfg.prompt_monitor
    assert cfg.layers == frozenset()
    assert not cfg.capability_bound_active
    assert not cfg.trust_scorer_active
    assert not cfg.quarantine_active, "PALISADE's slow tier and judge stay off"
    assert not cfg.judge_active


def test_stack_answers_every_gate_id_with_the_monitor() -> None:
    stack = build_gate_stack(PROMPT_MONITOR_CONFIG)
    assert isinstance(stack, MonitorGateStack)
    assert stack.monitor_only
    gates = {stack.gate_for(g) for g in ("G1", "G2", "G3", "G4", "G5", "G6", None)}
    assert len(gates) == 1
    assert isinstance(gates.pop(), PromptOnlyMonitorGate)


def test_mixed_config_is_rejected() -> None:
    bad = AblationConfig(
        name="monitor + capability bound",
        layers=frozenset({GateLayer.G4}),
        capability_bound_active=True,
        prompt_monitor=True,
    )
    with pytest.raises(ValueError, match="must run no"):
        build_gate_stack(bad)


# -----------------------------------------------------------------
# What the monitor sees
# -----------------------------------------------------------------


def test_render_hides_the_grounding_oracle_and_embeddings() -> None:
    text = render_action(
        {
            "kb_slug": "molten-salt-papers",
            "query": "viscosity of FLiNaK",
            "claim": {"type": "citation", "cited_id": "doi:10.1/x",
                      "resolved_source": {"doi": "doi:10.1/x"}},
            "retrieved_embeddings": [[0.987654, 0.123456]],
        }
    )
    assert "resolved_source" not in text
    assert "retrieved_embeddings" not in text
    assert "0.987654" not in text
    assert "doi:10.1/x" in text, "the cited identifier is content and stays"


def test_every_action_reaches_the_monitor(corpus) -> None:
    inst = _first(corpus, "xc_1_cross_boundary_chain")
    stub = _StubMonitor()
    _run(inst, stub)
    assert len(stub.prompts) == _n_actions(inst)


def test_no_label_reaches_the_monitor(corpus) -> None:
    """Across every sink-capable class: no value id, no source label, no taint
    field appears in anything the model is shown."""
    stub = _StubMonitor()
    leaks: list[str] = []
    for template in (
        "b1_9_attached_content", "b3_9_tool_return_injection",
        "b4_9_corpus_injected_code", "b5_11_injected_submission_field",
        "xc_1_cross_boundary_chain", "xc_4_taint_laundering",
        "xc_5_history_forgery",
    ):
        inst = _first(corpus, template)
        before = len(stub.prompts)
        _run(inst, stub)
        shown = "\n".join(stub.prompts[before:])
        for s in inst.sessions:
            for t in s.turns:
                for a in t.actions:
                    cap = a.capability
                    if cap is None:
                        continue
                    for needle in (cap.value_id, f"'{cap.source}'", '"taint"'):
                        if needle and needle in shown:
                            leaks.append(f"{template}: {needle}")
    assert leaks == []


def test_history_carries_earlier_actions_and_their_outcomes(corpus) -> None:
    inst = _first(corpus, "xc_1_cross_boundary_chain")
    stub = _StubMonitor(decision="deny")
    _run(inst, stub)
    assert "No earlier actions in this session." in stub.prompts[0]
    last = stub.prompts[-1]
    assert last.count("(denied, SEV2: stub)") == len(stub.prompts) - 1


def test_denial_carries_the_policy_severity(corpus) -> None:
    """A SEV1 the monitor assigns is recorded as SEV1 and shown to later
    actions, since the policy ends the session on it."""

    @dataclass
    class _Sev1(_StubMonitor):
        async def run(self, prompt: str) -> _Result:
            self.prompts.append(prompt)
            return _Result(MonitorVerdict(decision="deny", severity="SEV1", reason="x"))

    inst = _first(corpus, "xc_1_cross_boundary_chain")
    stub = _Sev1()
    trace = _run(inst, stub)
    assert all(r.incident_level == 1 for r in trace.actions)
    assert "(denied, SEV1: x)" in stub.prompts[-1]


def test_monitor_prompt_carries_the_benchmark_configuration() -> None:
    """The gates are built with the benchmark's allocation policy, not the
    operator file the policy text was written from; the monitor must be told
    the same values or it is scored against a configuration it never saw."""
    from siege.ablation_matrix import _EVAL_G5_ALLOCATION_POLICY
    from siege.redteam.baselines.prompt_monitor import (
        load_policy,
        monitor_system_prompt,
    )

    prompt = monitor_system_prompt()
    assert load_policy() in prompt, "the frozen policy is included unedited"
    for account in _EVAL_G5_ALLOCATION_POLICY.allocations:
        assert f"**{account}**" in prompt
    assert "172,800 seconds" in prompt


# -----------------------------------------------------------------
# Decisions, scoring and failure
# -----------------------------------------------------------------


def test_allow_everything_is_the_undefended_outcome(corpus) -> None:
    """With the model allowing everything, nothing else may deny: in particular
    not the capability bound, which runs before any gate when it is on."""
    inst = _first(corpus, "xc_5_history_forgery")
    trace = _run(inst, _StubMonitor(decision="allow"))
    assert all(r.allowed for r in trace.actions)
    score = score_trace(trace, inst)
    assert score.attack_success and score.hard_win


def test_deny_everything_leaves_no_win(corpus) -> None:
    inst = _first(corpus, "xc_1_cross_boundary_chain")
    trace = _run(inst, _StubMonitor(decision="deny"))
    assert not any(r.allowed for r in trace.actions)
    score = score_trace(trace, inst)
    assert not score.attack_success and not score.hard_win


def test_monitor_error_is_scored_fail_closed(corpus) -> None:
    inst = _first(corpus, "xc_1_cross_boundary_chain")
    stub = _StubMonitor(fail=True)
    trace = _run(inst, stub)
    assert not any(r.allowed for r in trace.actions)
    assert all("fail-closed" in (r.reason or "") for r in trace.actions)
    # The failure is still recorded in what later actions are shown.
    assert "denied: monitor error" in stub.prompts[-1]


def test_egress_follows_the_monitor_not_our_grounding_predicate(corpus) -> None:
    """A forged citation reaches the answer if the monitor admits the retrieval
    that asserts it; PALISADE's deterministic grounding check is not applied."""
    inst = _first(corpus, "b3_4_citation_forgery")
    admitted = _run(inst, _StubMonitor(decision="allow"))
    egress = [r for r in admitted.actions if r.kind == "response"]
    assert egress and all(r.allowed for r in egress)
    assert score_trace(admitted, inst).hard_win

    denied = _run(inst, _StubMonitor(decision="deny"))
    egress = [r for r in denied.actions if r.kind == "response"]
    assert egress and not any(r.allowed for r in egress)
