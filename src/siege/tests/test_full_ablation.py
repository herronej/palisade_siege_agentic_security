"""
Full SIEGE ablation driver (CI guard).

Runs the offline (``TestModel``) full ablation over a small subset and asserts
the merge produces all nine configs and that the Q-LLM is held fixed across the
``+Q-LLM`` and ``+both`` columns (the cache is replayed, not re-sampled).
"""

from __future__ import annotations

import asyncio

from pydantic_ai.models.test import TestModel

from siege.ablation_matrix import AUGMENTED_CONFIGS
from siege.full_ablation import _CachingAgent, run_full_ablation

_EXPECTED_CONFIGS = [
    "baseline",
    "+G4",
    "+G4+G3",
    "+G4+G3+G1",
    "+G4+G3+G1+G5",
    "full PALISADE",
    "full +semgrep",
    "full +slow",
    "full +all",
]


def test_augmented_configs_include_full_both():
    by_name = {c.name: c for c in AUGMENTED_CONFIGS}
    assert {"full +semgrep", "full +slow", "full +all"} <= set(by_name)
    both = by_name["full +all"]
    assert both.semgrep_active and both.quarantine_active


def test_caching_agent_memoizes_by_prompt():
    calls = {"n": 0}

    class _Inner:
        async def run(self, prompt, **kw):
            calls["n"] += 1
            return f"out:{prompt}"

    stats = {"model_calls": 0, "cache_hits": 0, "errors": 0}
    agent = _CachingAgent(_Inner(), stats)
    first = asyncio.run(agent.run("x"))
    second = asyncio.run(agent.run("x"))  # same prompt -> cache hit, no re-call
    assert first == second == "out:x"
    assert calls["n"] == 1
    assert stats == {"model_calls": 1, "cache_hits": 1, "errors": 0}


def test_full_ablation_merges_nine_configs_and_holds_qllm_fixed():
    result, stats = asyncio.run(run_full_ablation(TestModel(), max_instances=24))
    assert [c.name for c in result.configs] == _EXPECTED_CONFIGS
    # every config produced cells in the merged result
    assert set(_EXPECTED_CONFIGS) <= {c.config_name for c in result.cells}
    # the +both pass replayed the +Q-LLM cache rather than re-calling the model
    assert stats["model_calls"] > 0, "Q-LLM never fired -- wiring broken"
    assert stats["cache_hits"] > 0, "+both did not replay the +Q-LLM cache"
    assert stats["errors"] == 0, "TestModel calls should not error"


def test_full_ablation_multi_seed_pools_qllm_columns():
    # k=3 runs the Q-LLM columns over 3 independent draws and pools them: the
    # +Q-LLM / +both cells carry 3x the per-template attack instances (tighter
    # CI under a live model), while the deterministic baseline is single-pass.
    result, _ = asyncio.run(
        run_full_ablation(TestModel(), max_instances=24, qllm_samples=3)
    )
    assert result.qllm_samples == 3
    base = {c.template: c.n_attack for c in result.cells if c.config_name == "baseline"}
    qllm = {c.template: c.n_attack for c in result.cells if c.config_name == "full +slow"}
    both = {c.template: c.n_attack for c in result.cells if c.config_name == "full +all"}
    assert any(v > 0 for v in qllm.values()), "no attack instances in the subset"
    for template, n in qllm.items():
        if base.get(template):
            assert n == 3 * base[template], f"{template}: {n} != 3x{base[template]}"
            assert both[template] == 3 * base[template]
