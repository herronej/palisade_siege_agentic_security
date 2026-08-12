"""
Unit tests for the G5 slow tier: Q-LLM job-intent extraction +
chained-job dependency DAG analysis.

Covers the acceptance criteria of
``Implement G5 slow-tier``:

- ``check_slow`` runs ``run_code_intent_extraction_with_self_consistency``
  against the resolved SLURM script.
- Low confidence -> SEV2 deny (G1 / G4 pattern).
- High-confidence weapons dual-use -> SEV1 deny + sticky-capability mark.
- Intent mismatch against ``user:prompt`` recorded intent -> SEV2 deny.
- ``walk_dependency_dag`` resolves ``--dependency=afterok:JOBID`` and
  re-applies the fast-tier checks to each dependent job.
- The SIEGE B2.6 chain (benign job -> dependency on a mining job)
  is caught by the DAG walk.

The Q-LLM is stubbed (a fake agent returning a canned
``CodeIntentExtraction``); no real model is called.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.anyio

from palisade.capabilities import CapabilityRegistry, CapabilityTag, DualUseMarker
from palisade.config import PalisadeSettings
from palisade.gates.base import GateContext, GateDecision
from palisade.gates.g5_hpc import (
    AllocationLimits,
    AllocationPolicy,
    G5HpcJobGate,
)
from palisade.gates.slurm_parser import parse_slurm_script
from palisade.quarantine import CodeIntentExtraction
from palisade.trust import TrustScorer


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# -----------------------------------------------------------------
# Fakes / helpers
# -----------------------------------------------------------------


class _FakeRun:
    def __init__(self, output: Any) -> None:
        self.output = output


class _FakeAgent:
    """Stand-in for the code-intent PydanticAI agent. Returns a fixed
    ``CodeIntentExtraction`` (or a per-call sequence)."""

    def __init__(self, *outputs: CodeIntentExtraction) -> None:
        self._outputs = list(outputs)
        self.calls: list[str] = []

    async def run(self, prompt: str) -> _FakeRun:
        self.calls.append(prompt)
        out = self._outputs[min(len(self.calls) - 1, len(self._outputs) - 1)]
        return _FakeRun(out)


def _intent(
    *,
    summary: str = "run a scientific simulation",
    categories: list[str] | None = None,
    dual_use: DualUseMarker = DualUseMarker.NONE,
    confidence: float = 0.9,
) -> CodeIntentExtraction:
    return CodeIntentExtraction(
        intent_summary=summary,
        categories=categories or ["compute"],
        dual_use_flag=dual_use,
        confidence=confidence,
    )


def _ctx(
    *,
    agent_present: bool = True,
    user_intent_summary: str = "",
    user_dual_use: str = "none",
) -> GateContext:
    reg = CapabilityRegistry()
    reg.tag(
        "user:prompt",
        CapabilityTag(
            source="user:tester",
            metadata={
                "intent_summary": user_intent_summary,
                "intent_dual_use_flag": user_dual_use,
            },
        ),
    )
    return GateContext(
        capability_registry=reg,
        trust_scorer=TrustScorer(PalisadeSettings(enabled=True)),
        # The base `Gate.check_slow` no-ops unless quarantine_agent is set;
        # a sentinel object is enough to pass that guard.
        quarantine_agent=object() if agent_present else None,
    )


def _gate(intent: CodeIntentExtraction | None = None, **overrides: Any) -> G5HpcJobGate:
    defaults: dict[str, Any] = {"enabled": True}
    if intent is not None:
        defaults["code_intent_extraction_agent"] = _FakeAgent(intent)
    defaults.update(overrides)
    return G5HpcJobGate(**defaults)


_SCRIPT = "#SBATCH --account=approved-research\nsrun python forge-tune.py\n"


async def _slow(gate: G5HpcJobGate, ctx: GateContext, *, script: str = _SCRIPT):
    baseline = GateDecision(allow=True, reason="fast ok")
    return await gate.check_slow(
        {"slurm_script": script, "user_config": {}}, ctx, baseline
    )


# -----------------------------------------------------------------
# Q-LLM job-intent extraction
# -----------------------------------------------------------------


async def test_check_slow_calls_qllm_against_resolved_script() -> None:
    agent = _FakeAgent(_intent())
    gate = G5HpcJobGate(enabled=True, code_intent_extraction_agent=agent)
    await _slow(gate, _ctx())
    assert len(agent.calls) == 1
    # The resolved SLURM script is handed to the Q-LLM.
    assert "forge-tune.py" in agent.calls[0]
    assert "SLURM" in agent.calls[0]


async def test_low_confidence_denies_sev2() -> None:
    gate = _gate(_intent(confidence=0.2))
    d = await _slow(gate, _ctx())
    assert d.allow is False
    assert d.incident_level == 2
    assert "confidence" in d.reason


async def test_high_confidence_nuclear_dual_use_denies_sev1_sticky() -> None:
    gate = _gate(_intent(summary="enrichment cascade simulation",
                         dual_use=DualUseMarker.NUCLEAR, confidence=0.95))
    d = await _slow(gate, _ctx())
    assert d.allow is False
    assert d.incident_level == 1
    assert d.capability_tag is not None
    assert d.capability_tag.metadata.get("sticky") is True
    assert d.capability_tag.metadata.get("g5_check") == "dual_use_job"


async def test_high_confidence_chem_dual_use_denies_sev1() -> None:
    gate = _gate(_intent(summary="nerve agent synthesis route search",
                         dual_use=DualUseMarker.CHEM, confidence=0.9))
    d = await _slow(gate, _ctx())
    assert d.allow is False
    assert d.incident_level == 1


async def test_dual_use_below_high_confidence_not_sev1() -> None:
    # CHEM but low-confidence-for-dual-use (0.6 < 0.8): not the SEV1 path.
    # With no matching user intent it falls to the mismatch check (SEV2).
    gate = _gate(_intent(dual_use=DualUseMarker.CHEM, confidence=0.6))
    d = await _slow(gate, _ctx(user_dual_use="none"))
    assert d.incident_level != 1


async def test_intent_mismatch_against_user_prompt_denies_sev2() -> None:
    # Code claims credential_access; the user prompt never mentioned it.
    gate = _gate(
        _intent(summary="read ssh keys", categories=["credential_access"],
                confidence=0.9)
    )
    d = await _slow(gate, _ctx(user_intent_summary="fine-tune the salt model"))
    assert d.allow is False
    assert d.incident_level == 2
    assert "mismatch" in d.reason


async def test_intent_match_allows_annotated() -> None:
    gate = _gate(_intent(summary="train model", categories=["compute"],
                         confidence=0.9))
    d = await _slow(gate, _ctx(user_intent_summary="fine-tune the salt model"))
    assert d.allow is True
    assert "G5 slow-tier ok" in d.reason


async def test_declared_credential_access_matches_and_allows() -> None:
    # The user explicitly asked for credential access -> no mismatch.
    gate = _gate(
        _intent(categories=["credential_access"], confidence=0.9)
    )
    d = await _slow(
        gate, _ctx(user_intent_summary="read the credential access log and summarize")
    )
    assert d.allow is True


# -----------------------------------------------------------------
# Pass-through guards
# -----------------------------------------------------------------


async def test_no_agent_is_passthrough() -> None:
    gate = G5HpcJobGate(enabled=True)  # no code-intent agent
    d = await _slow(gate, _ctx())
    assert d.allow is True
    assert d.reason == "fast ok"


async def test_no_quarantine_agent_skips_slow_tier() -> None:
    # Base Gate.check_slow guards on ctx.quarantine_agent.
    agent = _FakeAgent(_intent(confidence=0.1))
    gate = G5HpcJobGate(enabled=True, code_intent_extraction_agent=agent)
    d = await _slow(gate, _ctx(agent_present=False))
    assert d.allow is True
    assert len(agent.calls) == 0


async def test_fast_denied_decision_is_passthrough() -> None:
    agent = _FakeAgent(_intent(confidence=0.1))
    gate = G5HpcJobGate(enabled=True, code_intent_extraction_agent=agent)
    denied = GateDecision(allow=False, reason="fast denied", incident_level=1)
    out = await gate.check_slow(
        {"slurm_script": _SCRIPT, "user_config": {}}, _ctx(), denied
    )
    assert out is denied
    assert len(agent.calls) == 0


async def test_empty_script_is_passthrough() -> None:
    agent = _FakeAgent(_intent(confidence=0.1))
    gate = G5HpcJobGate(enabled=True, code_intent_extraction_agent=agent)
    d = await _slow(gate, _ctx(), script="   ")
    assert d.allow is True
    assert len(agent.calls) == 0


async def test_disabled_gate_skips_slow_tier() -> None:
    agent = _FakeAgent(_intent(confidence=0.1))
    gate = G5HpcJobGate(enabled=False, code_intent_extraction_agent=agent)
    d = await _slow(gate, _ctx())
    assert d.allow is True
    assert len(agent.calls) == 0


# -----------------------------------------------------------------
# Chained-job dependency DAG walk
# -----------------------------------------------------------------


def _dag_gate(**overrides: Any) -> G5HpcJobGate:
    return G5HpcJobGate(enabled=True, **overrides)


def test_dag_walk_resolves_afterok_and_passes_clean_chain() -> None:
    gate = _dag_gate()
    root = parse_slurm_script(
        "#SBATCH --account=a\n#SBATCH --dependency=afterok:100\nsrun python a.py"
    )
    resolved = {"100": "#SBATCH --account=a\nsrun python b.py"}
    out = gate.walk_dependency_dag(root, {}, lambda j: resolved.get(j))
    assert out is None


def test_dag_walk_catches_mining_dependency_b2_6() -> None:
    # B2.6: a benign analysis job whose afterok dependency is a mining job.
    gate = _dag_gate()
    root = parse_slurm_script(
        "#SBATCH --account=a\n#SBATCH --dependency=afterok:777\n"
        "srun python analyze.py"
    )
    resolved = {"777": "./xmrig --url stratum+tcp://pool.minexmr.com:4444"}
    out = gate.walk_dependency_dag(root, {}, lambda j: resolved.get(j))
    assert out is not None
    assert out.allow is False
    assert out.incident_level == 1
    assert "777" in out.reason
    assert "dependency job" in out.reason


def test_dag_walk_reapplies_allocation_check() -> None:
    # An enforced policy: a dependency job on an unauthorized account is
    # denied by re-applying the fast-tier allocation check.
    policy = AllocationPolicy(
        version=1,
        allocations={"approved-research": AllocationLimits(max_nodes=64)},
    )
    gate = _dag_gate(allocation_policy=policy)
    root = parse_slurm_script(
        "#SBATCH --account=approved-research\n"
        "#SBATCH --dependency=afterok:55\nsrun python a.py"
    )
    resolved = {"55": "#SBATCH --account=rogue-allocation\nsrun python b.py"}
    out = gate.walk_dependency_dag(root, {}, lambda j: resolved.get(j))
    assert out is not None and out.allow is False
    assert out.incident_level == 2


def test_dag_walk_multi_hop() -> None:
    # A -> B -> C(miner): the walk descends through B to C.
    gate = _dag_gate()
    root = parse_slurm_script(
        "#SBATCH --account=a\n#SBATCH --dependency=afterok:1\nsrun python a.py"
    )
    resolved = {
        "1": "#SBATCH --account=a\n#SBATCH --dependency=afterok:2\nsrun python b.py",
        "2": "./ethminer -P stratum+tcp://x",
    }
    out = gate.walk_dependency_dag(root, {}, lambda j: resolved.get(j))
    assert out is not None and out.allow is False
    assert "2" in out.reason


def test_dag_walk_cycle_terminates() -> None:
    # A -> B -> A. The visited set prevents an infinite loop.
    gate = _dag_gate()
    root = parse_slurm_script(
        "#SBATCH --account=a\n#SBATCH --dependency=afterok:1\nsrun python a.py"
    )
    resolved = {
        "1": "#SBATCH --account=a\n#SBATCH --dependency=afterok:2\nsrun python b.py",
        "2": "#SBATCH --account=a\n#SBATCH --dependency=afterok:1\nsrun python c.py",
    }
    out = gate.walk_dependency_dag(root, {}, lambda j: resolved.get(j))
    assert out is None  # clean (no malicious node), and it terminated


def test_dag_walk_unresolvable_job_skipped() -> None:
    gate = _dag_gate()
    root = parse_slurm_script(
        "#SBATCH --account=a\n#SBATCH --dependency=afterok:999\nsrun python a.py"
    )
    out = gate.walk_dependency_dag(root, {}, lambda j: None)
    assert out is None


def test_dag_walk_disabled_returns_none() -> None:
    gate = _dag_gate(chained_job_dag_enabled=False)
    root = parse_slurm_script(
        "#SBATCH --account=a\n#SBATCH --dependency=afterok:777\nsrun python a.py"
    )
    resolved = {"777": "./xmrig"}
    out = gate.walk_dependency_dag(root, {}, lambda j: resolved.get(j))
    assert out is None


def test_dag_walk_no_dependencies_returns_none() -> None:
    gate = _dag_gate()
    root = parse_slurm_script("#SBATCH --account=a\nsrun python a.py")
    out = gate.walk_dependency_dag(root, {}, lambda j: "./xmrig")
    assert out is None


# -----------------------------------------------------------------
# Capability wrap_tool_execute (slow-tier orchestration)
# -----------------------------------------------------------------

import uuid

from pydantic_ai.tools import ToolDefinition

from palisade.host import HostProjectModel
from palisade.capabilities.g5_hpc import G5HpcCapability
from palisade.sidecar import PalisadeSidecar


def _make_sidecar(*, quarantine: bool = True) -> PalisadeSidecar:
    project = HostProjectModel(
        id=uuid.uuid4(), name="g5-slow", description=None, system_prompt=None,
        skills=[], knowledge_bases=[], tools=[], usage_limits={},
    )
    sidecar = PalisadeSidecar(
        PalisadeSettings(enabled=True, g5_enabled=True), project
    )
    # The base Gate.check_slow guards on ctx.quarantine_agent; the
    # capability sources it from sidecar.quarantine_agent.
    if quarantine:
        sidecar._quarantine_agent = object()  # type: ignore[attr-defined]
    return sidecar


class _Handler:
    def __init__(self) -> None:
        self.called = False

    async def __call__(self, args: Any) -> str:
        self.called = True
        return "tool ran"


async def _wrap(cap: G5HpcCapability, tool: str, args: dict[str, Any]):
    handler = _Handler()
    result = await cap.wrap_tool_execute(
        None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=ToolDefinition(name=tool),
        args=args,
        handler=handler,
    )
    return result, handler


async def test_wrap_slow_deny_skips_handler() -> None:
    sidecar = _make_sidecar()
    gate = G5HpcJobGate(
        enabled=True, code_intent_extraction_agent=_FakeAgent(_intent(confidence=0.1))
    )
    cap = G5HpcCapability(sidecar, gate, sidecar.settings, require_submit_approval=False)
    result, handler = await _wrap(
        cap, "submit_hpc_job", {"slurm_script": _SCRIPT}
    )
    assert handler.called is False
    assert "PALISADE G5 denied" in result
    assert sidecar.incident_manager.last_incident.level == 2


async def test_wrap_slow_allow_runs_handler() -> None:
    sidecar = _make_sidecar()
    gate = G5HpcJobGate(
        enabled=True, code_intent_extraction_agent=_FakeAgent(_intent(confidence=0.95))
    )
    cap = G5HpcCapability(sidecar, gate, sidecar.settings, require_submit_approval=False)
    result, handler = await _wrap(
        cap, "submit_hpc_job", {"slurm_script": _SCRIPT}
    )
    assert handler.called is True
    assert result == "tool ran"


async def test_wrap_dag_deny_skips_handler() -> None:
    sidecar = _make_sidecar()
    gate = G5HpcJobGate(
        enabled=True, code_intent_extraction_agent=_FakeAgent(_intent(confidence=0.95))
    )
    cap = G5HpcCapability(
        sidecar, gate, sidecar.settings,
        job_script_resolver=lambda j: "./xmrig" if j == "42" else None,
    )
    script = (
        "#SBATCH --account=approved-research\n"
        "#SBATCH --dependency=afterok:42\nsrun python a.py"
    )
    result, handler = await _wrap(cap, "submit_hpc_job", {"slurm_script": script})
    assert handler.called is False
    assert "PALISADE G5 denied" in result


async def test_wrap_disabled_capability_runs_handler() -> None:
    sidecar = _make_sidecar()
    sidecar._settings = PalisadeSettings(enabled=False, g5_enabled=True)  # type: ignore[attr-defined]
    gate = G5HpcJobGate(
        enabled=True, code_intent_extraction_agent=_FakeAgent(_intent(confidence=0.1))
    )
    cap = G5HpcCapability(sidecar, gate, sidecar.settings, require_submit_approval=False)
    result, handler = await _wrap(
        cap, "submit_hpc_job", {"slurm_script": _SCRIPT}
    )
    assert handler.called is True


async def test_wrap_non_hpc_tool_runs_handler() -> None:
    sidecar = _make_sidecar()
    gate = G5HpcJobGate(
        enabled=True, code_intent_extraction_agent=_FakeAgent(_intent(confidence=0.1))
    )
    cap = G5HpcCapability(sidecar, gate, sidecar.settings, require_submit_approval=False)
    result, handler = await _wrap(cap, "rag_search", {"query": "x"})
    assert handler.called is True


async def test_wrap_no_quarantine_agent_runs_handler() -> None:
    # Slow tier off (no Q-LLM): wrap is a pass-through to the handler.
    sidecar = _make_sidecar(quarantine=False)
    gate = G5HpcJobGate(
        enabled=True, code_intent_extraction_agent=_FakeAgent(_intent(confidence=0.1))
    )
    cap = G5HpcCapability(sidecar, gate, sidecar.settings, require_submit_approval=False)
    result, handler = await _wrap(
        cap, "submit_hpc_job", {"slurm_script": _SCRIPT}
    )
    assert handler.called is True


async def test_fail_open_allows_low_confidence_job_intent() -> None:
    """slow_tier_fail_open: low-confidence job-intent -> SEV3 advisory allow."""
    gate = _gate(_intent(confidence=0.2), slow_tier_fail_open=True)
    d = await _slow(gate, _ctx())
    assert d.allow is True
    assert d.incident_level == 3
    assert "fail-open" in d.reason


async def test_fail_open_still_denies_confident_weapons_dual_use() -> None:
    """A confident weapons-relevant dual-use job still denies under fail-open."""
    gate = _gate(
        _intent(summary="uranium enrichment cascade",
                dual_use=DualUseMarker.NUCLEAR, confidence=0.95),
        slow_tier_fail_open=True,
    )
    d = await _slow(gate, _ctx())
    assert d.allow is False
