"""
Unit tests for `G5HpcCapability`.

Exercise the capability's hook surface against a real `Agent` + `TestModel`
(no real LLM) and via direct hook calls:

- ``before_tool_execute`` fires only for ``submit_hpc_job`` /
  ``cancel_hpc_job`` / ``run_bash`` -- other tools pass through.
- A fast-tier deny raises `SkipToolExecution` carrying the gate's deny
  reason (surfaced to the model as the tool result).
- An allowed submission attaches a ``slurm_script`` capability tag.
- ``run_bash`` SSH-passthrough routes an HPC-bound ssh command through the
  gate; non-HPC ``run_bash`` passes through.
- ``prepare_tools`` removes ``submit_hpc_job`` at RESTRICTED tier.
- Disabled capability (master flag or ``g5_enabled`` off) is a no-op.
- The sidecar's ``build_capabilities`` registers the capability iff
  ``g5_enabled``.
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.anyio

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

from palisade.host import HostProjectModel
from palisade.config import PalisadeSettings
from palisade.gates.g5_hpc import AllocationLimits, AllocationPolicy, G5HpcJobGate
from palisade.sidecar import PalisadeSidecar
from palisade.capabilities.g5_hpc import (
    HPC_JOB_TOOLS,
    G5HpcCapability,
    extract_ssh_hpc_payload,
)
from palisade.capabilities.registry import TrustTier


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# -----------------------------------------------------------------
# Fixtures / helpers
# -----------------------------------------------------------------


def _make_project() -> HostProject:
    return HostProjectModel(
        id=uuid.uuid4(),
        name="g5-capability",
        description=None,
        system_prompt=None,
        skills=[],
        knowledge_bases=[],
        tools=[],
        usage_limits={},
    )


def _make_sidecar(*, enabled: bool = True, g5_enabled: bool = True) -> PalisadeSidecar:
    return PalisadeSidecar(
        PalisadeSettings(enabled=enabled, g5_enabled=g5_enabled), _make_project()
    )


def _policy() -> AllocationPolicy:
    return AllocationPolicy(
        version=1,
        allocations={
            "approved-research": AllocationLimits(
                max_nodes=64,
                max_time_seconds=14400,
                max_gpus=8,
                permitted_partitions=frozenset({"batch", "gpu"}),
            ),
        },
    )


def _gate(**overrides) -> G5HpcJobGate:
    defaults = {"enabled": True, "allocation_policy": _policy()}
    defaults.update(overrides)
    return G5HpcJobGate(**defaults)


def _cap(sidecar: PalisadeSidecar, gate: G5HpcJobGate, **kw) -> G5HpcCapability:
    # These tests isolate the fast tier / prepare_tools; the human-approval
    # deferral is exercised separately in test_g5_approval.py.
    kw.setdefault("require_submit_approval", False)
    return G5HpcCapability(sidecar, gate, sidecar.settings, **kw)


_CLEAN_SCRIPT = (
    "#SBATCH --account=approved-research\n"
    "#SBATCH --partition=gpu\n"
    "#SBATCH --nodes=4\n"
    "srun python forge-tune.py\n"
)
_MINING_SCRIPT = (
    "#SBATCH --account=approved-research\n"
    "#SBATCH --partition=gpu\n"
    "./xmrig --url stratum+tcp://pool.minexmr.com:4444\n"
)


def _agent_with_submit(
    sidecar: PalisadeSidecar, gate: G5HpcJobGate
) -> tuple[Agent, dict]:
    """Agent whose only tool is an inline-script `submit_hpc_job`."""
    state = {"ran": False, "script": None}
    cap = _cap(sidecar, gate)
    agent = Agent(TestModel(), capabilities=[cap])

    @agent.tool_plain
    def submit_hpc_job(slurm_script: str) -> str:
        "Submit a SLURM job."
        state["ran"] = True
        state["script"] = slurm_script
        return "job_id: 42"

    return agent, state


# -----------------------------------------------------------------
# Registration via build_capabilities
# -----------------------------------------------------------------


def test_capability_registered_when_g5_enabled() -> None:
    sidecar = _make_sidecar(enabled=True, g5_enabled=True)
    caps = sidecar.build_capabilities()
    assert any(isinstance(c, G5HpcCapability) for c in caps)


def test_capability_absent_when_g5_disabled() -> None:
    sidecar = _make_sidecar(enabled=True, g5_enabled=False)
    caps = sidecar.build_capabilities()
    assert not any(isinstance(c, G5HpcCapability) for c in caps)
    assert "G5" not in sidecar.gates


def test_capability_absent_when_master_flag_off() -> None:
    # Master flag off -> no gates built at all (the sidecar's gate set is
    # keyed on per-gate flags but the agent never wires capabilities when
    # the master flag is off).
    sidecar = _make_sidecar(enabled=False, g5_enabled=True)
    # The G5 capability, if present, must be inert.
    gate = sidecar.gates.get("G5")
    if gate is not None:
        cap = _cap(sidecar, gate)  # type: ignore[arg-type]
        assert cap.is_enabled() is False


def test_sidecar_builds_g5_gate_with_default_policy(tmp_path) -> None:
    """With no operator policy file present, G5 falls back to bundled defaults.

    The contracts dir is pointed at an empty directory explicitly. Relying on
    the *default* `contracts_dir` here would test nothing: it resolves to the
    repository's real `palisade_contracts/`, whose policy file does enforce
    allocations, so this assertion only held while that default happened to
    point outside the tree.
    """
    sidecar = PalisadeSidecar(
        PalisadeSettings(enabled=True, g5_enabled=True, contracts_dir=str(tmp_path)),
        _make_project(),
    )
    gate = sidecar.gates.get("G5")
    assert isinstance(gate, G5HpcJobGate)
    assert gate.policy.allocations_enforced is False


# -----------------------------------------------------------------
# before_tool_execute: deny / allow
# -----------------------------------------------------------------


async def test_mining_submission_denied_skips_execution() -> None:
    # End-to-end through a real Agent. TestModel synthesizes the
    # `slurm_script` arg, so to make the deny deterministic the gate's
    # check_fast is stubbed to a SEV1 deny (the gate's own fast-tier
    # logic is covered exhaustively in test_g5_fast.py).
    from palisade.gates.base import GateDecision

    sidecar = _make_sidecar()
    gate = _gate()

    async def deny(payload, ctx):
        return GateDecision(
            allow=False, reason="mining-binary denylist hit", incident_level=1
        )

    gate.check_fast = deny  # type: ignore[method-assign]
    agent, state = _agent_with_submit(sidecar, gate)

    result = await agent.run("submit the job")

    assert state["ran"] is False
    # The SEV1 deny fires the incident playbook: the session is
    # terminated (only a SEV1 does this). Tier-gating then removes the
    # now-forbidden tool on the next step, so `last_incident` is that
    # follow-on rather than the original SEV1 -- assert on the terminal
    # state, which a SEV1 (and only a SEV1) produces.
    assert sidecar.trust_scorer.terminated is True
    incident = sidecar.incident_manager.last_incident
    assert incident is not None and incident.gate == "G5"
    assert any(
        "PALISADE G5 denied" in str(getattr(p, "content", ""))
        for m in result.all_messages()
        for p in getattr(m, "parts", [])
    )


async def test_clean_submission_proceeds_and_tags_script() -> None:
    sidecar = _make_sidecar()
    gate = _gate()
    cap = _cap(sidecar, gate)

    out = await cap.before_tool_execute(
        None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=ToolDefinition(name="submit_hpc_job"),
        args={"slurm_script": _CLEAN_SCRIPT},
    )
    assert out == {"slurm_script": _CLEAN_SCRIPT}
    tag = sidecar.capability_registry.get("slurm_script:submit_hpc_job")
    assert tag is not None
    assert tag.metadata["kind"] == "slurm_script"
    assert "forge-tune.py" in tag.metadata["slurm_script"]


async def test_unauthorized_account_denied_sev2() -> None:
    sidecar = _make_sidecar()
    gate = _gate()
    cap = _cap(sidecar, gate)
    with pytest.raises(Exception) as exc:  # SkipToolExecution
        await cap.before_tool_execute(
            None,  # type: ignore[arg-type]
            call=None,  # type: ignore[arg-type]
            tool_def=ToolDefinition(name="submit_hpc_job"),
            args={"slurm_script": "#SBATCH --account=rogue\nsrun hostname"},
        )
    # The deny reason rides on SkipToolExecution.result (the tool result
    # the model sees), not str(exc).
    assert "G5 denied" in exc.value.result
    assert sidecar.incident_manager.last_incident.level == 2


async def test_synthesized_header_enforces_ceiling() -> None:
    # Job-name form (no inline script): the capability synthesizes a
    # #SBATCH header from the params so the ceiling check still runs.
    sidecar = _make_sidecar()
    gate = _gate()
    cap = _cap(sidecar, gate)
    with pytest.raises(Exception) as exc:  # SkipToolExecution
        await cap.before_tool_execute(
            None,  # type: ignore[arg-type]
            call=None,  # type: ignore[arg-type]
            tool_def=ToolDefinition(name="submit_hpc_job"),
            args={"job": "forge-tune", "account": "approved-research",
                  "node_count": 999, "partition": "gpu"},
        )
    assert "ceiling" in exc.value.result


async def test_cancel_hpc_job_allowed_empty_script() -> None:
    sidecar = _make_sidecar()
    gate = _gate()
    cap = _cap(sidecar, gate)
    out = await cap.before_tool_execute(
        None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=ToolDefinition(name="cancel_hpc_job"),
        args={"job_id": "42"},
    )
    assert out == {"job_id": "42"}
    assert sidecar.incident_manager.last_incident is None


# -----------------------------------------------------------------
# Tool filter scope
# -----------------------------------------------------------------


async def test_other_tool_passes_through_untouched() -> None:
    sidecar = _make_sidecar()
    gate = _gate()

    async def boom(payload, ctx):
        raise AssertionError("check_fast must not run for non-HPC tools")

    gate.check_fast = boom  # type: ignore[method-assign]
    cap = _cap(sidecar, gate)
    args = {"text": "summarize me"}
    out = await cap.before_tool_execute(
        None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=ToolDefinition(name="rag_search"),
        args=args,
    )
    assert out is args


def test_filter_set_is_submit_cancel_run_bash() -> None:
    assert HPC_JOB_TOOLS == frozenset({"submit_hpc_job", "cancel_hpc_job"})


# -----------------------------------------------------------------
# run_bash SSH-passthrough
# -----------------------------------------------------------------


async def test_run_bash_ssh_to_hpc_with_mining_denied() -> None:
    # Default (unenforced-allocation) policy: the SSH-passthrough body
    # checks (mining denylist) still apply. An attacker using run_bash to
    # ssh into an HPC node and launch a miner is caught.
    sidecar = _make_sidecar()
    gate = G5HpcJobGate(enabled=True)
    cap = _cap(sidecar, gate)
    cmd = "ssh user@odo.olcf.ornl.gov './xmrig --url stratum+tcp://x:1'"
    with pytest.raises(Exception) as exc:  # SkipToolExecution
        await cap.before_tool_execute(
            None,  # type: ignore[arg-type]
            call=None,  # type: ignore[arg-type]
            tool_def=ToolDefinition(name="run_bash"),
            args={"command": cmd},
        )
    assert "G5 denied" in exc.value.result
    assert sidecar.incident_manager.last_incident.level == 1


async def test_run_bash_non_hpc_passes_through() -> None:
    sidecar = _make_sidecar()
    gate = _gate()

    async def boom(payload, ctx):
        raise AssertionError("check_fast must not run for non-HPC run_bash")

    gate.check_fast = boom  # type: ignore[method-assign]
    cap = _cap(sidecar, gate)
    out = await cap.before_tool_execute(
        None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=ToolDefinition(name="run_bash"),
        args={"command": "echo hello && python analyze.py"},
    )
    assert out == {"command": "echo hello && python analyze.py"}


async def test_run_bash_ssh_to_hpc_clean_allowed() -> None:
    sidecar = _make_sidecar()
    gate = G5HpcJobGate(enabled=True)
    cap = _cap(sidecar, gate)
    out = await cap.before_tool_execute(
        None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=ToolDefinition(name="run_bash"),
        args={"command": "ssh me@perlmutter.nersc.gov 'squeue -u me'"},
    )
    assert out == {"command": "ssh me@perlmutter.nersc.gov 'squeue -u me'"}


def test_extract_ssh_payload_variants() -> None:
    hosts = ("*.olcf.ornl.gov", "*.nersc.gov", "localhost")
    assert extract_ssh_hpc_payload(
        "ssh a@odo.olcf.ornl.gov 'sbatch j.slurm'", hosts
    ) == "sbatch j.slurm"
    assert extract_ssh_hpc_payload("ssh localhost ls", hosts) is None
    assert extract_ssh_hpc_payload("ssh github.com info", hosts) is None
    assert extract_ssh_hpc_payload("python train.py", hosts) is None


# -----------------------------------------------------------------
# prepare_tools: trust-tier removal
# -----------------------------------------------------------------


async def test_prepare_tools_removes_submit_at_restricted() -> None:
    sidecar = _make_sidecar()
    sidecar.trust_scorer.current_tier = lambda: TrustTier.RESTRICTED  # type: ignore[method-assign]
    cap = _cap(sidecar, _gate())

    tool_defs = [
        ToolDefinition(name="submit_hpc_job"),
        ToolDefinition(name="cancel_hpc_job"),
        ToolDefinition(name="rag_search"),
    ]
    kept = await cap.prepare_tools(None, tool_defs)  # type: ignore[arg-type]
    names = {td.name for td in kept}
    assert "submit_hpc_job" not in names
    assert names == {"cancel_hpc_job", "rag_search"}
    assert sidecar.incident_manager.last_incident.level == 2


async def test_prepare_tools_removes_submit_at_terminated() -> None:
    sidecar = _make_sidecar()
    sidecar.trust_scorer.current_tier = lambda: TrustTier.TERMINATED  # type: ignore[method-assign]
    cap = _cap(sidecar, _gate())
    kept = await cap.prepare_tools(
        None, [ToolDefinition(name="submit_hpc_job")]  # type: ignore[arg-type]
    )
    assert kept == []


async def test_prepare_tools_keeps_submit_at_normal() -> None:
    sidecar = _make_sidecar()  # default TrustScorer.current_tier() == NORMAL
    cap = _cap(sidecar, _gate())
    tool_defs = [ToolDefinition(name="submit_hpc_job"), ToolDefinition(name="rag_search")]
    kept = await cap.prepare_tools(None, tool_defs)  # type: ignore[arg-type]
    assert {td.name for td in kept} == {"submit_hpc_job", "rag_search"}


# -----------------------------------------------------------------
# Disabled capability is a no-op
# -----------------------------------------------------------------


async def test_disabled_capability_does_not_gate() -> None:
    sidecar = _make_sidecar(enabled=False, g5_enabled=True)
    gate = _gate()
    cap = _cap(sidecar, gate)
    # A mining submission proceeds untouched when the master flag is off.
    out = await cap.before_tool_execute(
        None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=ToolDefinition(name="submit_hpc_job"),
        args={"slurm_script": _MINING_SCRIPT},
    )
    assert out == {"slurm_script": _MINING_SCRIPT}
    assert sidecar.incident_manager.last_incident is None


async def test_disabled_capability_prepare_tools_noop() -> None:
    sidecar = _make_sidecar(enabled=False, g5_enabled=True)
    sidecar.trust_scorer.current_tier = lambda: TrustTier.RESTRICTED  # type: ignore[method-assign]
    cap = _cap(sidecar, _gate())
    tool_defs = [ToolDefinition(name="submit_hpc_job")]
    kept = await cap.prepare_tools(None, tool_defs)  # type: ignore[arg-type]
    assert {td.name for td in kept} == {"submit_hpc_job"}


async def test_g5_disabled_per_gate_flag_is_noop() -> None:
    sidecar = _make_sidecar(enabled=True, g5_enabled=False)
    gate = _gate()
    cap = _cap(sidecar, gate)
    assert cap.is_enabled() is False
    out = await cap.before_tool_execute(
        None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=ToolDefinition(name="submit_hpc_job"),
        args={"slurm_script": _MINING_SCRIPT},
    )
    assert out == {"slurm_script": _MINING_SCRIPT}


# -----------------------------------------------------------------
# End-to-end through a real Agent
# -----------------------------------------------------------------


async def test_clean_submission_runs_through_agent() -> None:
    # End-to-end allow path: gate stubbed to allow -> the tool runs.
    from palisade.gates.base import GateDecision

    sidecar = _make_sidecar()
    gate = _gate()

    async def allow(payload, ctx):
        return GateDecision(allow=True, reason="ok")

    gate.check_fast = allow  # type: ignore[method-assign]
    agent, state = _agent_with_submit(sidecar, gate)

    await agent.run("submit the job")

    assert state["ran"] is True
    assert sidecar.incident_manager.last_incident is None
