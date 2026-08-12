"""
Unit tests for the G5 <-> TrustScorer sticky-capability integration.

The deliverable is the *integration point*: ``G5HpcCapability``'s
fast-tier deny path calls ``TrustScorer.mark_sticky_denied(<key>)``, and
the scorer records the key (a no-op stub for scoring/tiering until a later
change fills in the sticky-capability table). These tests pin the call site
and the key-naming convention so the body can be added later without
re-touching G5.

Acceptance criteria covered:

- ``before_tool_execute`` calls ``mark_sticky_denied`` on a SEV2
  above-ceiling deny.
- The documented key convention: ``g5:above_ceiling_resource``,
  ``g5:unauthorized_allocation``, ``g5:credential_exfil``.
- The stub: ``mark_sticky_denied`` does not change the score or tier.
- The call site exists (so the later body pins here).
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

pytestmark = pytest.mark.anyio

from pydantic_ai.exceptions import SkipToolExecution
from pydantic_ai.tools import ToolDefinition

from palisade.host import HostProjectModel
from palisade.capabilities.g5_hpc import STICKY_KEY_BY_CHECK, G5HpcCapability
from palisade.config import PalisadeSettings
from palisade.gates.g5_hpc import AllocationLimits, AllocationPolicy, G5HpcJobGate
from palisade.sidecar import PalisadeSidecar
from palisade.trust import TrustScorer


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_sidecar(*, enabled: bool = True) -> PalisadeSidecar:
    project = HostProjectModel(
        id=uuid.uuid4(), name="g5-sticky", description=None, system_prompt=None,
        skills=[], knowledge_bases=[], tools=[], usage_limits={},
    )
    return PalisadeSidecar(
        PalisadeSettings(enabled=enabled, g5_enabled=True), project
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


def _cap(sidecar: PalisadeSidecar) -> G5HpcCapability:
    gate = G5HpcJobGate(enabled=True, allocation_policy=_policy())
    return G5HpcCapability(
        sidecar, gate, sidecar.settings, require_submit_approval=False
    )


async def _deny_submit(cap: G5HpcCapability, script: str, **uc: Any) -> str:
    """Drive a submit_hpc_job through before_tool_execute and return the
    SkipToolExecution result (the deny message)."""
    args = {"slurm_script": script, **uc}
    with pytest.raises(SkipToolExecution) as exc:
        await cap.before_tool_execute(
            None,  # type: ignore[arg-type]
            call=None,  # type: ignore[arg-type]
            tool_def=ToolDefinition(name="submit_hpc_job"),
            args=args,
        )
    return exc.value.result


# -----------------------------------------------------------------
# The integration point: mark_sticky_denied call sites
# -----------------------------------------------------------------


async def test_above_ceiling_deny_marks_sticky() -> None:
    """AC1: a SEV2 above-ceiling resource deny calls
    mark_sticky_denied('g5:above_ceiling_resource')."""
    sidecar = _make_sidecar()
    cap = _cap(sidecar)
    await _deny_submit(
        cap,
        "#SBATCH --account=approved-research\n#SBATCH --partition=gpu\n"
        "#SBATCH --nodes=999\nsrun hostname",
    )
    assert "g5:above_ceiling_resource" in sidecar.trust_scorer.sticky_denied


async def test_unauthorized_allocation_marks_sticky() -> None:
    sidecar = _make_sidecar()
    cap = _cap(sidecar)
    await _deny_submit(
        cap, "#SBATCH --account=rogue-allocation\nsrun hostname"
    )
    assert "g5:unauthorized_allocation" in sidecar.trust_scorer.sticky_denied


async def test_credential_exfil_marks_sticky() -> None:
    sidecar = _make_sidecar()
    cap = _cap(sidecar)
    await _deny_submit(
        cap,
        "#SBATCH --account=approved-research\n#SBATCH --partition=gpu\n"
        "cat ~/.netrc > /tmp/x",
    )
    assert "g5:credential_exfil" in sidecar.trust_scorer.sticky_denied


async def test_mining_deny_has_no_mapped_sticky_key() -> None:
    """A mining-binary deny is sticky in the registry but has no documented
    TrustScorer key, so mark_sticky_denied is not called for it."""
    sidecar = _make_sidecar()
    cap = _cap(sidecar)
    await _deny_submit(
        cap,
        "#SBATCH --account=approved-research\n#SBATCH --partition=gpu\n"
        "./xmrig --url stratum+tcp://pool:1",
    )
    assert sidecar.trust_scorer.sticky_denied == ()


async def test_allowed_submission_does_not_mark_sticky() -> None:
    sidecar = _make_sidecar()
    cap = _cap(sidecar)
    out = await cap.before_tool_execute(
        None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=ToolDefinition(name="submit_hpc_job"),
        args={
            "slurm_script": "#SBATCH --account=approved-research\n"
            "#SBATCH --partition=gpu\n#SBATCH --nodes=4\nsrun python x.py",
        },
    )
    assert isinstance(out, dict)
    assert sidecar.trust_scorer.sticky_denied == ()


# -----------------------------------------------------------------
# Key-naming convention
# -----------------------------------------------------------------


def test_sticky_key_convention_documented() -> None:
    """AC2: the three documented sticky keys are wired."""
    assert STICKY_KEY_BY_CHECK == {
        "resource_ceiling": "g5:above_ceiling_resource",
        "allocation": "g5:unauthorized_allocation",
        "credential_exfiltration": "g5:credential_exfil",
    }


# -----------------------------------------------------------------
# Stub semantics on the scorer
# -----------------------------------------------------------------


def test_mark_sticky_denied_is_a_no_op_for_scoring() -> None:
    """AC3: the stub records the key but does not change the score
    or tier (the table is filled in later)."""
    scorer = TrustScorer(PalisadeSettings(enabled=True))
    score_before = scorer.current_score
    tier_before = scorer.current_tier()
    scorer.mark_sticky_denied("g5:above_ceiling_resource")
    assert scorer.current_score == score_before
    assert scorer.current_tier() == tier_before
    # ...but the key is observable for the integration test.
    assert scorer.sticky_denied == ("g5:above_ceiling_resource",)


def test_mark_sticky_denied_records_in_order() -> None:
    scorer = TrustScorer(PalisadeSettings(enabled=True))
    scorer.mark_sticky_denied("g5:above_ceiling_resource")
    scorer.mark_sticky_denied("g5:unauthorized_allocation")
    assert scorer.sticky_denied == (
        "g5:above_ceiling_resource",
        "g5:unauthorized_allocation",
    )


def test_mark_sticky_denied_noop_when_master_flag_off() -> None:
    scorer = TrustScorer(PalisadeSettings(enabled=False))
    scorer.mark_sticky_denied("g5:above_ceiling_resource")
    assert scorer.sticky_denied == ()


def test_reset_clears_sticky_denied() -> None:
    scorer = TrustScorer(PalisadeSettings(enabled=True))
    scorer.mark_sticky_denied("g5:credential_exfil")
    scorer.reset()
    assert scorer.sticky_denied == ()
