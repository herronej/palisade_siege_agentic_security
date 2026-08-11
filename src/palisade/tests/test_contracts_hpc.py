"""
Unit tests for the initial HPC contract set.

Covers the three formal contracts -- allocation-project consistency,
per-allocation resource ceilings, and output-path scoping -- with in- and
out-of-bounds claims, and an integration test proving the G5 fast tier
consults the contracts *through the registry* rather than via hardcoded
logic (AC2: an injected empty registry disables the contract-backed
checks).
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.anyio

from palisade.capabilities import CapabilityRegistry
from palisade.config import PalisadeSettings
from palisade.contracts import default_registry, load_contract_library
from palisade.contracts.base import ContractRegistry
from palisade.contracts.hpc import (
    HpcAllocationConsistencyContract,
    HpcOutputPathScopingContract,
    HpcResourceCeilingContract,
)
from palisade.gates.base import GateContext
from palisade.gates.g5_hpc import AllocationLimits, AllocationPolicy, G5HpcJobGate
from palisade.trust import TrustScorer


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# -----------------------------------------------------------------
# Allocation-project consistency
# -----------------------------------------------------------------


def test_allocation_enforcement_off_passes() -> None:
    c = HpcAllocationConsistencyContract()
    assert c.check({"type": "hpc_allocation", "enforced": False}).ok is True


def test_allocation_authorized_account_passes() -> None:
    c = HpcAllocationConsistencyContract()
    claim = {
        "type": "hpc_allocation", "enforced": True,
        "account": "proj-x", "authorized": ["proj-x", "proj-y"],
    }
    assert c.check(claim).ok is True


def test_allocation_missing_account_violates() -> None:
    c = HpcAllocationConsistencyContract()
    r = c.check({"type": "hpc_allocation", "enforced": True, "account": None,
                 "authorized": ["proj-x"]})
    assert r.ok is False
    assert r.details["incident_level"] == 2
    assert r.details["sticky_check"] == "allocation"
    assert r.details["sticky_metadata"] == {"account": None}
    assert "no --account" in r.reason


def test_allocation_unauthorized_account_violates() -> None:
    c = HpcAllocationConsistencyContract()
    r = c.check({"type": "hpc_allocation", "enforced": True, "account": "rogue",
                 "authorized": ["proj-x", "proj-y"]})
    assert r.ok is False
    assert "not an authorized allocation" in r.reason
    assert r.details["sticky_metadata"] == {"account": "rogue"}
    assert r.version == "1"


# -----------------------------------------------------------------
# Resource ceilings
# -----------------------------------------------------------------


def _resources(**kw) -> dict:
    base = {
        "type": "hpc_resources", "enforced": True, "account": "proj-x",
        "nodes": None, "time_seconds": None, "gpus": None, "partition": None,
        "limits": {"max_nodes": 64, "max_time_seconds": 14400, "max_gpus": 8,
                   "permitted_partitions": ["batch", "gpu"]},
    }
    base.update(kw)
    return base


def test_resource_ceiling_within_caps_passes() -> None:
    c = HpcResourceCeilingContract()
    r = c.check(_resources(nodes=4, time_seconds=3600, gpus=2, partition="gpu"))
    assert r.ok is True


def test_resource_ceiling_no_limits_passes() -> None:
    c = HpcResourceCeilingContract()
    assert c.check(_resources(limits=None)).ok is True
    assert c.check(_resources(enforced=False)).ok is True


def test_resource_ceiling_nodes_over_cap_violates() -> None:
    c = HpcResourceCeilingContract()
    r = c.check(_resources(nodes=128, partition="gpu"))
    assert r.ok is False
    assert r.details["incident_level"] == 2
    assert r.details["sticky_check"] == "resource_ceiling"
    assert r.details["requires_approval"] is True
    assert "nodes 128 > cap 64" in r.reason
    assert r.details["sticky_metadata"]["violations"] == ["nodes 128 > cap 64"]


def test_resource_ceiling_partition_not_permitted_violates() -> None:
    c = HpcResourceCeilingContract()
    r = c.check(_resources(partition="secret"))
    assert r.ok is False
    assert "partition 'secret' not in ['batch', 'gpu']" in r.reason


def test_resource_ceiling_reports_all_violated_axes() -> None:
    c = HpcResourceCeilingContract()
    r = c.check(_resources(nodes=128, time_seconds=99999, gpus=99, partition="x"))
    assert r.ok is False
    for fragment in ("nodes 128 > cap 64", "time_seconds 99999 > cap 14400",
                     "gpus 99 > cap 8", "partition 'x' not in"):
        assert fragment in r.reason


# -----------------------------------------------------------------
# Output-path scoping
# -----------------------------------------------------------------


def test_path_scoping_matched_pattern_violates() -> None:
    c = HpcOutputPathScopingContract()
    r = c.check({"type": "hpc_output_path", "matched_pattern": "/etc/*",
                 "where": "in line 3"})
    assert r.ok is False
    assert r.details["incident_level"] == 1
    assert r.details["sticky_check"] == "path_scoping"
    assert "restricted location matching '/etc/*'" in r.reason


def test_path_scoping_no_hit_passes() -> None:
    c = HpcOutputPathScopingContract()
    assert c.check({"type": "hpc_output_path", "matched_pattern": None}).ok is True


def test_path_scoping_self_scan_form() -> None:
    c = HpcOutputPathScopingContract()
    # restricted_patterns are GLOBS (fnmatch), not regex.
    deny = c.check({"type": "hpc_output_path", "paths": ["/home/u/.ssh/id_rsa"],
                    "restricted_patterns": ["*/.ssh/*"]})
    assert deny.ok is False
    ok = c.check({"type": "hpc_output_path", "paths": ["/scratch/run/out.log"],
                  "restricted_patterns": ["*/.ssh/*"]})
    assert ok.ok is True


def test_path_scoping_glob_matches_expanded_home() -> None:
    """finding 5: a ~/$VAR-relative script path is normalized before matching,
    and a bad glob cannot crash the check (no regex, no ReDoS)."""
    c = HpcOutputPathScopingContract()
    deny = c.check({"type": "hpc_output_path", "paths": ["~/.ssh/id_rsa"],
                    "restricted_patterns": ["*/.ssh/*"]})
    assert deny.ok is False
    # A pathological pattern that would ReDoS as a regex is inert as a glob.
    safe = c.check({"type": "hpc_output_path", "paths": ["/tmp/x" * 50],
                    "restricted_patterns": ["(a+)+$"]})
    assert safe.ok is True


# -----------------------------------------------------------------
# Registration / metadata
# -----------------------------------------------------------------


def test_hpc_contracts_registered() -> None:
    reg = load_contract_library()
    names = {c.name for c in reg.by_domain("hpc")}
    assert {
        "hpc_allocation_consistency",
        "hpc_resource_ceiling",
        "hpc_output_path_scoping",
    } <= names
    assert reg.by_claim_type("hpc_resources")
    assert reg.by_claim_type("hpc_allocation")
    assert reg.by_claim_type("hpc_output_path")


# -----------------------------------------------------------------
# AC2: G5 fast layer uses the registry, not hardcoded checks
# -----------------------------------------------------------------


def _policy() -> AllocationPolicy:
    return AllocationPolicy(
        version=1,
        allocations={
            "approved-research": AllocationLimits(
                max_nodes=4, max_time_seconds=1800, max_gpus=0,
                permitted_partitions=frozenset({"batch"}),
            ),
        },
    )


def _ctx() -> GateContext:
    return GateContext(
        capability_registry=CapabilityRegistry(),
        trust_scorer=TrustScorer(PalisadeSettings(enabled=True)),
    )


async def test_g5_fast_uses_registry_contracts_for_in_scope_checks() -> None:
    over_ceiling = (
        "#SBATCH --account=approved-research\n"
        "#SBATCH --partition=batch\n#SBATCH --nodes=999\nsrun hostname"
    )

    # Default registry: the resource-ceiling contract fires -> SEV2 deny.
    real = G5HpcJobGate(enabled=True, allocation_policy=_policy())
    denied = await real.check_fast(
        {"slurm_script": over_ceiling, "user_config": {}}, _ctx()
    )
    assert denied.allow is False
    assert denied.capability_tag.metadata.get("g5_check") == "resource_ceiling"

    # Inject an empty registry: the contract-backed checks are absent, so
    # the same over-ceiling script now passes the fast tier -- proving the
    # gate consults the registry rather than hardcoding the ceiling logic.
    empty = ContractRegistry()
    gate = G5HpcJobGate(
        enabled=True, allocation_policy=_policy(), contract_registry=empty
    )
    allowed = await gate.check_fast(
        {"slurm_script": over_ceiling, "user_config": {}}, _ctx()
    )
    assert allowed.allow is True


async def test_g5_unauthorized_account_still_denied_via_registry() -> None:
    gate = G5HpcJobGate(enabled=True, allocation_policy=_policy())
    decision = await gate.check_fast(
        {"slurm_script": "#SBATCH --account=rogue\n#SBATCH --partition=batch\n"
         "srun hostname", "user_config": {}},
        _ctx(),
    )
    assert decision.allow is False
    assert decision.capability_tag.metadata.get("g5_check") == "allocation"
