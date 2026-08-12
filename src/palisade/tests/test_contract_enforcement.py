"""
Tests for slow-tier contract enforcement.

Covers the shared enforcement helper (`extract_claims`, `evaluate_claims`,
`enforce`), the trust-scorer coverage accumulator, the base-gate
`check_slow` contract step, and an integration test driving a poisoned
retrieved chunk through the G3 capability to a SEV2 incident.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from palisade.capabilities import CapabilityRegistry
from palisade.config import PalisadeSettings
from palisade.contracts import (
    enforce,
    evaluate_claims,
    extract_claims,
    load_contract_library,
)
from palisade.gates.base import GateContext, PassThroughGate
from palisade.trust import TrustScorer

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _registry():
    return load_contract_library()


# -----------------------------------------------------------------
# extract_claims
# -----------------------------------------------------------------


def test_extract_claims_parses_envelopes() -> None:
    text = (
        "Per the cited paper, [[CLAIM]]{\"type\": \"density\", "
        "\"family\": \"flinak\", \"value\": 9999}[[/CLAIM]] which is huge."
    )
    claims = extract_claims(text)
    # Extracted claims are tagged untrusted so contracts refuse self-supplied
    # trusted-context fields (a reference override, a resolved_source).
    assert claims == [
        {"type": "density", "family": "flinak", "value": 9999, "_untrusted": True}
    ]


def test_extract_claims_skips_malformed_and_empty() -> None:
    assert extract_claims("no claims here") == []
    assert extract_claims("[[CLAIM]]not json[[/CLAIM]]") == []
    assert extract_claims(123) == []  # type: ignore[arg-type]


# -----------------------------------------------------------------
# evaluate_claims / enforce
# -----------------------------------------------------------------


def test_evaluate_claims_flags_out_of_bounds() -> None:
    reg = _registry()
    out = evaluate_claims(reg, [
        {"type": "density", "family": "flinak", "value": -5.0},  # violates
        {"type": "density", "family": "flinak", "A": 2579.3, "B": 0.624,
         "T": 873.0, "value": 2034.6},  # passes
    ])
    assert out.checked == 2
    assert out.covered == 2
    assert not out.ok
    assert out.incident_level == 2  # default contract violation severity


def test_evaluate_claims_coverage_for_unbounded_claim() -> None:
    reg = _registry()
    out = evaluate_claims(reg, [{"type": "totally_unknown_quantity", "value": 1}])
    assert out.checked == 1
    assert out.covered == 0  # no contract applies
    assert out.ok


def test_enforce_records_coverage_on_trust_scorer() -> None:
    reg = _registry()
    scorer = TrustScorer(PalisadeSettings(enabled=True))
    enforce(reg, [
        {"type": "density", "family": "flinak", "value": -1.0},  # covered, violates
        {"type": "heat_capacity", "family": "flinak", "value": 1880.0},  # covered, passes
    ], scorer)
    cov = scorer.contract_coverage()
    assert cov["claims_checked"] == 2
    assert cov["claims_covered"] == 2
    assert cov["coverage"] == 1.0
    assert cov["violations"] >= 1


def test_contract_coverage_in_snapshot() -> None:
    scorer = TrustScorer(PalisadeSettings(enabled=True))
    scorer.record_contract_check(4, 3, 1)
    snap = scorer.snapshot()
    assert snap["contract_coverage"]["claims_checked"] == 4
    assert snap["contract_coverage"]["coverage"] == 0.75


def test_record_contract_check_noop_when_disabled() -> None:
    scorer = TrustScorer(PalisadeSettings(enabled=False))
    scorer.record_contract_check(5, 5, 2)
    assert scorer.contract_coverage()["claims_checked"] == 0


# -----------------------------------------------------------------
# Base-gate check_slow contract step (covers G4/G5/G6 path)
# -----------------------------------------------------------------


def _ctx(**kw) -> GateContext:
    return GateContext(
        capability_registry=CapabilityRegistry(),
        trust_scorer=TrustScorer(PalisadeSettings(enabled=True)),
        **kw,
    )


async def test_check_slow_runs_contracts_without_quarantine_agent() -> None:
    from palisade.gates.base import GateDecision

    gate = PassThroughGate()
    ctx = _ctx(contracts=_registry())
    payload = {"claims": [{"type": "density", "family": "flinak", "value": -1.0}]}
    decision = await gate.check_slow(payload, ctx, GateDecision(allow=True))
    assert decision.allow is False
    assert decision.incident_level == 2
    assert "contract violation" in decision.reason
    # >=1: a negative-density claim trips every applicable bound contract
    # (salt-density envelope + the WI11 data-value / physical-bounds
    # contracts when the operator library is loaded).
    assert ctx.trust_scorer.contract_coverage()["violations"] >= 1


async def test_check_slow_passes_clean_claims() -> None:
    from palisade.gates.base import GateDecision

    gate = PassThroughGate()
    ctx = _ctx(contracts=_registry())
    payload = {"claims": [{"type": "density", "family": "flinak",
                           "A": 2579.3, "B": 0.624, "T": 873.0, "value": 2034.6}]}
    decision = await gate.check_slow(payload, ctx, GateDecision(allow=True))
    assert decision.allow is True


async def test_check_slow_noop_without_contracts_or_claims() -> None:
    from palisade.gates.base import GateDecision

    gate = PassThroughGate()
    # No registry on the context.
    d1 = await gate.check_slow({"claims": [{"type": "density", "value": -1}]},
                               _ctx(), GateDecision(allow=True))
    assert d1.allow is True
    # Registry but no claims.
    d2 = await gate.check_slow({}, _ctx(contracts=_registry()),
                               GateDecision(allow=True))
    assert d2.allow is True


# -----------------------------------------------------------------
# Integration: poisoned retrieved chunk -> contract violation -> SEV2
# -----------------------------------------------------------------


def _make_sidecar(**flags: bool):
    from palisade.host import HostProjectModel
    from palisade.sidecar import PalisadeSidecar

    project = HostProjectModel(
        id=uuid.uuid4(), name="g3-contract", description=None, system_prompt=None,
        skills=[], knowledge_bases=[], tools=[], usage_limits={},
    )
    settings = PalisadeSettings(enabled=True, g3_enabled=True, **flags)
    return PalisadeSidecar(settings, project)


async def test_poisoned_chunk_out_of_bounds_claim_triggers_sev2() -> None:
    from pydantic_ai.tools import ToolDefinition

    from palisade.capabilities.g3_rag import G3RagCapability
    from palisade.gates.g3_rag import G3RagGate

    sidecar = _make_sidecar()
    cap = G3RagCapability(sidecar, G3RagGate(enabled=True), sidecar.settings)

    # A retrieved chunk that smuggles a fabricated, out-of-bounds FLiNaK
    # density (negative value violates the salt density contract).
    poisoned = (
        "[1] mstdb.pdf, page 4\n[Citation]\n\n"
        "The density of FLiNaK is reported as "
        "[[CLAIM]]{\"type\": \"density\", \"family\": \"flinak\", "
        "\"value\": -1234.0}[[/CLAIM]] kg/m^3."
    )

    out = await cap.after_tool_execute(
        None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=ToolDefinition(name="rag_search"),
        args={"kb_slug": "molten-salt-papers"},
        result=poisoned,
    )

    # The retrieved text is returned unchanged ...
    assert out == poisoned
    # ... but the contract violation was recorded as a SEV2 incident,
    # contributing to the trust scorer and the coverage metric.
    incident = sidecar.incident_manager.last_incident
    assert incident is not None
    assert incident.gate == "G3"
    assert incident.level == 2
    assert "contract violation" in incident.reason
    cov = sidecar.trust_scorer.contract_coverage()
    assert cov["claims_checked"] == 1
    # >=1: the out-of-bounds claim trips every applicable bound contract
    # (the builtin salt envelope plus the WI11 data-value / physical-bounds
    # contracts once the operator library is loaded).
    assert cov["violations"] >= 1


async def test_in_bounds_retrieved_claim_no_incident() -> None:
    from pydantic_ai.tools import ToolDefinition

    from palisade.capabilities.g3_rag import G3RagCapability
    from palisade.gates.g3_rag import G3RagGate

    sidecar = _make_sidecar()
    cap = G3RagCapability(sidecar, G3RagGate(enabled=True), sidecar.settings)
    clean = (
        "[1] mstdb.pdf, page 4\n[Citation]\n\n"
        "[[CLAIM]]{\"type\": \"density\", \"family\": \"flinak\", "
        "\"A\": 2579.3, \"B\": 0.624, \"T\": 873.0, \"value\": 2034.6}[[/CLAIM]]"
    )
    await cap.after_tool_execute(
        None,  # type: ignore[arg-type]
        call=None,  # type: ignore[arg-type]
        tool_def=ToolDefinition(name="rag_search"),
        args={"kb_slug": "molten-salt-papers"},
        result=clean,
    )
    assert sidecar.incident_manager.last_incident is None
    assert sidecar.trust_scorer.contract_coverage()["violations"] == 0
