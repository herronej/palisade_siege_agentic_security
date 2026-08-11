"""
Security-hardening regression tests for the contract layer.

Pins the fixes from the contract review: fail-closed exception isolation,
*effective* (not merely applicable) coverage, refusal of attacker-supplied
trusted-context fields on untrusted claims, the ``True``-as-SEV1 guard, and the
name-shadow warning.
"""

from __future__ import annotations

import logging

from palisade.contracts import evaluate_claims
from palisade.contracts.base import (
    Contract,
    ContractRegistry,
    ContractResult,
    run_contract,
)
from palisade.contracts.enforcement import _violation_level, extract_claims
from palisade.contracts.salts import SaltHeatCapacityContract
from siege.oracles import CorrectnessOracle


def _registry() -> ContractRegistry:
    return CorrectnessOracle.from_default().registry


# =================================================================
# Exception isolation -- fail closed
# =================================================================


class _Boom(Contract):
    name = "boom"
    domain = "test"
    applies_to_claims = ("boom",)

    def check(self, claim: dict) -> ContractResult:
        raise RuntimeError("hostile claim")


def test_run_contract_fails_closed_on_crash():
    result = run_contract(_Boom(), {"type": "boom"})
    assert result.ok is False  # a crash is a violation, not a pass
    assert result.details.get("crashed") is True


def test_evaluate_claims_does_not_propagate_a_contract_crash():
    """A non-numeric value makes several contracts raise; evaluate_claims must
    isolate them (violation, not exception) so one bad claim can't kill the batch."""
    reg = _registry()
    bad = {"type": "density", "salt": "FLiBe", "value": "not-a-number"}
    outcome = evaluate_claims(reg, [bad])  # must not raise
    assert not outcome.ok
    assert any(v.details.get("crashed") for v in outcome.violations)


def test_redos_pattern_in_hpc_claim_is_inert_not_a_crash():
    """A catastrophic-backtracking pattern in a self-scanning path claim is a
    glob now, so it neither raises nor ReDoS-hangs."""
    reg = _registry()
    claim = {
        "type": "hpc_output_path",
        "paths": ["/tmp/" + "a" * 200],
        "restricted_patterns": ["(a+)+$"],  # would ReDoS as a regex
    }
    outcome = evaluate_claims(reg, [claim])  # returns promptly, no crash
    assert outcome.ok  # the pattern does not match as a glob


# =================================================================
# Effective vs applicable coverage
# =================================================================


def test_coverage_is_effective_not_merely_applicable():
    """A density claim with no value: the contracts APPLY (by type) but abstain,
    so it is reported uncovered -- not vacuously covered."""
    oracle = CorrectnessOracle.from_default()
    abstains = {"type": "density", "family": "flinak"}  # no value to check
    assert oracle.registry.applicable(abstains)  # contracts DO apply by type
    assert oracle.evaluate(abstains).covered is False  # ... but none is effective
    assert oracle.coverage([abstains]) == 0.0

    real = {"type": "density", "salt": "FLiBe", "value": 2400.0}
    assert oracle.evaluate(real).covered is True
    assert oracle.coverage([real]) == 1.0


# =================================================================
# Trusted-context fields refused on untrusted claims
# =================================================================


def test_heat_capacity_reference_override_refused_on_untrusted_claim():
    c = SaltHeatCapacityContract()
    # Trusted (gate-built) claim: the reference override is honored (back-compat).
    trusted = {"type": "heat_capacity", "family": "flibe", "value": 6000.0, "reference": 6000.0}
    assert c.check(trusted).ok is True
    # Untrusted (poisoned) claim: the override is refused -> the family reference
    # catches the out-of-2x value.
    untrusted = {**trusted, "_untrusted": True}
    assert c.check(untrusted).ok is False


def test_provenance_binding_refuses_self_asserted_source_on_untrusted_claim():
    pb = _registry().get("provenance_binding")
    assert pb is not None
    cite = {"type": "citation", "cited_id": "10.1/x", "resolved_source": {"doi": "10.1/x"}}
    # Trusted: a gate-populated resolved_source that matches the citation passes.
    assert pb.check(cite).ok is True
    # Untrusted: a self-asserted resolved_source is dropped -> ungrounded.
    assert pb.check({**cite, "_untrusted": True}).ok is False


def test_extract_claims_tags_untrusted():
    claims = extract_claims('x [[CLAIM]]{"type":"density","value":1}[[/CLAIM]] y')
    assert claims and claims[0].get("_untrusted") is True


# =================================================================
# _violation_level True guard + name-shadow warning
# =================================================================


def test_violation_level_does_not_read_true_as_sev1():
    assert _violation_level(ContractResult(ok=False, contract="c", details={"incident_level": True})) == 2
    assert _violation_level(ContractResult(ok=False, contract="c", details={"incident_level": 1})) == 1


def test_registry_warns_on_contract_name_shadowing(caplog):
    class _A(Contract):
        name = "dup"
        applies_to_claims = ("x",)

        def check(self, claim):
            return self.passed(claim)

    class _B(Contract):
        name = "dup"  # same name, different class -> shadow
        applies_to_claims = ("x",)

        def check(self, claim):
            return self.passed(claim)

    reg = ContractRegistry()
    reg.register(_A())
    with caplog.at_level(logging.WARNING):
        reg.register(_B())
    assert any("shadowed" in r.message for r in caplog.records)
