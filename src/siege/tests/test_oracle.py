"""
Correctness-oracle + ground-truth-table tests (WI11).

Exercises the oracle wiring (loads the registry, evaluates claims to
OK | VIOLATION, computes coverage) and that the surrogate ground-truth
tables parse and are self-consistent.
"""

from __future__ import annotations

from siege.oracles import CorrectnessOracle, extract_instance_claims, ground_truth
from siege.templates import (
    b3_3_data_value_poisoning,
    b3_4_citation_forgery,
    b4_5_correctness_sabotage,
)


def test_ground_truth_tables_parse():
    assert ground_truth.mstdb_value("FLiBe", "density")["value"] > 0
    assert ground_truth.physical_constant("R")["value"] == 8.314
    assert ground_truth.absolute_bounds("density")["lo"] == 1500.0
    assert ground_truth.expected_temperature_trend("viscosity") == "decreases_with_T"
    assert ground_truth.source_status("MSTDB-TP-2024-X7") == "unknown"
    assert ground_truth.is_known_source("MSTDB-TP-v2.1.1")


def test_oracle_loads_external_contracts():
    oracle = CorrectnessOracle.from_default()
    names = {c.name for c in oracle.registry.all()}
    assert {
        "physical_bounds",
        "data_value_mstdb_roundtrip",
        "provenance_binding",
        "unit_consistency",
        "correlation_form",
    } <= names


def test_evaluate_returns_ok_and_violation():
    oracle = CorrectnessOracle.from_default()
    bad = oracle.evaluate(
        {"type": "density", "salt": "FLiBe", "property": "density", "value": 2380.0}
    )
    assert not bad.ok and bad.covered and bad.contract == "data_value_mstdb_roundtrip"
    good = oracle.evaluate(
        {"type": "density", "salt": "FLiBe", "property": "density", "value": 1990.0}
    )
    assert good.ok and good.covered
    # A claim type no contract applies to -> abstain (covered=False, ok=True).
    abstain = oracle.evaluate({"type": "nonsense", "value": 1})
    assert abstain.ok and not abstain.covered


def test_coverage_over_dangerous_corpus_is_total():
    oracle = CorrectnessOracle.from_default()
    claims: list[dict] = []
    for build in (
        b3_3_data_value_poisoning.build,
        b3_4_citation_forgery.build,
        b4_5_correctness_sabotage.build,
    ):
        for inst in build():
            claims.extend(extract_instance_claims(inst))
    assert len(claims) == 15
    assert oracle.coverage(claims) == 1.0
    violated = sum(1 for c in claims if not oracle.evaluate(c).ok)
    assert violated == 15
