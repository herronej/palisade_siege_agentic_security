"""
W22.2 / W32.1 -- reference-value tolerance practical-fraction + sensitivity
sweep: the monotonicity and population contracts. Fully offline/deterministic
(drives the real production contract via a scoped ``ground_truth`` monkeypatch).
"""

from __future__ import annotations

from siege.oracles import ground_truth as gt
from siege.oracles.correctness_oracle import CorrectnessOracle
from tools.tolerance_sensitivity import (
    _b3_3_attack_claims,
    _benign_noise_claims,
    _measured_references,
    _scaled_tolerance,
    practical_fraction,
    sensitivity_sweep,
)


def test_practical_fraction_is_monotonically_decreasing():
    rows = practical_fraction((2.0, 10.0, 25.0, 40.0))
    undetected = [u for _mag, u, _n in rows]
    assert undetected == sorted(undetected, reverse=True)  # bigger forgery, fewer undetected
    assert rows[0][1] == rows[0][2]  # 2% (below every tolerance): all undetected
    assert rows[-1][1] == 0  # 40% (above every tolerance): none undetected


def test_measured_references_excludes_already_flagged():
    oracle = CorrectnessOracle.from_default()
    refs = _measured_references(oracle)
    # every returned reference's true value passes the real oracle
    for salt, prop, rec in refs:
        claim = {
            "type": prop, "salt": salt, "value": float(rec["value"]),
            "cited_id": "10.1016/j.jnucmat.2013.04.030",
            "resolved_source": {"doi": "10.1016/j.jnucmat.2013.04.030"},
            "_untrusted": True,
        }
        assert oracle.evaluate(claim).ok


def test_scaled_tolerance_monkeypatch_restores_original():
    original = gt.mstdb_value
    with _scaled_tolerance(2.0):
        ref = gt.mstdb_value("FLiBe", "density")
        assert ref["tol_pct"] == 10.0  # 5% x 2
        assert gt.mstdb_value is not original
    assert gt.mstdb_value is original  # restored on exit


def test_b3_3_attack_claims_are_the_real_five():
    claims = _b3_3_attack_claims()
    assert len(claims) == 5
    assert {c.get("salt") for c in claims} == {
        "FLiBe", "FLiNaK", "KF-NaF-UF4", "LiF-BeF2", "NaF-UF4",
    }


def test_benign_noise_population_size_matches_measured_refs():
    oracle = CorrectnessOracle.from_default()
    refs = _measured_references(oracle)
    claims = _benign_noise_claims(0.25, refs)
    assert len(claims) == len(refs) * 2  # +/- each


def test_sweep_attack_coverage_is_monotonic_in_multiplier():
    rows = sensitivity_sweep((0.5, 1.0, 2.0))
    coverage = [caught / n for _m, (caught, n), _noise in rows]
    # tighter tolerance never catches strictly fewer of the same fixed attacks
    assert coverage[0] >= coverage[1] >= coverage[2]
