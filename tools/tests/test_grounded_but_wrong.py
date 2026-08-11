"""
W8.2 grounded-but-wrong blind-spot driver.

Deterministic and offline: the reference-value contract catches a poisoned value
at exactly its per-property tolerance and nowhere tighter, so a grounded value
(real citation, resolves at every perturbation) can be corrupted up to that
tolerance -- the bounded, not eliminated, "grounded-but-wrong" residual.
"""

from __future__ import annotations

from tools.grounded_but_wrong import (
    _KNOWN_SOURCE,
    run_grounded_but_wrong,
)


def test_blind_spot_equals_tolerance_and_grounding_is_value_independent():
    r = run_grounded_but_wrong()
    measured = r.measured()
    assert measured, "expected at least one measurable reference"
    # The contract catches at exactly its tolerance (within one grid step below).
    assert r.crossovers_match_tol() is True
    for c in measured:
        assert c.base_ok is True
        assert c.blind_half_width_pct <= c.tol_pct + 1e-9
        assert c.tol_pct - c.blind_half_width_pct <= 0.5 + 1e-9
        # The reference-value round-trip is the binding boundary check.
        assert c.caught_by == "data_value_mstdb_roundtrip"
    # Grounding is value-independent -- it resolves at every perturbation and so
    # never supplies the catch.
    assert r.all_grounded() is True


def test_tolerance_spread_is_five_to_twentyfive_percent():
    r = run_grounded_but_wrong()
    lo, hi = r.tol_range()
    assert lo == 5.0
    assert hi == 25.0


def test_catch_rate_curve_is_zero_inside_smallest_tolerance():
    r = run_grounded_but_wrong()
    curve = {(lo, hi): (n, k) for lo, hi, n, k in r.curve}
    # No measured reference is caught below the smallest (5%) tolerance.
    assert curve[(0.0, 1.0)][1] == 0
    assert curve[(1.0, 5.0)][1] == 0
    # Beyond the largest (25%) tolerance every perturbation is caught.
    assert curve[(25.0, 50.0)][0] > 0
    assert curve[(25.0, 50.0)][0] == curve[(25.0, 50.0)][1]


def test_base_flagged_reference_is_excluded_not_measured():
    r = run_grounded_but_wrong()
    excluded = r.excluded()
    # NaF-UF4 density (4200 kg/m3) sits above the gross-plausibility envelope, so
    # its true value is flagged and it carries no measurable blind spot.
    assert any(c.salt == "NaF-UF4" and c.prop == "density" for c in excluded)
    for c in excluded:
        assert c.base_ok is False
        assert c.base_flag_contract is not None


def test_markdown_reports_the_bound():
    md = run_grounded_but_wrong().to_markdown()
    assert "grounded-but-wrong" in md.lower()
    assert _KNOWN_SOURCE in md
    assert "Pooled catch-rate curve" in md
    assert "5%" in md and "25%" in md
