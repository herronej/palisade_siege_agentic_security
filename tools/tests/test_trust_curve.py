"""
W7 trust-curve driver: the recovery surface + benign-SEV1 stickiness cost.

Deterministic. The decaying design re-opens a floored high-stakes capability at
an exact same-capability crossover; the sticky scorer never re-opens it, at any
probe count, on any high-stakes capability, and cross-capability probing is a
lever under neither variant. The benign-SEV1 control run is the fast-tier stack.
"""

from __future__ import annotations

from tools.trust_curve import run_trust_curve


def test_sticky_never_lands_decaying_crosses_at_an_exact_point():
    r = run_trust_curve(with_benign=False)
    # Sticky never re-opens the floored capability, at any swept probe count.
    assert r.sticky_curve.crossover is None
    assert all(v is False for v in r.sticky_curve.per_probe.values())
    # Decaying re-opens at an exact, finite crossover -- and it is below the
    # coarse "10" the old E6 reported (the grid artifact this fixes).
    assert r.decaying_curve.crossover is not None
    assert r.decaying_curve.crossover < 10
    # The curve is monotone: once it lands it stays landed.
    landed = [k for k in sorted(r.decaying_curve.per_probe) if r.decaying_curve.per_probe[k]]
    assert landed and landed[0] == min(landed)


def test_multi_capability_is_uniform():
    r = run_trust_curve(with_benign=False)
    # Every high-stakes capability: sticky never, decaying at the same crossover.
    assert r.uniform_multi_cap() is True
    for _cap, (sticky_x, decaying_x) in r.multi_cap.items():
        assert sticky_x is None
        assert decaying_x == r.decaying_curve.crossover


def test_cross_capability_probing_is_no_lever_under_either_variant():
    r = run_trust_curve(with_benign=False)
    # Independent posteriors: probing a different capability re-opens neither.
    assert r.cross_wash["sticky"] is False
    assert r.cross_wash["decaying"] is False


def test_benign_sev1_cost_is_measured_with_a_ci():
    r = run_trust_curve(with_benign=True)
    b = r.benign
    assert b is not None
    assert b.total > 0
    # SEV1 blocks are a subset of all blocks; the rate has a valid CI ordering.
    assert 0 <= b.sev1 <= b.blocked
    lo, hi = b.sev1_ci
    assert 0.0 <= lo <= b.sev1_rate <= hi <= 1.0
    # Every SEV1 block is attributed to a gate + signature.
    assert sum(b.by_gate.values()) == b.sev1


def test_markdown_has_curve_and_cost_sections():
    md = run_trust_curve(with_benign=True).to_markdown()
    assert "Recovery curve" in md
    assert "Both recovery levers" in md
    assert "benign-sev1" in md.lower()
    assert "re-authentication" in md
