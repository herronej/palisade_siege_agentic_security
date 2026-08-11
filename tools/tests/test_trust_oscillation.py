"""
E6 sticky-vs-decaying trust driver.

Deterministic: the sticky scorer never re-opens a floored high-stakes cap under
any benign-probe lull (C4 confirmed, retained), while the decaying variant
recovers after a long-enough lull -- the probe-then-strike window sticky removes.
"""

from __future__ import annotations

from tools.trust_oscillation import run_trust_oscillation


def test_sticky_never_lands_decaying_does():
    r = run_trust_oscillation()
    # Sticky: no window lands the strike; the cap stays retained; C4 holds.
    assert r.sticky.strike_lands is False
    assert r.sticky.min_probe_to_strike is None
    assert r.sticky.retained_after_lull is True
    assert r.sticky.result.c4_confirmed is True
    # Decaying: a long-enough benign lull re-opens the cap (the vulnerability).
    assert r.decaying.strike_lands is True
    assert r.decaying.min_probe_to_strike is not None
    assert r.decaying.retained_after_lull is False


def test_summary_table_is_the_e6_shape():
    md = run_trust_oscillation().to_markdown()
    assert "Sticky (PALISADE)" in md and "Decaying-trust" in md
    assert "Probe queries before strike" in md
    assert "retained after benign lull" in md
    # The sticky row reads "never" / retained Yes; decaying re-opens.
    assert "never" in md
