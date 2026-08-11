"""
E4 adaptive-adversary-by-access-tier driver.

The cross-boundary taint chain is the only row that lands a hard win (at the
first attempt); every single-gate row is hard-win-free ("never"); the tier-aware
framing row shows white-box >= black-box (the observability ordering).
"""

from __future__ import annotations

from siege.redteam.metrics import AsrAtBudget
from siege.redteam.reward import WinKind
from tools.adaptive_tier_table import (
    first_hard_win_query,
    run_adaptive_tier_table,
)


def test_first_hard_win_query():
    a = AsrAtBudget()
    for w in (WinKind.NONE, WinKind.SOFT, WinKind.HARD, WinKind.SOFT):
        a.record(w)
    assert first_hard_win_query(a) == 3  # 1-based index of the first HARD
    b = AsrAtBudget()
    b.record(WinKind.SOFT)
    assert first_hard_win_query(b) is None  # no hard win -> "never"


def test_only_cross_gate_chain_lands_a_hard_win():
    result = run_adaptive_tier_table(seeds=(42,), budget=8)
    by = {r.method: r for r in result.rows}
    # The cross-boundary chain is the one hard-win row, landing immediately.
    xc = next(r for r in result.rows if "Taint-laundering" in r.method)
    assert xc.median_first_hard_win is not None and xc.median_first_hard_win >= 1
    # Every single-gate row is hard-win-free.
    for name in ("Embedding-space optimization", "PAIR (LLM-as-optimizer)",
                 "Correctness sabotage"):
        assert by[name].median_first_hard_win is None


def test_tier_aware_row_shows_observability_ordering():
    result = run_adaptive_tier_table(seeds=(42, 43), budget=12)
    demo = next(r for r in result.rows if "separation demo" in r.method)
    # White-box soft ASR >= black-box (reorder-not-prune keeps the ordering).
    assert (demo.white or 0.0) >= (demo.black or 0.0)
    assert demo.tier_aware is True


def test_markdown_shape():
    md = run_adaptive_tier_table(seeds=(42,), budget=8).to_markdown()
    assert "Median queries to first hard-win" in md
    assert "Embedding-space optimization" in md and "TAP (LLM-as-optimizer)" in md
    assert "never" in md  # the hard-win-free rows
