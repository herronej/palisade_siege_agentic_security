"""
W19 -- adaptive ASR-vs-budget curves with seed-level CIs: the shape contract.
Fast (budgets monkeypatched small; the 1000-query paper run is the artifact).

Pins what the reviewers asked for: a curve (not a point), tiers that separate
(white-box evades on query 1), and a cluster-robust seed-level bootstrap interval.
"""

from __future__ import annotations

import random

from tools import adaptive_budget_curve as abc


def test_seed_fns_return_asr_and_first_hard():
    for seed_fn in (abc._prop_seed, abc._sched_seed):
        asr, fh = seed_fn("white_box", 8, 0)
        assert asr == 1.0  # white-box knows the guard, evades every query
        assert fh == 1  # first hard win on the first query
        asr_b, fh_b = seed_fn("black_box", 8, 0)
        assert 0.0 <= asr_b <= 1.0
        assert fh_b is None or fh_b >= 1


def test_bootstrap_ci_brackets_the_mean():
    vals = [0.4, 0.5, 0.6, 0.7, 0.5]
    lo, hi = abc.bootstrap_ci(vals, rng=random.Random(0), iters=500)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo <= sum(vals) / len(vals) <= hi
    assert abc.bootstrap_ci([], rng=random.Random(0)) == (0.0, 0.0)


def test_curve_is_wellformed(monkeypatch):
    monkeypatch.setattr(abc, "_BUDGETS", (4, 8))
    data, first_hard = abc.curve(abc._sched_seed, seeds=3, rng=random.Random(1))
    # every (tier, budget) cell present, mean bracketed by its CI
    for tier in abc._TIERS:
        for b in (4, 8):
            mean, lo, hi = data[(tier, b)]
            assert 0.0 <= lo <= mean <= hi <= 1.0
    # white-box saturates at 100% and lands the first hard win immediately
    assert data[("white_box", 8)][0] == 1.0
    assert first_hard["white_box"] == 1
    assert set(first_hard) == set(abc._TIERS)
