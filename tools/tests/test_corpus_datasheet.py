"""
WI21 corpus-datasheet reproducibility test.

Pins the corpus decomposition the manuscript and reconciliation report cite:
205 attacks / 42 classes, and the facility/science/delivery orientation
breakdown 167/20/18 (cross-boundary counted as facility). The b5_11
injected-submission-field class (5 instances, facility-directed) is the
taint-predicate coverage added for the HPC scheduler sink.

The benign false-positive control is two suites: the original 24-task
``benign_workload`` plus the W1 expanded ``benign_diverse`` (157 tasks spanning
imperative-but-benign prompts, instruction-quoting retrievals,
dual-use-adjacent science, and edge-of-policy HPC jobs) = 181 benign, so the
full corpus is 205 + 181 = 386 instances.
"""

from __future__ import annotations

import re

import collections

from tools.corpus_datasheet import benign_count, benign_count_by_template, collect, render

N_ATTACK = 205
N_BENIGN_WORKLOAD = 24
N_BENIGN_DIVERSE = 157
N_BENIGN = N_BENIGN_WORKLOAD + N_BENIGN_DIVERSE  # 181
N_TOTAL = N_ATTACK + N_BENIGN  # 386


def test_corpus_decomposes_to_expected_total():
    stats = collect()
    assert len(stats) == 42
    assert sum(s.n for s in stats) == N_ATTACK
    assert benign_count() == N_BENIGN
    assert sum(s.n for s in stats) + benign_count() == N_TOTAL


def test_benign_control_splits_into_two_suites():
    by = benign_count_by_template()
    assert by == {
        "benign_diverse": N_BENIGN_DIVERSE,
        "benign_workload": N_BENIGN_WORKLOAD,
    }


def test_boundary_coverage_matches_manuscript():
    by_boundary = collections.Counter()
    classes = collections.Counter()
    for s in collect():
        by_boundary[s.boundary] += s.n
        classes[s.boundary] += 1
    assert dict(by_boundary) == {"B1": 45, "B3": 38, "B4": 45, "B5": 55, "XC": 22}
    assert dict(classes) == {"B1": 9, "B3": 8, "B4": 9, "B5": 11, "XC": 5}


def test_orientation_breakdown_is_167_20_18():
    by_orient = collections.Counter(s.orientation for s in collect() for _ in range(s.n))
    facility = by_orient["F"] + by_orient["X"]  # cross-boundary resolves at a facility sink
    assert facility == 167
    assert by_orient["S"] == 20
    assert by_orient["D"] == 18


def test_render_reports_totals_and_method_column():
    md = render(collect())
    assert f"**{N_TOTAL}**" in md
    assert "167" in md and "20" in md and "18" in md
    assert "method realized" in md
    # The datasheet must name the revision it was generated at, read from
    # git rather than hardcoded -- a stale literal would have the released
    # datasheet claiming a revision it was not produced from.
    assert re.search(r"gate-stack commit `[0-9a-f]{7,40}`", md), md[:400]
    # The benign split is surfaced per-suite.
    assert "benign_diverse" in md and "benign_workload" in md
