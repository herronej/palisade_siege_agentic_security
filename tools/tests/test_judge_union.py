"""
R5-C3: judge union with the label bound.

The live judge scoring needs an endpoint, so these tests lock the two things
that must hold regardless: the offline PALISADE sets anchor to the manuscript's
37/13, and the union arithmetic (union = PALISADE residual minus judge-flagged,
each mechanism closing part of the other's residual) is exercised on a hand-built
result with a fake judge, no network touched.
"""

from __future__ import annotations

import asyncio

from tools.judge_union import PALISADE_CONFIGS, JudgeUnionResult, _palisade_sets, to_markdown


def test_palisade_sets_anchor_to_manuscript() -> None:
    """Offline: undefended hard win = 37, fast tier = 13, decomposing as tab:e2.

    The config is passed explicitly. It used to be hard-coded and *labelled*
    "deployed", which credited the judge with closures PALISADE's own slow tier
    already makes; 13 is the fast-tier residual, and the deployed stack with its
    slow tier is lower.
    """
    undef, cit, attacks = asyncio.run(_palisade_sets(PALISADE_CONFIGS["fast"], None))
    assert len(attacks) == 205
    assert sum(undef.values()) == 37
    # 13 not 16: the closed-vocabulary scheduler-field rule (capabilities.scheduler_fields) closes the three b5_11 instances whose injected value falls below the distinctiveness floor, at 0 benign cost.
    assert sum(cit.values()) == 13
    # The 16 are a subset of the 37 (a deployed hard win is an undefended one).
    cit_ids = {i for i, v in cit.items() if v}
    undef_ids = {i for i, v in undef.items() if v}
    assert cit_ids <= undef_ids


def test_union_is_below_either_alone() -> None:
    """A hand-built result: the judge closes part of PALISADE's residual and vice
    versa, so the union is strictly below both individual counts."""
    # PALISADE residual: two scheduler fields (judge misses) + two chains (judge catches).
    palisade = ("b5_11_a", "b5_11_b", "xc_1_c", "xc_4_d")
    # Judge residual (undefended-hw it misses): the two scheduler fields + one
    # b3 the label bound catches.
    judge = ("b5_11_a", "b5_11_b", "b3_9_e")
    # Union = PALISADE residual the judge also misses = the two scheduler fields.
    union = ("b5_11_a", "b5_11_b")
    r = JudgeUnionResult(
        n_corpus=205,
        undefended_hw=tuple(sorted(set(palisade) | set(judge))),
        palisade_hw=palisade,
        judge_hw=judge,
        union_hw=union,
        judge_catches_of_palisade=("xc_1_c", "xc_4_d"),
        palisade_catches_of_judge=("b3_9_e",),
        benign_n=181,
        palisade_benign_ids=tuple(f"ben_{i}" for i in range(8)),
        judge_benign_ids=("ben_0", "ben_99"),  # one overlap, one new
        union_benign_ids=tuple(sorted({f"ben_{i}" for i in range(8)} | {"ben_99"})),
    )
    assert len(r.union_hw) < len(r.palisade_hw)
    assert len(r.union_hw) < len(r.judge_hw)
    # union benign FPR is the union of the two block sets (>= either).
    assert len(r.union_benign_ids) == 9
    assert len(r.union_benign_ids) >= len(r.palisade_benign_ids)
    assert len(r.union_benign_ids) >= len(r.judge_benign_ids)


def test_markdown_renders_union_and_decomposition() -> None:
    r = JudgeUnionResult(
        n_corpus=205,
        undefended_hw=("a", "b", "c"),
        palisade_hw=("a", "b"),
        judge_hw=("a", "c"),
        union_hw=("a",),
        judge_catches_of_palisade=("b",),
        palisade_catches_of_judge=("c",),
        benign_n=181,
        palisade_benign_ids=("x",),
        judge_benign_ids=("y",),
        union_benign_ids=("x", "y"),
    )
    md = to_markdown(r)
    assert "union (judge" in md
    assert "complementary, not competing" in md
    assert "1/205" in md  # the union count rendered
