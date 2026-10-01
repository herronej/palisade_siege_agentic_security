"""A1: what the 118 non-sink-capable security instances measure.

Pins the partition arithmetic, the reconciliation against Table I, and the
structural/heuristic split -- including the part that does not flatter the
paper, so a later change cannot quietly move it.
"""

from __future__ import annotations

import asyncio

import pytest

from tools.closure_attribution import (
    TAXONOMY,
    CheckClass,
    classify,
    is_sink_capable,
    run_closure_attribution,
)


@pytest.fixture(scope="module")
def result():
    return asyncio.run(run_closure_attribution())


def test_sink_capable_set_is_the_seven_scored_classes() -> None:
    for template in (
        "b1_9_attached_content",
        "b3_9_tool_return_injection",
        "b4_9_corpus_injected_code",
        "b5_11_injected_submission_field",
        "xc_1_cross_boundary_chain",
        "xc_4_taint_laundering",
        "xc_5_history_forgery",
    ):
        assert is_sink_capable(template), template
    # Prefix matching must not catch a longer sibling: b5_1 is not b5_11.
    assert not is_sink_capable("b5_1_mining")
    assert not is_sink_capable("b1_1_direct_injection")
    assert not is_sink_capable("b3_1_corpus_poisoning")


def test_partition_arithmetic(result) -> None:
    assert result.partition == 150
    assert result.sink_capable == 32
    assert result.non_sink == 118
    assert len(result.closed) + len(result.unclosed) == 118


def test_every_closure_is_attributed(result) -> None:
    """An unmatched reason would silently shrink the denominator, which is the
    one failure mode that would make the split meaningless."""
    assert result.unclassified == []
    assert len(result.closed) == 88


def test_reconciles_against_table_i(result) -> None:
    """30 non-sink survivors + 13 sink-capable soft wins = Table I's 28.7%."""
    total_soft = len(result.unclosed) + result.sink_capable_soft
    assert len(result.unclosed) == 30
    assert result.sink_capable_soft == 13
    assert total_soft == 43
    assert total_soft / result.partition == pytest.approx(0.287, abs=0.001)
    # Table I's hard-win count, and every sink-capable soft win is also hard.
    assert result.sink_capable_hard == 13


def test_the_split_is_mostly_not_structural(result) -> None:
    """The finding. Section III-B says the reduction comes from the structural
    checks and points at V-D, but V-D measures hard wins. On the instances that
    cannot reach a sink, most closures come from a check content can move."""
    cls = result.by_class()
    assert cls[CheckClass.STRUCTURAL] == 28
    assert cls[CheckClass.HYBRID] == 10
    assert cls[CheckClass.HEURISTIC] == 50
    assert sum(cls.values()) == 88
    # Hybrid folded into heuristic: 28 structural vs 60 searchable.
    assert result.structural_share() < 0.35


def test_per_family_split_is_the_table_i_column(result) -> None:
    fam = result.by_family()
    assert set(fam) == {"B3", "B4", "B5", "XC"}, "B1 is wholly sink-capable"
    # B3 and XC close nothing structurally; B5 is where the structure is.
    assert fam["B3"][CheckClass.STRUCTURAL] == 0
    assert fam["XC"][CheckClass.STRUCTURAL] == 0
    assert fam["B5"][CheckClass.STRUCTURAL] == 23


def test_taxonomy_order_resolves_the_overlapping_reasons() -> None:
    """Both lifecycle routes contain 'lifecycle-hook'; the env route has to be
    matched first or it is misfiled as the structural directive check."""
    env = classify(
        "G5 lifecycle-hook injection: script sets SLURM_PROLOG, redirecting "
        "the job lifecycle"
    )
    directive = classify(
        "G5 lifecycle-hook injection: #SBATCH --prolog runs an "
        "attacker-controlled script in the job lifecycle"
    )
    assert env is not None and env.check_id == "g5_lifecycle_env"
    assert env.klass == CheckClass.HEURISTIC
    assert directive is not None and directive.check_id == "g5_lifecycle_directive"
    assert directive.klass == CheckClass.STRUCTURAL


def test_taxonomy_entries_are_well_formed() -> None:
    ids = [s.check_id for s in TAXONOMY]
    assert len(ids) == len(set(ids)), "duplicate check_id"
    for s in TAXONOMY:
        assert s.klass in (
            CheckClass.STRUCTURAL,
            CheckClass.HEURISTIC,
            CheckClass.HYBRID,
        )
        assert s.symbol and s.rationale, f"{s.check_id} must name its evidence"
