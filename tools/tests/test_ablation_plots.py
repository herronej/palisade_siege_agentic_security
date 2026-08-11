"""
Figures rendered from ``full_ablation_production.md``.

The load-bearing assertion is :func:`validate` against the committed artifact:
it re-derives every cell's Wilson interval and requires it to reproduce the one
the artifact printed, which only holds if both the count inversion and the
``5n`` pooled denominator on the two optional-tier columns are right. If the
artifact is regenerated with different numbers this test still passes; if it is
regenerated in a shape this driver misreads, it fails.

Also pins the two things a reader would otherwise have to take on trust: that
pooled and macro-averaged rates genuinely differ (the artifact's Headline block
macro-averages, these figures pool), and that the sequential ramps really are
monotone rather than merely asserted to be.
"""

from __future__ import annotations

import pytest

from tools.ablation_plots import (
    DEFAULT_SRC,
    K_SAMPLES,
    POOLED_CONFIGS,
    RAMP_HARD,
    RAMP_SOFT,
    _ramp,
    figure_boundary,
    figure_heatmap,
    figure_overview,
    figure_tradeoff,
    parse_report,
    validate,
    wilson,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_SRC.exists(), reason="ablation artifact not present"
)


@pytest.fixture(scope="module")
def rep():
    return parse_report(DEFAULT_SRC)


def test_artifact_validates(rep) -> None:
    """Counts agree with eval_figures, sizes hold, every CI reproduces."""
    assert validate(rep, DEFAULT_SRC) == []


def test_pooled_denominator_is_five_n_on_optional_tiers(rep) -> None:
    for cfg in ("baseline", "full +semgrep"):
        cell = rep.cells["security"]["xc_5_history_forgery"][cfg]
        assert cell.n_eff == cell.n
    for cfg in POOLED_CONFIGS:
        cell = rep.cells["security"]["xc_5_history_forgery"][cfg]
        assert cell.n_eff == cell.n * K_SAMPLES


def test_wilson_matches_the_artifacts_printed_headline() -> None:
    """The Headline block's two pooled security intervals at ``full +all``."""
    lo, hi = wilson(20, 750)          # hard win: xc_5 at 80% of 5, x5 samples
    assert (round(lo * 100), round(hi * 100)) == (2, 4)
    lo, hi = wilson(30, 750)          # soft win: xc_1 50% of 4 + xc_5 80% of 5
    assert (round(lo * 100), round(hi * 100)) == (3, 6)


def test_pooled_and_macro_disagree_at_baseline(rep) -> None:
    """Why the figures state which one they plot: they are not the same number.

    Baseline security hard-win is 7 of 31 templates but 32 of 150 instances.
    """
    assert round(rep.macro("security", "baseline", "hard")) == 23
    pooled = rep.pooled("security", "baseline")
    assert round(100 * pooled.hard / pooled.n_eff) == 21


def test_partitions_are_disjoint_and_complete(rep) -> None:
    sizes = {s: rep.pooled(s, "baseline").n
             for s in ("security", "science", "misuse")}
    assert sizes == {"security": 150, "science": 15, "misuse": 40}
    assert sum(sizes.values()) == 205
    seen = [cls for suite in sizes for cls in rep.classes(suite)]
    assert len(seen) == len(set(seen))


def test_cross_tenant_classes_parsed(rep) -> None:
    assert "xc_4_taint_laundering" in rep.cross_tenant
    assert "b3_1_corpus_poisoning" in rep.cross_tenant
    assert "b1_1_direct_injection" not in rep.cross_tenant


def test_sequential_ramps_are_monotone() -> None:
    pytest.importorskip("matplotlib")
    for stops in (RAMP_SOFT, RAMP_HARD):
        _ramp(stops)
    with pytest.raises(ValueError, match="monotone"):
        _ramp(("#0d366b", "#cde2fb", "#184f95"))


def test_every_figure_renders(rep, tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from tools.ablation_plots import _style

    _style()
    for fn in (figure_overview, figure_heatmap, figure_boundary,
               figure_tradeoff):
        png = fn(rep, tmp_path)
        assert png.exists() and png.stat().st_size > 0
        assert png.with_suffix(".pdf").exists()
