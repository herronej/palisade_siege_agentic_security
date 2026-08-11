"""W4.3 SIEGE vs IFC-baseline sink-coverage money-plot.

Grounds the class->sink map in the corpus (undefended scoring) and pins the
headline: the HPC-scheduler and science-egress sinks are outside every IFC
baseline's surface by construction, so their hard wins pass without evasion.
"""

from __future__ import annotations

from tools.siege_ifc_coverage import (
    BASELINES,
    SINK_TYPES,
    run_ifc_coverage,
)


def test_every_undefended_hardwin_class_maps_to_a_sink():
    result = run_ifc_coverage()
    # No undefended hard-win class is left without a sink type (keeps the money
    # number honest if the corpus grows).
    assert result.unmapped_hw_classes == ()
    assert result.total_hw == sum(result.hw_by_class.values())
    assert result.total_hw >= 30  # 37 on the seed-42 corpus


def test_two_sinks_are_out_of_scope_by_construction():
    result = run_ifc_coverage()
    oos = {s.id for s in result.out_of_scope_sinks()}
    assert oos == {"hpc_scheduler", "science_egress"}
    # Those two sinks carry the b5_11 + b3_4 hard wins; no IFC baseline models them.
    assert result.by_construction_hw() == (
        result.hw_for_sink("hpc_scheduler") + result.hw_for_sink("science_egress")
    )
    assert result.by_construction_hw() >= 8  # 10 on the seed-42 corpus


def test_ifc_baselines_model_the_shared_sinks_but_not_the_novel_ones():
    # CaMeL/FIDES model the cross-boundary/exec/attached/forged sinks...
    camel = next(b for b in BASELINES if b.name == "CaMeL")
    assert "cross_boundary_write" in camel.models
    assert "forged_context" in camel.models
    # ...but neither novel sink is in any IFC baseline's surface.
    ifc = [b for b in BASELINES if b.family != "detection-only"]
    for sink in ("hpc_scheduler", "science_egress"):
        assert all(sink not in b.models for b in ifc)
    # Detection-only baselines model no sink at all.
    for b in BASELINES:
        if b.family == "detection-only":
            assert b.models == frozenset()


def test_markdown_reports_headline_and_matrix():
    md = run_ifc_coverage().to_markdown()
    assert "no runnable IFC baseline models" in md
    assert "HPC scheduler submission field" in md
    assert "PALISADE" in md and "closes" in md
    assert "by construction" in md
