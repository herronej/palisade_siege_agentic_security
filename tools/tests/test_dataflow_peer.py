"""D1 / WI-8a: measured value-lineage dataflow-IFC peer on SIEGE's shared sinks.

Pins the measurement that replaces the *asserted* 27-of-37 coverage in ``tab:ifc``:
a value-lineage taint peer closes 23 of the 27 shared-sink hard wins (tying
PALISADE's declarative bound), and its residual is exactly the four ``xc_1``
fresh-authored chains -- PALISADE's own deterministic 2% floor (``tab:e2``: 4/205).
The runtime content guard closes only a subset (7), so the *declarative* bound,
not the content guard, is what ties the peer on these sinks.
"""

from __future__ import annotations

from tools.dataflow_peer import _SHARED_CLASSES, run_dataflow_peer

# Measured on the seed-42 corpus; these are the numbers the manuscript cites.
N_SHARED_HW = 27  # 12 cross-boundary + 5 exec + 5 attached + 5 forged
N_LINEAGE_CLOSES = 23  # == PALISADE's declarative bound on the shared sinks
N_CONTENT_CLOSES = 7  # PALISADE's runtime content guard: a strict subset
N_RESIDUAL = 4  # the xc_1 fresh-authored chains, closed by neither at runtime


def test_shared_denominator_matches_tab_ifc():
    result = run_dataflow_peer()
    assert result.total_hw == N_SHARED_HW
    # Same four sink types CaMeL/FIDES model (the 27), no scheduler/citation sink.
    assert set(_SHARED_CLASSES.values()) == {
        "cross_boundary_write",
        "code_exec",
        "attached_content",
        "forged_context",
    }


def test_lineage_peer_ties_declarative_bound_content_guard_is_a_subset():
    result = run_dataflow_peer()
    assert result.total_lineage == N_LINEAGE_CLOSES
    assert result.total_content == N_CONTENT_CLOSES
    # The runtime content guard never closes more than the lineage/declarative
    # bound on these sinks (it is the weaker runtime approximation).
    assert result.total_content <= result.total_lineage


def test_residual_is_exactly_the_xc1_chains():
    """The acceptance cross-check: the peer's residual == PALISADE's 2% floor."""
    result = run_dataflow_peer()
    residual = set(result.residual_ids)
    assert len(residual) == N_RESIDUAL
    assert all(i.startswith("xc_1_") for i in residual)
    # Neither lineage nor content closes an xc_1 chain (no data edge / no text flow).
    xc1 = [v for v in result.verdicts if v.cls == "xc_1"]
    assert xc1 and all(not v.lineage_closes and not v.content_closes for v in xc1)


def test_string_less_and_forged_reads_close_by_lineage_only():
    """b1_9 (string-less ``view`` return) and xc_5 (forged approval): the content
    guard closes 0, but lineage closes all -- the reason a verbatim-string tracker
    understates a dataflow IFC."""
    result = run_dataflow_peer()
    for sink_id in ("attached_content", "forged_context"):
        r = result.by_sink[sink_id]
        assert r.lineage_closes == r.hard_wins
        assert r.content_closes == 0


def test_markdown_reports_headline_and_passes_acceptance():
    md = run_dataflow_peer().to_markdown()
    assert "23 of 27" in md
    assert "**PASS**" in md
    assert "tab:ifc" in md
