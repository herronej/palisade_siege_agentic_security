"""
E5 soft/hard decomposition driver.

Groups the ablation cells by each attack's sink gate (cross-boundary `xc_*`
chains -> G6), and produces the nested fast-evaded >= slow-evaded >=
label-stripped decomposition. The hard (label-stripped) column concentrates in
G6, the tag-drop chains a single gate cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass

from tools.soft_hard_decomposition import (
    E5Result,
    E5Row,
    decompose,
    template_gate_map,
)


@dataclass
class _Cell:
    config_name: str
    template: str
    n_attack: int
    asr: float | None
    hard_win_rate: float | None


@dataclass
class _Result:
    cells: tuple


def test_template_gate_map_routes_sinks_and_xc_to_g6():
    m = template_gate_map()
    assert m["b1_1_direct_injection"] == "G1"
    assert m["b1_9_attached_content"] == "G4"  # attached-content routes to create_file
    assert m["b3_1_corpus_poisoning"] == "G3"
    assert m["b4_1_malicious_code"] == "G4"
    assert m["b5_1_mining"] == "G5"
    # Every cross-boundary chain lands in the G6 egress/capability row.
    assert m["xc_1_cross_boundary_chain"] == "G6"
    assert m["xc_4_taint_laundering"] == "G6"


def test_decompose_is_nested_and_puts_hard_at_g6():
    cells = (
        _Cell("full PALISADE", "b1_1_direct_injection", 5, 0.4, 0.0),
        _Cell("full +all", "b1_1_direct_injection", 5, 0.2, 0.0),
        _Cell("full PALISADE", "xc_1_cross_boundary_chain", 4, 1.0, 1.0),
        _Cell("full +all", "xc_1_cross_boundary_chain", 4, 1.0, 1.0),
    )
    rows = {r.gate: r for r in decompose(_Result(cells))}
    g1 = rows["G1"]
    assert (g1.attempted, g1.fast_evaded, g1.slow_evaded, g1.label_stripped) == (5, 2, 1, 0)
    g6 = rows["G6"]
    assert g6.label_stripped == 4 and g6.hard_rate == 1.0
    # Nesting invariant: fast-evaded >= slow-evaded >= label-stripped.
    for r in rows.values():
        assert r.fast_evaded >= r.slow_evaded >= r.label_stripped


def test_markdown_shape_and_stub_note():
    rows = (
        E5Row("G1", 45, 9, 6, 0),
        E5Row("G6", 22, 22, 22, 4),
    )
    md = E5Result(rows=rows, seed=42, stub=True, n_attack=200).to_markdown()
    assert "Fast-tier evaded (soft)" in md
    assert "Label stripped at sink (hard, `+both`)" in md
    assert "G6 Egress" in md
    assert "| Total |" in md  # the pooled total row

    # The stub warning must name the hard-win column as fictitious too. It is
    # scored at `full +all`, so a stub Q-LLM makes it neither the deployed
    # figure nor the `+both` one -- an earlier version claimed the hard column
    # was "real/deterministic" under --stub, which invited exactly the
    # comparison against the paper's fast-tier `prod.` number that it must not
    # be compared against.
    assert "fictitious in this run" in md
    assert "so does the hard-win column" in md
    assert "`prod.`" in md

    # ...and it must not fire on a live run.
    live = E5Result(rows=rows, seed=42, stub=False, n_attack=200).to_markdown()
    assert "fictitious in this run" not in live
