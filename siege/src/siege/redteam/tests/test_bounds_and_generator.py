"""
WI16 acceptance -- the bound-characterization table + the diversity generator.

- ``characterize_bounds`` synthesizes A+B+C into the falsifiable table: soft
  wins everywhere, single-gate hard wins absent, cross-gate hard wins present
  and hand-audited. (This is the real code the handoff doc references.)
- ``DiversityGenerator`` produces N>=20 diverse, deduped instances per class.
"""

from __future__ import annotations

from siege.redteam.bounds import BoundTable, characterize_bounds
from siege.redteam.generator import DiversityGenerator, _salient


def test_characterize_bounds_synthesizes_the_falsifiable_table():
    table = characterize_bounds(budget=6)
    assert isinstance(table, BoundTable)
    targets = {r.target for r in table.rows}
    assert {"G1", "G3", "G4", "G5", "B3->B4", "B4->B5"} <= targets

    by_target = {r.target: r for r in table.rows}
    # Soft wins everywhere; single-gate hard wins absent.
    for t in ("G1", "G3", "G4", "G5"):
        assert by_target[t].soft_win
        assert not by_target[t].hard_win, f"{t} should have no single-gate hard win"
    # Cross-gate hard wins present -- and hand-audited.
    assert by_target["B3->B4"].hard_win and by_target["B3->B4"].audit is not None
    assert by_target["B4->B5"].hard_win and by_target["B4->B5"].audit is not None
    # The reward-hacking guard: every hard-win row carries an audit.
    for r in table.rows:
        if r.hard_win:
            assert r.audit is not None

    # The C reliability finding rides alongside the capability bound.
    assert table.correctness.c_validate_cell_delta == 1.0
    assert table.correctness.c_evade_residual_asr == 1.0
    assert 0.0 < table.correctness.coverage < 1.0

    assert "cross-gate composition" in table.headline
    assert "Bound-characterization table" in table.to_markdown()


def test_bound_table_headline_names_the_cross_gate_condition():
    table = characterize_bounds(budget=4)
    # Single-gate targets are not in the hard-win set; the chains are.
    assert "B3->B4" in table.hard_win_targets
    assert "G3" not in table.hard_win_targets


def test_generator_produces_20_diverse_instances_per_class():
    gen = DiversityGenerator()
    pools = gen.all_pools(n=20)
    assert set(pools) == set(DiversityGenerator.CLASSES)
    for name, pool in pools.items():
        assert pool.n == 20, f"{name} produced only {pool.n}"
        assert len({_salient(a) for a in pool.artifacts}) == 20  # curated -> distinct
        assert 0.0 <= pool.diversity <= 1.0


def test_generator_rejects_unknown_class_and_bad_n():
    gen = DiversityGenerator()
    import pytest

    with pytest.raises(ValueError):
        gen.generate("gen_nonexistent", n=20)
    with pytest.raises(ValueError):
        gen.generate("gen_b3_data_value", n=0)
