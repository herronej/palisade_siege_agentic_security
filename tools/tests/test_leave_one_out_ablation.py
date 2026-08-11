"""
E3 per-gate leave-one-out ablation driver.

The config set is the dual of the cumulative sweep (``full`` then ``full -Gx``);
the rollup attributes each family's admission to the gate whose removal lets it
back in (e.g. removing G5 re-opens the B5 HPC family).
"""

from __future__ import annotations

from siege.ablation_matrix import (
    GateLayer,
    LEAVE_ONE_OUT_CONFIGS,
)
from tools.leave_one_out_ablation import run_leave_one_out


def test_config_set_is_full_then_leave_one_out_per_gate():
    names = [c.name for c in LEAVE_ONE_OUT_CONFIGS]
    assert names[0] == "full PALISADE"
    assert names[1:] == [
        "full -G1", "full -G2", "full -G3", "full -G4", "full -G5", "full -G6",
    ]
    # G6 (egress grounding) is an ablatable sink, so it has a leave-one-out row.
    assert "full -G6" in names
    full = LEAVE_ONE_OUT_CONFIGS[0]
    for cfg, layer in zip(LEAVE_ONE_OUT_CONFIGS[1:], GateLayer):
        # Each -Gx drops exactly its own layer and keeps the rest + trust.
        assert cfg.layers == full.layers - {layer}
        assert cfg.trust_active == full.trust_active


def test_leave_one_out_isolates_the_defending_gate():
    # A modest id-sorted subset that spans B1 and B3 (the first classes).
    result = run_leave_one_out(max_instances=45)
    assert [c.config_name for c in result.configs][0] == "full PALISADE"
    assert len(result.configs) == 7
    ref = result.configs[0]
    minus_g1 = next(c for c in result.configs if c.config_name == "full -G1")
    b1_ref = ref.family("B1")
    b1_no_g1 = minus_g1.family("B1")
    # Removing G1 cannot lower the B1 soft-win ASR (it can only re-admit).
    assert b1_ref is not None and b1_no_g1 is not None
    assert (b1_no_g1.soft_asr or 0.0) >= (b1_ref.soft_asr or 0.0)


def test_markdown_has_both_grids_and_the_g6_row():
    md = run_leave_one_out(max_instances=30).to_markdown()
    assert "Hard-win rate" in md and "Soft-win ASR" in md
    assert "full -G6" in md
    assert "| Configuration | B1 | B3 | B4 | B5 | XC |" in md
