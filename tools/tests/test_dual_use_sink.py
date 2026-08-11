"""W5.2 dual-use sink demonstrator: the axis does work the taint bound does not.

Runs the ``dual_use_sink_demo`` instances through the real SessionRunner
(undefended vs full) and pins that every full-block is the dual-use axis, none
the taint bound, and every case was an undefended hard win -- i.e. the axis
blocks values the taint bound admits (R1-M6 independent-work test).
"""

from __future__ import annotations

from tools.dual_use_sink import run_dual_use_sink


def test_dual_use_axis_does_independent_work():
    result = run_dual_use_sink()
    assert result.n >= 5
    # Every case: undefended hard win, full-blocked by the dual-use axis, and the
    # taint bound never made the block (trusted source -> taint carve-out).
    assert result.n_undef_hard_win == result.n
    assert result.n_blocked_by_dual_use == result.n
    assert result.n_blocked_by_taint == 0
    assert result.independent is True


def test_block_reason_names_the_dual_use_axis():
    result = run_dual_use_sink()
    reasons = [r.full_reason for r in result.rows]
    assert all("dual-use flow" in r for r in reasons)
    assert all("capability flow" not in r for r in reasons)  # not the taint bound
