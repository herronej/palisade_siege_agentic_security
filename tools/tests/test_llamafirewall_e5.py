"""E5: the soft/hard construction on a stub detector, no model required.

The real comparator needs the gated Prompt-Guard-2 weights; these pin the
scoring logic so a wiring mistake fails here, not on a 5-minute model run.
"""

from __future__ import annotations

import pytest

from siege.redteam.baselines.detectors import DetectorVerdict
from tools.llamafirewall_e5 import run_e5


class _Stub:
    """A detector whose flagging is fully controlled."""

    def __init__(self, predicate, *, name="stub", available=True, scanners=("STUB",)):
        self.name = name
        self.scanners = scanners
        self._predicate = predicate
        self._available = available
        self.calls = 0

    def available(self) -> bool:
        return self._available

    def flag(self, text: str) -> DetectorVerdict:
        self.calls += 1
        return DetectorVerdict(bool(self._predicate(text)), "stub", self.name)


def test_flagging_nothing_leaves_every_undefended_win_standing():
    r = run_e5(_Stub(lambda t: False))
    assert r.soft.llamafirewall == r.soft.undefended
    assert r.hard.llamafirewall == r.hard.undefended
    assert r.benign_blocked == 0
    assert r.soft.catch_on_wins == 0.0 and r.hard.catch_on_wins == 0.0
    # Sanity: the undefended control actually has wins to leave standing.
    assert r.soft.undefended > 0 and r.hard.undefended > 0


def test_flagging_everything_closes_both_columns_and_blocks_every_benign():
    r = run_e5(_Stub(lambda t: bool(t)))
    assert r.soft.llamafirewall == 0
    assert r.hard.llamafirewall == 0
    assert r.benign_blocked == r.benign_n
    assert r.soft.catch_on_wins == 1.0 and r.hard.catch_on_wins == 1.0


def test_bases_are_the_paper_denominators():
    r = run_e5(_Stub(lambda t: False))
    assert r.soft.n == 150
    assert r.hard.n == 32
    assert r.benign_n == 181


def test_palisade_full_is_below_undefended_on_both_bases():
    """The side-by-side is real: full closes wins the undefended control has."""
    r = run_e5(_Stub(lambda t: False))
    assert r.soft.palisade_full < r.soft.undefended
    assert r.hard.palisade_full < r.hard.undefended


def test_an_unavailable_comparator_is_refused_not_scored_empty():
    with pytest.raises(RuntimeError, match="unavailable"):
        run_e5(_Stub(lambda t: False, available=False))


def test_identical_texts_are_scanned_once():
    stub = _Stub(lambda t: "id_rsa" in t)
    r = run_e5(stub)
    # The cache scans each distinct string once; the corpus repeats many, so the
    # inner detector sees fewer calls than the report's scan count would be
    # without dedup.
    assert stub.calls == r.scans
    assert r.scans > 0


def test_report_carries_the_banner_and_both_bases():
    md = run_e5(_Stub(lambda t: False)).to_markdown()
    assert "Source module: tools.llamafirewall_e5" in md[:200]
    assert "security soft wins" in md
    assert "sink-capable hard wins" in md
