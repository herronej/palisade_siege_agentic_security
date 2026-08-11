"""
W2.3 -- taint-propagation coverage (L̂ vs L*) regression guard.

Pins the two load-bearing invariants of the coverage study so a change to the
propagation primitive can't silently regress them:

* **precision:** zero false propagations (no trusted value marked tainted) --
  the guarantee that the containment hardening leaves benign work untouched;
* **recall shape:** containment recognizes exactly the textual-flow laundering
  (xc_4 3/5, and the corpus-injected-code / injected-field families), and misses
  the fresh-authoring / tag-drop families (xc_1, b1_9, xc_5) -- the characterized
  residual.
"""

from __future__ import annotations

import collections

from siege import load_instances
from siege.corpus_builder import CORPUS_DIR
from tools.taint_coverage import collect_rows

_ROWS = collect_rows(load_instances(CORPUS_DIR))
_UNTRUSTED = [r for r in _ROWS if r.l_star]


def test_no_false_propagations():
    # A trusted value (L*=untainted) must never be marked tainted by L̂.
    false_props = [r for r in _ROWS if (not r.l_star) and r.l_hat]
    assert false_props == [], f"false propagations: {[r.instance_id for r in false_props]}"


def test_containment_recognizes_textual_flow_only():
    recognized = [r for r in _UNTRUSTED if r.l_hat]
    gaps = [r for r in _UNTRUSTED if r.gap]
    # Some untrusted values are recognized (containment works) and the residual
    # is the majority (tag-drop dominates) -- both non-trivial.
    assert recognized, "containment recognized nothing -- propagation is dead"
    assert gaps, "no gaps -- the residual should not be empty"
    assert len(recognized) + len(gaps) == len(_UNTRUSTED)

    gap_by_tmpl = collections.Counter(r.template for r in gaps)
    # Pure tag-drop / fresh-authoring families carry no textual flow: all gap.
    assert gap_by_tmpl["xc_1_cross_boundary_chain"] == 8
    # xc_4 splits 3 caught / 2 residual (hex_decode, concat_import).
    assert gap_by_tmpl["xc_4_taint_laundering"] == 2
