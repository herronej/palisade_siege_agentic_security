"""
R5-C2: benign-FPR operating curve.

Locks the money result and its two internal-consistency anchors: the untuned
operating point reproduces the manuscript's 8/181 FPR and 13/205 deployed
hard-win, the curve is monotone-decreasing in FPR as signatures drop, the
single-family attribution matches benign_fpr.md's per-signature counts, the two
posture families are carried by the config rather than by patching the gate
classes, and the one remaining patched signature is restored on exit.
"""

from __future__ import annotations

import asyncio

from palisade.gates import g1_prompt, g3_rag, g5_hpc
from palisade.capabilities import g3_rag as cap_g3
from tools.fpr_curve import (
    SIGNATURE_ORDER,
    SIGNATURES,
    config_for,
    disabled_signatures,
    run_fpr_curve,
)


def test_disabled_signatures_restores_the_patched_family() -> None:
    """``db_authority`` is the only family still disabled by patching, and it is
    patched at both binding sites and restored on exit."""
    orig_gate = g3_rag.match_db_authority_spoof
    orig_cap = cap_g3.match_db_authority_spoof
    with disabled_signatures(frozenset(SIGNATURE_ORDER)):
        assert g3_rag.match_db_authority_spoof is not orig_gate
        assert cap_g3.match_db_authority_spoof is not orig_cap  # both binding sites
    assert g3_rag.match_db_authority_spoof is orig_gate
    assert cap_g3.match_db_authority_spoof is orig_cap


def test_disabled_signatures_leaves_the_gate_methods_alone() -> None:
    """The lifecycle and weaponization families are configuration now, so the
    curve must not reach into the gate classes to disable them. Patching them
    made the posture unselectable outside this tool -- which is why the
    recommended configuration could not be searched."""
    orig_life = g5_hpc.G5HpcJobGate._check_lifecycle_hooks
    orig_weap = g1_prompt.G1PromptGate._match_weaponization
    with disabled_signatures(frozenset(SIGNATURE_ORDER)):
        assert g5_hpc.G5HpcJobGate._check_lifecycle_hooks is orig_life
        assert g1_prompt.G1PromptGate._match_weaponization is orig_weap


def test_config_for_carries_the_two_posture_families() -> None:
    assert config_for(frozenset()).g5_lifecycle_hook is True
    assert config_for(frozenset()).g1_weaponization_signature is True

    off = config_for(frozenset({"lifecycle", "weaponization"}))
    assert off.g5_lifecycle_hook is False
    assert off.g1_weaponization_signature is False

    # db_authority has no config field, so it leaves both posture fields on.
    db_only = config_for(frozenset({"db_authority"}))
    assert db_only.g5_lifecycle_hook is True
    assert db_only.g1_weaponization_signature is True


def test_config_for_rejects_unknown() -> None:
    try:
        config_for(frozenset({"nope"}))
    except ValueError as e:
        assert "nope" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on unknown signature")


def test_disabled_signatures_rejects_unknown() -> None:
    try:
        with disabled_signatures(frozenset({"nope"})):
            pass
    except ValueError as e:
        assert "nope" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on unknown signature")


def test_curve_reproduces_operating_point_and_is_free() -> None:
    result = asyncio.run(run_fpr_curve())
    base = result.cumulative[0]
    full_off = result.cumulative[-1]

    # Untuned operating point reproduces the manuscript's two anchors.
    assert (base.fpr_k, base.fpr_n) == (8, 181)
    # 13 not 16: the closed-vocabulary scheduler-field rule (capabilities.scheduler_fields) closes the three b5_11 instances whose injected value falls below the distinctiveness floor, at 0 benign cost.
    assert (base.hw_k, base.hw_n) == (13, 205)

    # Fully tuned reaches 0% benign FPR.
    assert full_off.fpr_k == 0

    # Deployed hard-win is flat across the whole frontier (the money result).
    assert {p.hw_k for p in result.cumulative} == {13}

    # FPR is monotone non-increasing as more signatures drop.
    fprs = [p.fpr_k for p in result.cumulative]
    assert fprs == sorted(fprs, reverse=True)
    assert fprs == [8, 3, 1, 0]  # 8 -> drop lifecycle(5) -> +weap(2) -> +db(1)


def test_single_family_attribution_matches_counts() -> None:
    """Dropping one family should reduce the 8 false blocks by exactly that
    family's benign_fpr.md count, at no hard-win cost."""
    result = asyncio.run(run_fpr_curve())
    by_sig = {p.disabled[0]: p for p in result.singles}
    for sig, (_label, count) in SIGNATURES.items():
        p = by_sig[sig]
        assert p.fpr_k == 8 - count, f"{sig}: expected {8 - count}, got {p.fpr_k}"
        assert p.hw_k == 13  # no hard-win cost
