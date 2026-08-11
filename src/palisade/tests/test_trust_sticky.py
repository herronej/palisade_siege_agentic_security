"""
Unit tests for sticky high-stakes capability lock-in.

Acceptance criteria pinned here:

  * sticky-tier policy is applied when ``sticky_high_stakes=true`` (the
    default): a high-stakes denial persists across 100 clean tool calls;
  * an explicit re-auth event clears the stickiness and restores the
    locked capability;
  * with ``sticky_high_stakes=false`` (benchmark mode) the tier recovers
    normally.

The lock-in itself lives in `TrustScorer`: a high-stakes violation pins
a per-capability floor tier that clean calls cannot lift, and
`reauthenticate()` is the only in-process way to lift it. The re-auth
*API endpoint* is a separate issue; these tests exercise the scorer
method it will call.
"""

from __future__ import annotations

from palisade.capabilities.registry import TrustTier
from palisade.config import PalisadeSettings
from palisade.trust import _TIER_RANK, TrustScorer


def _scorer(**overrides) -> TrustScorer:
    return TrustScorer(PalisadeSettings(enabled=True, **overrides))


# -----------------------------------------------------------------
# AC: high-stakes denial persists across 100 clean tool calls
# -----------------------------------------------------------------


def test_high_stakes_denial_persists_across_100_clean_calls() -> None:
    scorer = _scorer()  # sticky_high_stakes defaults to True
    scorer.record_violation(1, capability_kind="G5", high_stakes=True)
    locked = scorer.current_tier_for("G5")
    assert locked is not TrustTier.NORMAL

    for _ in range(100):
        scorer.record_clean_call("G5")

    # The posterior mean has fully recovered ...
    assert scorer.capability_scores()["G5"] > 0.9
    # ... but the sticky floor holds the tier at the locked level.
    assert scorer.current_tier_for("G5") is locked
    assert scorer.current_tier() is locked


def test_clean_calls_on_other_capabilities_do_not_lift_the_lock() -> None:
    """Probing a *different* boundary can't recover the locked one."""
    scorer = _scorer()
    scorer.record_violation(1, capability_kind="G5", high_stakes=True)
    locked = scorer.current_tier_for("G5")
    for _ in range(100):
        scorer.record_clean_call("G2")
    assert scorer.current_tier_for("G5") is locked


# -----------------------------------------------------------------
# AC: re-auth clears stickiness
# -----------------------------------------------------------------


def test_reauthenticate_clears_sticky_floor_and_restores_capability() -> None:
    scorer = _scorer()
    scorer.record_violation(1, capability_kind="G5", high_stakes=True)
    assert scorer.current_tier_for("G5") is not TrustTier.NORMAL

    unlocked = scorer.reauthenticate()

    assert "G5" in unlocked
    assert scorer.current_tier_for("G5") is TrustTier.NORMAL
    assert scorer.current_tier() is TrustTier.NORMAL
    # The snapshot no longer reports the capability as sticky/floored.
    cap = scorer.snapshot()["capabilities"]["G5"]
    assert cap["sticky"] is False
    assert cap["floor"] is None


def test_reauthenticate_clears_sticky_denied_registry() -> None:
    scorer = _scorer()
    scorer.mark_sticky_denied("g5:above_ceiling_resource")
    scorer.mark_sticky_denied("g5:credential_exfil")
    assert scorer.sticky_denied == (
        "g5:above_ceiling_resource",
        "g5:credential_exfil",
    )

    scorer.reauthenticate()

    assert scorer.sticky_denied == ()


def test_reauthenticate_preserves_history_and_non_sticky_capabilities() -> None:
    """Re-auth lifts the lock but keeps the audit trail and does not
    touch capabilities that were degraded without a high-stakes lock."""
    scorer = _scorer()
    scorer.record_violation(1, capability_kind="G5", high_stakes=True)
    scorer.record_violation(2, capability_kind="G2", high_stakes=False)
    g2_before = scorer.capability_scores()["G2"]

    scorer.reauthenticate()

    # History (audit) survives the re-auth.
    assert len(scorer.history) == 2
    # G2 was never sticky, so re-auth leaves its posterior alone.
    assert scorer.capability_scores()["G2"] == g2_before
    # G5 was restored to its prior.
    assert scorer.current_tier_for("G5") is TrustTier.NORMAL


def test_reauthenticate_is_noop_without_stickiness() -> None:
    scorer = _scorer()
    scorer.record_violation(2, capability_kind="G2", high_stakes=False)
    before = scorer.capability_scores()["G2"]
    assert scorer.reauthenticate() == ()
    assert scorer.capability_scores()["G2"] == before


def test_reauthenticate_is_noop_when_master_flag_off() -> None:
    scorer = TrustScorer(PalisadeSettings(enabled=False))
    assert scorer.reauthenticate() == ()


# -----------------------------------------------------------------
# AC: sticky_high_stakes=false recovers normally (benchmark mode)
# -----------------------------------------------------------------


def test_tier_recovers_normally_when_sticky_disabled() -> None:
    scorer = _scorer(sticky_high_stakes=False)
    scorer.record_violation(1, capability_kind="G5", high_stakes=True)
    assert scorer.current_tier_for("G5") is not TrustTier.NORMAL

    for _ in range(100):
        scorer.record_clean_call("G5")

    # No floor was ever set, so the recovered posterior lifts the tier.
    assert scorer.current_tier_for("G5") is TrustTier.NORMAL
    # And there is nothing for re-auth to unlock.
    assert scorer.reauthenticate() == ()


def test_sticky_floor_only_worsens_then_reauth_fully_clears() -> None:
    scorer = _scorer()
    scorer.record_violation(2, capability_kind="G5", high_stakes=True)
    first = scorer.current_tier_for("G5")
    scorer.record_violation(1, capability_kind="G5", high_stakes=True)
    second = scorer.current_tier_for("G5")
    assert _TIER_RANK[second] >= _TIER_RANK[first]

    scorer.reauthenticate()
    assert scorer.current_tier_for("G5") is TrustTier.NORMAL
