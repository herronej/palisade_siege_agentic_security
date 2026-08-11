"""
Unit tests for the Bayesian per-capability trust scorer.

These pin the acceptance criteria of the "Implement Bayesian
per-capability trust scoring" issue:

  * per-capability prior/posterior tracking;
  * `TrustScorer.current_tier_for(capability_kind)` returns a tier per
    capability;
  * clean G2 calls restore the G2 tier but not the G3 tier (the
    posteriors are independent);
  * a probe-then-clean oscillation attacker cannot restore a
    high-stakes capability's tier (the sticky floor holds).
"""

from __future__ import annotations

from palisade.capabilities.registry import TrustTier
from palisade.config import PalisadeSettings
from palisade.trust import TierPolicy, TrustScorer


def _scorer(**overrides) -> TrustScorer:
    return TrustScorer(PalisadeSettings(enabled=True, **overrides))


# -----------------------------------------------------------------
# TierPolicy
# -----------------------------------------------------------------


def test_tier_policy_thresholds() -> None:
    policy = TierPolicy(PalisadeSettings(enabled=True))
    assert policy.tier_for_score(1.0) is TrustTier.NORMAL
    assert policy.tier_for_score(0.7) is TrustTier.NORMAL
    assert policy.tier_for_score(0.69) is TrustTier.ELEVATED
    assert policy.tier_for_score(0.4) is TrustTier.ELEVATED
    assert policy.tier_for_score(0.39) is TrustTier.RESTRICTED
    assert policy.tier_for_score(0.1) is TrustTier.RESTRICTED
    assert policy.tier_for_score(0.05) is TrustTier.TERMINATED


def test_out_of_range_score_is_clamped() -> None:
    policy = TierPolicy(PalisadeSettings(enabled=True))
    assert policy.tier_for_score(-1.0) is TrustTier.TERMINATED
    assert policy.tier_for_score(2.0) is TrustTier.NORMAL


# -----------------------------------------------------------------
# Idle / default-compatible behavior
# -----------------------------------------------------------------


def test_idle_session_is_normal() -> None:
    scorer = _scorer()
    assert scorer.current_score == 1.0
    assert scorer.current_tier() is TrustTier.NORMAL
    assert scorer.current_tier_for("G2") is TrustTier.NORMAL
    assert scorer.capability_scores() == {}


def test_recording_is_a_noop_when_master_flag_off() -> None:
    scorer = TrustScorer(PalisadeSettings(enabled=False))
    scorer.record_violation(1, capability_kind="G2", high_stakes=True)
    scorer.record_clean_call("G2")
    scorer.notify_incident(level=1, gate="G2")
    assert scorer.current_tier() is TrustTier.NORMAL
    assert scorer.current_score == 1.0
    assert scorer.capability_scores() == {}


# -----------------------------------------------------------------
# Per-capability prior/posterior tracking
# -----------------------------------------------------------------


def test_per_capability_posteriors_are_independent() -> None:
    """A violation on one capability does not move another's score."""
    scorer = _scorer()
    scorer.record_violation(1, capability_kind="G3", high_stakes=False)

    scores = scorer.capability_scores()
    assert scores["G3"] < 1.0
    # G2 has no posterior yet, so it still reports the initial tier.
    assert "G2" not in scores
    assert scorer.current_tier_for("G2") is TrustTier.NORMAL
    assert scorer.current_tier_for("G3") is not TrustTier.NORMAL


def test_clean_call_raises_capability_score() -> None:
    scorer = _scorer()
    scorer.record_violation(1, capability_kind="G2", high_stakes=False)
    degraded = scorer.capability_scores()["G2"]
    for _ in range(20):
        scorer.record_clean_call("G2")
    assert scorer.capability_scores()["G2"] > degraded


def test_invalid_severity_raises() -> None:
    scorer = _scorer()
    for bad in (0, 4, -1):
        try:
            scorer.record_violation(bad, capability_kind="G2")
        except ValueError:
            pass
        else:  # pragma: no cover - the assert is the failure path
            raise AssertionError(f"severity {bad!r} should have raised")


# -----------------------------------------------------------------
# AC: clean G2 calls restore G2-tier but not G3-tier
# -----------------------------------------------------------------


def test_clean_g2_calls_restore_g2_tier_but_not_g3() -> None:
    scorer = _scorer()

    # Degrade both G2 and G3 below NORMAL with a (non-high-stakes)
    # violation each, so neither is held by a sticky floor.
    scorer.record_violation(1, capability_kind="G2", high_stakes=False)
    scorer.record_violation(1, capability_kind="G3", high_stakes=False)
    assert scorer.current_tier_for("G2") is not TrustTier.NORMAL
    assert scorer.current_tier_for("G3") is not TrustTier.NORMAL

    # A stream of clean G2 calls only touches G2's posterior.
    for _ in range(100):
        scorer.record_clean_call("G2")

    assert scorer.current_tier_for("G2") is TrustTier.NORMAL
    # G3 was never touched by the clean G2 calls -- it stays degraded.
    assert scorer.current_tier_for("G3") is not TrustTier.NORMAL
    # And the session tier reflects the worst capability (still G3).
    assert scorer.current_tier() is scorer.current_tier_for("G3")


# -----------------------------------------------------------------
# AC: oscillation attack cannot restore a high-stakes capability
# -----------------------------------------------------------------


def test_oscillation_attack_cannot_restore_high_stakes_tier() -> None:
    """probe-then-clean: a high-stakes violation pins a sticky floor
    that no number of clean calls can lift."""
    scorer = _scorer()  # sticky_high_stakes defaults to True

    scorer.record_violation(1, capability_kind="G5", high_stakes=True)
    floor = scorer.current_tier_for("G5")
    assert floor is not TrustTier.NORMAL

    # The attacker runs many clean calls trying to wash out the breach.
    for _ in range(500):
        scorer.record_clean_call("G5")

    # The posterior mean has recovered ...
    assert scorer.capability_scores()["G5"] > 0.9
    # ... but the sticky floor holds the tier down.
    assert scorer.current_tier_for("G5") is floor
    assert scorer.current_tier_for("G5") is not TrustTier.NORMAL


def test_sticky_floor_only_ratchets_worse() -> None:
    """A second, more severe high-stakes violation worsens the floor;
    it never relaxes it."""
    scorer = _scorer()
    scorer.record_violation(2, capability_kind="G5", high_stakes=True)
    first = scorer.current_tier_for("G5")
    scorer.record_violation(1, capability_kind="G5", high_stakes=True)
    second = scorer.current_tier_for("G5")
    # SEV1 adds more failure mass than SEV2, so the floor can only get
    # worse (or stay the same), never better.
    assert second.value != "normal"
    from palisade.trust import _TIER_RANK  # type: ignore[attr-defined]

    assert _TIER_RANK[second] >= _TIER_RANK[first]


def test_high_stakes_recovers_when_sticky_disabled() -> None:
    """With sticky_high_stakes=False (benchmark mode) a high-stakes
    capability recovers normally once clean calls accumulate."""
    scorer = _scorer(sticky_high_stakes=False)
    scorer.record_violation(1, capability_kind="G5", high_stakes=True)
    assert scorer.current_tier_for("G5") is not TrustTier.NORMAL
    for _ in range(200):
        scorer.record_clean_call("G5")
    assert scorer.current_tier_for("G5") is TrustTier.NORMAL


# -----------------------------------------------------------------
# Incident wiring + session tier
# -----------------------------------------------------------------


def test_notify_incident_routes_scoring_to_the_named_gate() -> None:
    # SEV3 has no playbook action, so it isolates the scoring effect:
    # only G3's posterior moves, G2 is untouched.
    scorer = _scorer()
    scorer.notify_incident(level=3, gate="G3")
    assert scorer.capability_scores()["G3"] < 1.0
    assert "G2" not in scorer.capability_scores()
    assert scorer.current_tier_for("G2") is TrustTier.NORMAL


def test_session_tier_is_worst_capability() -> None:
    scorer = _scorer()
    scorer.record_violation(3, capability_kind="G2", high_stakes=False)
    scorer.record_violation(1, capability_kind="G5", high_stakes=True)
    g5 = scorer.current_tier_for("G5")
    from palisade.trust import _TIER_RANK  # type: ignore[attr-defined]

    assert scorer.current_tier() is g5
    assert _TIER_RANK[scorer.current_tier()] >= _TIER_RANK[
        scorer.current_tier_for("G2")
    ]


def test_reset_clears_everything() -> None:
    scorer = _scorer()
    scorer.record_violation(1, capability_kind="G5", high_stakes=True)
    scorer.mark_sticky_denied("g5:above_ceiling_resource")
    scorer.reset()
    assert scorer.capability_scores() == {}
    assert scorer.sticky_denied == ()
    assert scorer.current_tier() is TrustTier.NORMAL
    assert scorer.current_tier_for("G5") is TrustTier.NORMAL
    assert scorer.history == ()


def test_snapshot_shape() -> None:
    scorer = _scorer()
    scorer.record_violation(1, capability_kind="G5", high_stakes=True)
    scorer.record_clean_call("G2")
    snap = scorer.snapshot()
    assert set(snap) == {
        "score", "tier", "terminated", "reauth_required",
        "violations", "sticky_denied", "contract_coverage", "capabilities",
    }
    assert snap["violations"] == 1
    assert snap["capabilities"]["G5"]["sticky"] is True
    assert snap["capabilities"]["G5"]["floor"] is not None
    assert snap["capabilities"]["G2"]["sticky"] is False
