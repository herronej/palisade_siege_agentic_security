"""
W23.5 -- utility / DoS dashboard tests.

Offline / deterministic (the tool drives the fast-tier full config;
no Q-LLM or live endpoint required).

The attack side models session termination, because the replay harness does
not perform it. These tests pin that model: a SEV1 cuts the rest of its
session, a later session is untouched, and the collateral the dashboard
reports is the session ending, not a gate denial.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from tools.utility_dos import (
    AttackUtilityReport,
    BenignUtilityReport,
    UtilityDosResult,
    _measure_attack,
    _measure_benign,
    cut_by_termination,
    is_terminating,
    render,
)


@pytest.fixture(scope="module")
def benign() -> BenignUtilityReport:
    return asyncio.run(_measure_benign())


@pytest.fixture(scope="module")
def attack() -> AttackUtilityReport:
    return asyncio.run(_measure_attack())


# -----------------------------------------------------------------
# The termination model, on synthetic traces
# -----------------------------------------------------------------


@dataclass(frozen=True)
class _Rec:
    session_id: str
    allowed: bool
    incident_level: int | None = None
    name: str = ""


@dataclass(frozen=True)
class _Trace:
    actions: tuple


def test_is_terminating_is_a_sev1_denial_only() -> None:
    assert is_terminating(_Rec("s", allowed=False, incident_level=1))
    assert not is_terminating(_Rec("s", allowed=False, incident_level=2))
    assert not is_terminating(_Rec("s", allowed=False, incident_level=None))
    # An allowed action is never a termination, whatever it carries.
    assert not is_terminating(_Rec("s", allowed=True, incident_level=1))


def test_cut_takes_everything_after_the_first_sev1_in_its_session() -> None:
    a = _Rec("s1", True, name="a")
    sev1 = _Rec("s1", False, 1, name="sev1")
    b = _Rec("s1", True, name="b")
    c = _Rec("s1", False, 2, name="c")
    trace = _Trace((a, sev1, b, c))
    assert [r.name for r in cut_by_termination(trace)] == ["b", "c"]


def test_cut_does_not_reach_across_sessions() -> None:
    """The runner rebuilds the trust scorer per session, so a termination in
    one session says nothing about the next."""
    sev1 = _Rec("s1", False, 1, name="sev1")
    later_same = _Rec("s1", True, name="later_same")
    other = _Rec("s2", True, name="other")
    trace = _Trace((sev1, other, later_same))
    assert [r.name for r in cut_by_termination(trace)] == ["later_same"]


def test_cut_is_empty_without_a_sev1() -> None:
    trace = _Trace((_Rec("s1", False, 2), _Rec("s1", True)))
    assert cut_by_termination(trace) == []


def test_cut_accepts_a_narrower_terminating_predicate() -> None:
    """sev2_routing keys the cut on lifecycle-hook SEV1s only."""
    first = _Rec("s1", False, 1, name="other_sev1")
    target = _Rec("s1", False, 1, name="target")
    after = _Rec("s1", True, name="after")
    trace = _Trace((first, target, after))
    cut = cut_by_termination(trace, lambda r: r.name == "target")
    assert [r.name for r in cut] == ["after"]


# -----------------------------------------------------------------
# Benign side
# -----------------------------------------------------------------


def test_benign_population_matches_181(benign) -> None:
    assert benign.total == 181
    assert (
        benign.completed + benign.blocked_retryable + benign.blocked_lockout
        == benign.total
    )


def test_benign_lockout_is_the_3_9_pct(benign) -> None:
    assert benign.blocked_lockout == 7
    assert abs(benign.lockout_rate - 7 / 181) < 1e-6


# -----------------------------------------------------------------
# Attack side
# -----------------------------------------------------------------


def test_attack_population_matches_205(attack) -> None:
    assert attack.total == 205
    assert attack.lockout_induced <= attack.total
    assert attack.lockout_with_utility_loss <= attack.lockout_induced


def test_modeling_termination_does_not_move_attack_blocking(attack) -> None:
    """No admitted attack action follows a SEV1 anywhere in the corpus, so the
    termination model changes only the collateral rows."""
    assert attack.attack_blocked == 126
    assert attack.lockout_induced == 13


def test_collateral_is_the_session_ending_not_a_gate_denial(attack) -> None:
    """The corrected finding. The gates allow every legitimate task on its
    merits -- zero denials -- yet every SEV1-triggering attack session loses
    its task, because the task is ordered after the denial."""
    assert attack.utility_lost_to_denial == 0
    assert attack.utility_lost_to_termination == 13
    assert attack.utility_lost == 13
    assert attack.utility_preserved == 205 - 13
    assert attack.lockout_with_utility_loss == 13
    assert attack.dos_amplification == 1.0


def test_lost_utility_partitions_by_route(attack) -> None:
    assert (
        attack.utility_lost_to_denial + attack.utility_lost_to_termination
        == attack.utility_lost
    )
    assert attack.utility_preserved + attack.utility_lost == attack.total


def test_attack_lockout_signatures_account_for_every_lockout(attack) -> None:
    assert sum(attack.lockout_signatures.values()) == attack.lockout_induced
    # The lifecycle-hook check produces two of these rows; SEV2 routing of that
    # check is what sev2_routing measures.
    lifecycle = attack.lockout_signatures.get(
        "G5 lifecycle-hook injection", 0
    ) + attack.lockout_signatures.get("G5 directive command substitution", 0)
    assert lifecycle == 5


# -----------------------------------------------------------------
# Report
# -----------------------------------------------------------------


def test_render_produces_markdown_with_both_sections(benign, attack) -> None:
    md = render(UtilityDosResult(benign=benign, attack=attack))
    assert "## Benign side" in md
    assert "## Attack side" in md
    assert "## Synthesis" in md
    assert "3.9%" in md


def test_render_carries_its_provenance_banner(benign, attack) -> None:
    """The shipped report had a banner the tool never emitted, so regenerating
    it silently failed the result-map check."""
    md = render(UtilityDosResult(benign=benign, attack=attack))
    assert md.startswith(
        "<!-- Generated report. Source module: tools.utility_dos."
    )


def test_render_no_longer_calls_the_primitive_harmless(benign, attack) -> None:
    md = render(UtilityDosResult(benign=benign, attack=attack))
    assert "not a denial-of-service *vulnerability*" not in md
    assert "13/13 = 100%" in md
    # SEV1 termination is final; the report must not promise recovery.
    assert "until re-authentication" not in md
