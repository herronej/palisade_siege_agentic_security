"""
W23.5 -- utility / DoS dashboard tests.

Offline / deterministic (the tool drives the fast-tier full config;
no Q-LLM or live endpoint required).
"""

from __future__ import annotations

import asyncio

from tools.utility_dos import (
    AttackUtilityReport,
    BenignUtilityReport,
    UtilityDosResult,
    _measure_attack,
    _measure_benign,
    render,
)


def test_benign_population_matches_181():
    report = asyncio.run(_measure_benign())
    assert report.total == 181
    assert report.completed + report.blocked_retryable + report.blocked_lockout == report.total


def test_benign_lockout_is_the_3_9_pct():
    report = asyncio.run(_measure_benign())
    assert report.blocked_lockout == 7
    assert abs(report.lockout_rate - 7 / 181) < 1e-6


def test_attack_population_matches_205():
    report = asyncio.run(_measure_attack())
    assert report.total == 205
    assert report.lockout_induced <= report.total
    assert report.lockout_with_utility_loss <= report.lockout_induced


def test_render_produces_markdown_with_both_sections():
    benign = asyncio.run(_measure_benign())
    attack = asyncio.run(_measure_attack())
    result = UtilityDosResult(benign=benign, attack=attack)
    md = render(result)
    assert "## Benign side" in md
    assert "## Attack side" in md
    assert "## Synthesis" in md
    assert "3.9%" in md or "lockout" in md.lower()
