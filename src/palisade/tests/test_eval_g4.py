"""
Unit tests for the G4 evaluation harness.

Mirror the structure of ``test_eval_g2.py`` and
``test_eval_g1.py``: determinism, shape checks,
baseline-allows-everything, each defense catches its target,
FPR bounds.
"""

from __future__ import annotations

import random

import pytest

# anyio's pytest plugin requires async tests to be marked.
pytestmark = pytest.mark.anyio

from siege.eval import (
    DEFAULT_G4_CONFIGS,
    G4_ATTACK_GENERATORS,
    G4_DISPLAY_NAMES,
    G4AttackScenario,
    G4EvalConfig,
    G4EvaluationResult,
    format_g4_report,
    generate_g4_attacks,
    generate_g4_benign_workload,
    run_g4_evaluation,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# -----------------------------------------------------------------
# Scenario generation: determinism + shape
# -----------------------------------------------------------------


def test_attack_generators_registry_has_expected_classes() -> None:
    """The nine G4 code-boundary attack classes the corpus authors against."""
    assert set(G4_ATTACK_GENERATORS.keys()) == {
        "reverse_shell",
        "eval_exec",
        "pickle_rce",
        "credential_file_read",
        "typo_squat_import",
        "env_var_hijack",
        "direct_network_exfil",
        "cryptomining",
        "network_dos_disguised",
    }


def test_display_names_cover_every_attack_class() -> None:
    for class_name in G4_ATTACK_GENERATORS:
        assert class_name in G4_DISPLAY_NAMES
        assert G4_DISPLAY_NAMES[class_name]


def test_generate_attacks_is_deterministic_under_fixed_rng() -> None:
    a = generate_g4_attacks("reverse_shell", random.Random(7), n=5)
    b = generate_g4_attacks("reverse_shell", random.Random(7), n=5)
    assert [s.args for s in a] == [s.args for s in b]


def test_generate_attacks_produces_n_scenarios() -> None:
    for class_name in G4_ATTACK_GENERATORS:
        scenarios = generate_g4_attacks(class_name, random.Random(0), n=4)
        assert len(scenarios) == 4
        for s in scenarios:
            assert s.attack_class == class_name
            assert s.tool_name in {"run_bash", "create_file"}
            assert isinstance(s.args, dict)


def test_every_attack_carries_at_least_one_finding() -> None:
    """The attacks model semgrep firing; the harness's mock
    returns these as-is. An empty findings tuple would mean the
    scenario isn't testing a defense path."""
    for class_name in G4_ATTACK_GENERATORS:
        for scenario in generate_g4_attacks(class_name, random.Random(0), n=4):
            assert len(scenario.findings) >= 1, (
                f"scenario in {class_name} has no findings"
            )
            check_id, severity, _ = scenario.findings[0]
            assert check_id.startswith("vista-"), (
                f"finding {check_id} not in the vista-* namespace"
            )
            assert severity in {"ERROR", "WARNING", "INFO"}


def test_direct_network_exfil_is_warning_only() -> None:
    """The work item documents direct_network_exfil as the one
    WARNING-class scenario (slow-tier promotes it to deny)."""
    scenarios = generate_g4_attacks(
        "direct_network_exfil", random.Random(0), n=4,
    )
    for s in scenarios:
        assert s.findings[0][1] == "WARNING"


def test_generate_attacks_rejects_unknown_class() -> None:
    with pytest.raises(KeyError):
        generate_g4_attacks("nope", random.Random(0), n=1)


def test_benign_workload_is_deterministic() -> None:
    a = generate_g4_benign_workload(random.Random(11), n=10)
    b = generate_g4_benign_workload(random.Random(11), n=10)
    assert [s.args for s in a] == [s.args for s in b]


def test_benign_workload_carries_no_findings() -> None:
    """Benign scenarios model "Semgrep saw nothing" -- findings
    is empty by construction."""
    for s in generate_g4_benign_workload(random.Random(0), n=20):
        assert s.findings == ()
        assert s.attack_class in {"benign_python", "benign_bash"}


# -----------------------------------------------------------------
# Runner: per-class ASR + FPR
# -----------------------------------------------------------------


async def test_baseline_lets_every_attack_succeed() -> None:
    """Baseline (G4 disabled) is the control row. Every attack
    class should report ASR=1.0."""
    result = await run_g4_evaluation(
        rng_seed=42, n_attacks_per_class=5, n_benign=10,
    )
    baseline_cells = [
        c for c in result.asr_cells if c.config_name == "baseline"
    ]
    for cell in baseline_cells:
        assert cell.asr == 1.0


async def test_semgrep_off_is_noop_passthrough() -> None:
    """semgrep_enabled=False -> the Semgrep tier is a no-op, so ASR matches
    baseline -- EXCEPT for the classes the always-on Tier-0 deterministic
    floor covers without Semgrep. ``typo_squat_import`` is now one of those
    (the Tier-0 denylist + near-miss check, sharing one source of truth with
    the ``typo_squat.yml`` rule), so it denies at Tier-0 even with Semgrep off;
    the remaining classes still pass through to a (disabled) Semgrep."""
    result = await run_g4_evaluation(
        rng_seed=42, n_attacks_per_class=5, n_benign=10,
    )
    off = {
        c.attack_class: c.asr
        for c in result.asr_cells
        if c.config_name == "G4 fast-only (semgrep off)"
    }
    # Always-on Tier-0 floor catches these even with Semgrep disabled.
    tier0_covered = {"typo_squat_import"}
    assert off.get("typo_squat_import") == 0.0
    for attack_class, asr in off.items():
        if attack_class in tier0_covered:
            continue
        assert asr == 1.0, (
            f"semgrep-off ASR for {attack_class} should be 1.0 "
            f"(no-op pass-through); got {asr}"
        )


async def test_semgrep_on_catches_every_error_class() -> None:
    """semgrep_enabled=True -> ERROR-class scenarios deny on the
    fast tier; ASR=0%. WARNING-class scenarios still pass."""
    result = await run_g4_evaluation(
        rng_seed=42, n_attacks_per_class=5, n_benign=10,
    )
    on_cells = {
        c.attack_class: c.asr
        for c in result.asr_cells
        if c.config_name == "G4 fast-only (semgrep on)"
    }
    error_classes = {
        "reverse_shell",
        "eval_exec",
        "pickle_rce",
        "credential_file_read",
        "typo_squat_import",
        "env_var_hijack",
    }
    for cls in error_classes:
        assert on_cells[cls] == 0.0, (
            f"semgrep-on ASR for {cls} should be 0.0 (ERROR class); "
            f"got {on_cells[cls]}"
        )
    # WARNING-class is allowed to pass the fast tier; the
    # slow-tier Q-LLM is the deny-promoter.
    assert on_cells["direct_network_exfil"] == 1.0


async def test_benign_workload_fpr_is_zero() -> None:
    """No rule fires on the benign templates by construction, so
    FPR is zero across all non-baseline configurations."""
    result = await run_g4_evaluation(
        rng_seed=42, n_attacks_per_class=2, n_benign=50,
    )
    for cell in result.fpr_cells:
        assert cell.fpr == 0.0, (
            f"FPR for {cell.config_name} should be 0.0 (benign "
            f"workload has no findings); got {cell.fpr}"
        )


# -----------------------------------------------------------------
# Result shape + reproducibility
# -----------------------------------------------------------------


async def test_run_g4_evaluation_is_reproducible() -> None:
    a = await run_g4_evaluation(
        rng_seed=42, n_attacks_per_class=3, n_benign=10,
    )
    b = await run_g4_evaluation(
        rng_seed=42, n_attacks_per_class=3, n_benign=10,
    )
    assert tuple((c.attack_class, c.config_name, c.asr) for c in a.asr_cells) \
        == tuple((c.attack_class, c.config_name, c.asr) for c in b.asr_cells)


async def test_result_shape_pins_expected_keys() -> None:
    result = await run_g4_evaluation(
        rng_seed=42, n_attacks_per_class=2, n_benign=5,
    )
    expected_cells = len(G4_ATTACK_GENERATORS) * len(DEFAULT_G4_CONFIGS)
    assert len(result.asr_cells) == expected_cells
    assert len(result.fpr_cells) == len(DEFAULT_G4_CONFIGS)
    assert result.rng_seed == 42


# -----------------------------------------------------------------
# Report formatter
# -----------------------------------------------------------------


async def test_format_g4_report_includes_headline_and_tables() -> None:
    result = await run_g4_evaluation(
        rng_seed=42, n_attacks_per_class=2, n_benign=10,
    )
    body = format_g4_report(result)
    assert "G4 evaluation" in body
    assert "Headline" in body
    assert "ASR results" in body
    assert "False-positive rate" in body
    for display in G4_DISPLAY_NAMES.values():
        assert display in body, (
            f"display name {display!r} missing from report"
        )
