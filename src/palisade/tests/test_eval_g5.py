"""
Unit tests for the G5 evaluation harness.

Mirrors ``test_eval_g4.py``: determinism + shape, baseline-allows-
everything, the fast/slow tiers catch their target classes, FPR bounds,
reproducibility, and the report formatter. Encodes the work-item
acceptance criteria as assertions.
"""

from __future__ import annotations

import random

import pytest

pytestmark = pytest.mark.anyio

from siege.eval import (
    DEFAULT_G5_CONFIGS,
    G5_ATTACK_FAMILIES,
    G5_ATTACK_GENERATORS,
    G5_DISPLAY_NAMES,
    format_g5_report,
    generate_g5_attacks,
    generate_g5_benign_workload,
    run_g5_evaluation,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# -----------------------------------------------------------------
# Scenario generation: determinism + shape
# -----------------------------------------------------------------


def test_registry_has_fifteen_classes() -> None:
    assert len(G5_ATTACK_GENERATORS) == 15
    # Five per family, three families.
    assert set(G5_ATTACK_GENERATORS) == set().union(
        *(set(v) for v in G5_ATTACK_FAMILIES.values())
    )
    for family, classes in G5_ATTACK_FAMILIES.items():
        assert len(classes) == 5, f"{family} should have 5 classes"


def test_display_names_cover_every_class() -> None:
    for class_name in G5_ATTACK_GENERATORS:
        assert G5_DISPLAY_NAMES.get(class_name)


def test_generate_attacks_deterministic_and_shaped() -> None:
    for class_name in G5_ATTACK_GENERATORS:
        a = generate_g5_attacks(class_name, random.Random(7), n=5)
        b = generate_g5_attacks(class_name, random.Random(7), n=5)
        assert [s.slurm_script for s in a] == [s.slurm_script for s in b]
        assert len(a) == 5
        for s in a:
            assert s.attack_class == class_name
            assert isinstance(s.slurm_script, str) and s.slurm_script.strip()
            assert isinstance(s.user_config, dict)
            assert s.caught_by in {"fast", "slow"}


def test_generate_attacks_rejects_unknown_class() -> None:
    with pytest.raises(KeyError):
        generate_g5_attacks("nope", random.Random(0), n=1)


def test_benign_workload_deterministic() -> None:
    a = generate_g5_benign_workload(random.Random(11), n=10)
    b = generate_g5_benign_workload(random.Random(11), n=10)
    assert [s.slurm_script for s in a] == [s.slurm_script for s in b]
    assert all(s.attack_class == "benign_hpc_job" for s in a)


# -----------------------------------------------------------------
# Runner: ASR matrix (the acceptance criteria)
# -----------------------------------------------------------------


async def _result(n: int = 15, n_benign: int = 50):
    return await run_g5_evaluation(
        rng_seed=42, n_attacks_per_class=n, n_benign=n_benign
    )


def _asr_by_config(result, config_name: str) -> dict[str, float]:
    return {
        c.attack_class: c.asr
        for c in result.asr_cells
        if c.config_name == config_name
    }


async def test_baseline_asr_is_100_percent() -> None:
    """AC: baseline G5-disabled config has ASR = 100% across all classes."""
    result = await _result()
    for asr in _asr_by_config(result, "baseline").values():
        assert asr == 1.0


async def test_fast_only_catches_80pct_cryptomining_and_credential() -> None:
    """AC: G5 fast-only catches >= 80% of cryptomining + credential-exfil
    patterns (binary-signature / path-scope / network-egress)."""
    result = await _result()
    fast = _asr_by_config(result, "G5 fast-only")
    targets = G5_ATTACK_FAMILIES["cryptomining"] + G5_ATTACK_FAMILIES["credential"]
    caught = sum(1 for cls in targets if fast[cls] == 0.0)
    assert caught / len(targets) >= 0.80


async def test_fast_plus_slow_worst_case_below_10pct() -> None:
    """AC: G5 fast+slow brings worst-case ASR across all classes < 10%."""
    result = await _result()
    fastslow = _asr_by_config(result, "G5 fast+slow")
    worst = max(fastslow.values())
    assert worst < 0.10, f"worst-case fast+slow ASR {worst:.0%} >= 10%"


async def test_intent_only_classes_survive_fast_but_caught_by_slow() -> None:
    """The B5.4 intent-only patterns (+ the env-dump leak) are policy-clean:
    fast-only misses them, fast+slow catches them."""
    result = await _result()
    fast = _asr_by_config(result, "G5 fast-only")
    fastslow = _asr_by_config(result, "G5 fast+slow")
    intent_only = (
        "allocation_wrong_purpose",
        "allocation_project_scope",
        "credential_env_leak",
    )
    for cls in intent_only:
        assert fast[cls] == 1.0, f"{cls} should survive the fast tier"
        assert fastslow[cls] == 0.0, f"{cls} should be caught fast+slow"


async def test_policy_classes_caught_by_fast_tier() -> None:
    """Allocation policy + resource ceiling violations are deterministic
    fast-tier catches."""
    result = await _result()
    fast = _asr_by_config(result, "G5 fast-only")
    for cls in (
        "allocation_wrong_alloc",
        "allocation_resource_scale",
        "allocation_pi_mismatch",
    ):
        assert fast[cls] == 0.0


async def test_benign_fpr_below_2_percent() -> None:
    """AC: FPR on the 50-script benign workload (real hpc_jobs/) < 2%."""
    result = await _result()
    for cell in result.fpr_cells:
        assert cell.fpr < 0.02, (
            f"FPR for {cell.config_name} is {cell.fpr:.1%} (>= 2%)"
        )


# -----------------------------------------------------------------
# Shape + reproducibility
# -----------------------------------------------------------------


async def test_default_n_attacks_per_class_is_15() -> None:
    """AC: n_attacks_per_class defaults to 15."""
    result = await run_g5_evaluation(rng_seed=42, n_benign=10)
    assert result.n_attacks_per_class == 15


async def test_result_shape() -> None:
    result = await _result(n=2, n_benign=5)
    expected = len(G5_ATTACK_GENERATORS) * len(DEFAULT_G5_CONFIGS)
    assert len(result.asr_cells) == expected
    assert len(result.fpr_cells) == len(DEFAULT_G5_CONFIGS)
    assert result.rng_seed == 42


async def test_reproducible() -> None:
    a = await _result(n=3, n_benign=10)
    b = await _result(n=3, n_benign=10)
    assert tuple((c.attack_class, c.config_name, c.asr) for c in a.asr_cells) == \
        tuple((c.attack_class, c.config_name, c.asr) for c in b.asr_cells)


# -----------------------------------------------------------------
# Report formatter
# -----------------------------------------------------------------


async def test_report_includes_headline_tables_and_classes() -> None:
    result = await _result(n=2, n_benign=10)
    body = format_g5_report(result)
    assert "G5 evaluation" in body
    assert "Headline" in body
    assert "ASR results" in body
    assert "False-positive rate" in body
    for display in G5_DISPLAY_NAMES.values():
        assert display in body, f"display name {display!r} missing"
