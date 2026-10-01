"""
Ablation-matrix tests.

Pins the 6-configuration **B4-first** cumulative shape
(``baseline -> +G4 -> +G4+G3 -> +G4+G3+G1 -> +G4+G3+G1+G5 -> full``) and
that ``build_gate_stack`` produces the real, correctly-enabled gates per
config.
"""

from __future__ import annotations

from siege.ablation_matrix import (
    AUGMENTED_CONFIGS,
    CUMULATIVE_CONFIGS,
    RECOMMENDED_CONFIG,
    RECOMMENDED_SEV2_CONFIG,
    GateLayer,
    build_gate_stack,
)


def test_six_cumulative_configs_b4_first():
    names = [c.name for c in CUMULATIVE_CONFIGS]
    assert names == [
        "baseline",
        "+G4",
        "+G4+G3",
        "+G4+G3+G1",
        "+G4+G3+G1+G5",
        "full PALISADE",
    ]
    # baseline has no layers; full has all five and the trust scorer.
    assert CUMULATIVE_CONFIGS[0].layers == frozenset()
    assert CUMULATIVE_CONFIGS[0].trust_active is False
    assert CUMULATIVE_CONFIGS[-1].layers == frozenset(GateLayer)
    assert CUMULATIVE_CONFIGS[-1].trust_active is True


def test_code_gate_added_first():
    # B4-first: the first non-baseline config adds exactly G4.
    assert CUMULATIVE_CONFIGS[1].layers == frozenset({GateLayer.G4})


def test_configs_are_cumulative():
    # Each config's layers are a superset of the previous one's; the four
    # single-layer steps grow by exactly one, and `full` adds the G2
    # enforcement wrapper on top.
    prev = frozenset()
    for config in CUMULATIVE_CONFIGS:
        assert prev <= config.layers
        prev = config.layers


def test_trust_active_only_in_full():
    active = [c.name for c in CUMULATIVE_CONFIGS if c.trust_active]
    assert active == ["full PALISADE"]


def test_g2_only_in_full():
    # The B2 enforcement wrapper (G2) appears only in the full config.
    with_g2 = [c.name for c in CUMULATIVE_CONFIGS if GateLayer.G2 in c.layers]
    assert with_g2 == ["full PALISADE"]


def test_gate_stack_baseline_has_no_gates():
    stack = build_gate_stack(CUMULATIVE_CONFIGS[0])
    assert stack.live_ids() == frozenset()
    assert stack.gate_for("G4") is None


def test_gate_stack_g4_first_config():
    stack = build_gate_stack(CUMULATIVE_CONFIGS[1])
    assert stack.live_ids() == frozenset({"G4"})
    g4 = stack.gate_for("G4")
    assert g4 is not None and g4.enabled
    # G1/G3/G5 not yet live in the B4-first first step.
    assert stack.gate_for("G1") is None


def test_gate_stack_full_has_all_base_gates():
    stack = build_gate_stack(CUMULATIVE_CONFIGS[-1])
    assert stack.live_ids() == frozenset({"G1", "G2", "G3", "G4", "G5", "G6"})
    for gid in ("G1", "G2", "G3", "G4", "G5", "G6"):
        gate = stack.gate_for(gid)
        assert gate is not None and gate.enabled


# -----------------------------------------------------------------
# Posture configs
# -----------------------------------------------------------------
#
# The manuscript recommends a posture -- full, G1 weaponization signature off,
# stickiness off -- that used to exist only as a monkeypatch inside one
# analysis tool. That made it unselectable by a reader and unreachable by a
# search driver, which is why it is the one configuration the adaptive runs
# never covered. These pin it as a real config that threads into the gates.


def test_standing_configs_keep_todays_posture():
    """Every standing config must default to the configuration the reported
    numbers were measured under, or adding the knobs moves a result."""
    for cfg in (*CUMULATIVE_CONFIGS, *AUGMENTED_CONFIGS):
        assert cfg.g1_weaponization_signature is True, cfg.name
        assert cfg.g5_lifecycle_hook is True, cfg.name
        assert cfg.g5_lifecycle_hook_severity == 1, cfg.name
        assert cfg.sticky_high_stakes is True, cfg.name


def test_recommended_config_is_full_with_the_two_knobs_off():
    full = CUMULATIVE_CONFIGS[-1]
    rec = RECOMMENDED_CONFIG
    # Same layers and the same two full-only enforcement mechanisms.
    assert rec.layers == full.layers
    assert rec.capability_bound_active is True
    assert rec.trust_scorer_active is True
    # The two knobs the recommendation turns off.
    assert rec.g1_weaponization_signature is False
    assert rec.sticky_high_stakes is False
    # And the one it explicitly keeps: G5's lifecycle-hook check is the only
    # check that sees --prolog / --epilog.
    assert rec.g5_lifecycle_hook is True
    assert rec.g5_lifecycle_hook_severity == 1


def test_recommended_sev2_config_only_moves_the_severity():
    rec, sev2 = RECOMMENDED_CONFIG, RECOMMENDED_SEV2_CONFIG
    assert sev2.g5_lifecycle_hook is True, "SEV2 routing keeps the check"
    assert sev2.g5_lifecycle_hook_severity == 2
    for field in (
        "layers",
        "capability_bound_active",
        "trust_scorer_active",
        "semgrep_active",
        "quarantine_active",
        "g1_weaponization_signature",
        "sticky_high_stakes",
    ):
        assert getattr(sev2, field) == getattr(rec, field), field


def test_posture_knobs_reach_the_gates():
    """The knobs are worth nothing if build_gate_stack drops them."""
    stack = build_gate_stack(RECOMMENDED_SEV2_CONFIG)
    g1, g5 = stack.gate_for("G1"), stack.gate_for("G5")
    assert g1._match_weaponization("weapons-grade plutonium") is None
    assert g5._lifecycle_hook_severity == 2
    assert g5._lifecycle_hook_enabled is True

    default = build_gate_stack(CUMULATIVE_CONFIGS[-1])
    assert default.gate_for("G1")._match_weaponization("weapons-grade plutonium")
    assert default.gate_for("G5")._lifecycle_hook_severity == 1
