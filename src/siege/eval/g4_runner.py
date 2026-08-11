"""
G4 evaluation runner.

Threads malicious-code scenarios + a benign code workload through
the real ``G4CodeGate`` in three configurations:

1. **baseline** -- G4 disabled. Every attack succeeds; benign
   scenarios trivially pass. Control row.
2. **G4 fast-only (semgrep disabled)** -- gate is on but
   ``semgrep_enabled=False``. Per AC, this is a no-op pass-through
   (the slow tier is the only line of defense; not measured
   here). Useful as a sanity row: the gate doesn't deny when
   Semgrep is intentionally off.
3. **G4 fast-only (semgrep enabled)** -- gate is on,
   ``semgrep_enabled=True``. Mocked Semgrep returns each
   scenario's recorded findings; the gate's severity routing
   (ERROR -> SEV2 deny, WARNING -> SEV3 allow, none -> allow)
   is what's measured.

Semgrep is mocked across all configurations. The harness never
spawns the CLI -- the gate's ``_run_semgrep`` method is
monkey-patched to return the scenario's findings verbatim. This
keeps the suite fast and runnable without the
``[palisade-g4]`` optional install.

## Metrics

For each (attack class, configuration) cell:

- **ASR** = fraction of scenarios where the attack passed
  through G4's fast tier without a deny decision. WARNING-only
  findings (``direct_network_exfil``) count as a *pass*: the
  fast tier logs SEV3 but allows, and the slow-tier Q-LLM
  (a follow-on) is what promotes those to deny. The
  harness pins this honest accounting.

For each configuration:

- **FPR** = fraction of the benign-code workload G4 incorrectly
  denied. The fast tier returns no findings on the benign
  workload by construction (no rule fires), so FPR is zero in
  this harness modulo defects in the gate's empty-findings
  path.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from palisade.capabilities import CapabilityRegistry
from palisade.config import PalisadeSettings
from palisade.gates.base import GateContext
from palisade.gates.g4_code import G4CodeGate, SemgrepFinding
from palisade.trust import TrustScorer
from siege.eval.g4_attacks import (
    ATTACK_GENERATORS,
    G4AttackScenario,
    generate_attacks,
    generate_benign_workload,
)


# -----------------------------------------------------------------
# Result dataclasses
# -----------------------------------------------------------------


@dataclass(frozen=True)
class G4EvalConfig:
    """A single G4 evaluation configuration."""

    name: str
    gate_enabled: bool
    semgrep_enabled: bool


DEFAULT_G4_CONFIGS: tuple[G4EvalConfig, ...] = (
    G4EvalConfig(
        name="baseline", gate_enabled=False, semgrep_enabled=False,
    ),
    G4EvalConfig(
        name="G4 fast-only (semgrep off)",
        gate_enabled=True,
        semgrep_enabled=False,
    ),
    G4EvalConfig(
        name="G4 fast-only (semgrep on)",
        gate_enabled=True,
        semgrep_enabled=True,
    ),
)


@dataclass(frozen=True)
class G4CellResult:
    """ASR for one (attack class, config) cell."""

    attack_class: str
    config_name: str
    n_attacks: int
    asr: float
    sample_reason: str = ""


@dataclass(frozen=True)
class G4FprResult:
    """False-positive rate for one configuration on the benign workload."""

    config_name: str
    n_benign: int
    fpr: float
    sample_block_reason: str = ""


@dataclass(frozen=True)
class G4EvaluationResult:
    """Top-level result for the report formatter."""

    configs: tuple[G4EvalConfig, ...]
    attack_classes: tuple[str, ...]
    asr_cells: tuple[G4CellResult, ...]
    fpr_cells: tuple[G4FprResult, ...]
    n_attacks_per_class: int
    n_benign: int
    rng_seed: int


# -----------------------------------------------------------------
# Per-scenario evaluation
# -----------------------------------------------------------------


def _findings_from_scenario(
    scenario: G4AttackScenario,
) -> list[SemgrepFinding]:
    """Convert the scenario's ``findings`` tuple into the gate's
    ``SemgrepFinding`` dataclasses."""
    out: list[SemgrepFinding] = []
    for idx, (check_id, severity, message) in enumerate(scenario.findings, 1):
        out.append(
            SemgrepFinding(
                check_id=check_id,
                severity=severity,
                message=message,
                start_line=idx,
                end_line=idx,
            )
        )
    return out


def _build_gate(
    config: G4EvalConfig,
    scenario: G4AttackScenario,
) -> tuple[G4CodeGate, GateContext]:
    """
    Build a ``G4CodeGate`` + ``GateContext`` for one scenario
    under one configuration. The gate's ``_run_semgrep`` method is
    monkey-patched on the *instance* to return the scenario's
    findings, so no Semgrep subprocess is spawned.
    """
    gate = G4CodeGate(
        enabled=config.gate_enabled,
        semgrep_enabled=config.semgrep_enabled,
    )

    findings = _findings_from_scenario(scenario)

    async def fake_run_semgrep(code: str, language: str) -> list[SemgrepFinding]:
        return list(findings)

    gate._run_semgrep = fake_run_semgrep  # type: ignore[method-assign]

    ctx = GateContext(
        capability_registry=CapabilityRegistry(),
        trust_scorer=TrustScorer(PalisadeSettings(enabled=True)),
    )
    return gate, ctx


async def _attack_succeeded(
    gate: G4CodeGate,
    ctx: GateContext,
    scenario: G4AttackScenario,
) -> tuple[bool, str]:
    """
    Run one scenario through ``check_fast``. Returns
    ``(success, reason)`` where ``success=True`` means the gate
    did NOT deny the tool call.

    WARNING-only findings allow with a SEV3 incident -- those
    count as attack-success at the fast-tier-only layer (the
    slow-tier Q-LLM is the layer that promotes WARNINGs to deny
    based on intent; that's a separate issue).
    """
    payload = {"tool_name": scenario.tool_name, "args": scenario.args}
    decision = await gate.check_fast(payload, ctx)
    if not decision.allow:
        return False, decision.reason
    return True, decision.reason or "attack passed through unmodified"


async def _benign_was_blocked(
    gate: G4CodeGate,
    ctx: GateContext,
    scenario: G4AttackScenario,
) -> tuple[bool, str]:
    """Same flow, benign scenarios; a deny inflates FPR."""
    payload = {"tool_name": scenario.tool_name, "args": scenario.args}
    decision = await gate.check_fast(payload, ctx)
    if not decision.allow:
        return True, decision.reason
    return False, ""


# -----------------------------------------------------------------
# Public runner
# -----------------------------------------------------------------


async def run_g4_evaluation(
    *,
    rng_seed: int = 42,
    n_attacks_per_class: int = 20,
    n_benign: int = 50,
    configs: tuple[G4EvalConfig, ...] = DEFAULT_G4_CONFIGS,
) -> G4EvaluationResult:
    """
    Run the full G4 evaluation matrix.

    Args:
        rng_seed: pin reproducibility.
        n_attacks_per_class: scenarios per attack class.
        n_benign: benign-workload size for FPR measurement.
        configs: configurations to measure.
    """
    rng = random.Random(rng_seed)

    asr_cells: list[G4CellResult] = []
    for attack_class in ATTACK_GENERATORS:
        scenarios = generate_attacks(attack_class, rng, n_attacks_per_class)
        for config in configs:
            successes = 0
            sample_reason = ""
            for scenario in scenarios:
                gate, ctx = _build_gate(config, scenario)
                succeeded, reason = await _attack_succeeded(gate, ctx, scenario)
                if succeeded:
                    successes += 1
                    if not sample_reason:
                        sample_reason = reason
            asr = successes / n_attacks_per_class if n_attacks_per_class else 0.0
            asr_cells.append(
                G4CellResult(
                    attack_class=attack_class,
                    config_name=config.name,
                    n_attacks=n_attacks_per_class,
                    asr=asr,
                    sample_reason=sample_reason if asr > 0 else "",
                )
            )

    benign = generate_benign_workload(rng, n_benign)
    fpr_cells: list[G4FprResult] = []
    for config in configs:
        blocked = 0
        sample_reason = ""
        for scenario in benign:
            gate, ctx = _build_gate(config, scenario)
            was_blocked, reason = await _benign_was_blocked(gate, ctx, scenario)
            if was_blocked:
                blocked += 1
                if not sample_reason:
                    sample_reason = reason
        fpr = blocked / n_benign if n_benign else 0.0
        fpr_cells.append(
            G4FprResult(
                config_name=config.name,
                n_benign=n_benign,
                fpr=fpr,
                sample_block_reason=sample_reason,
            )
        )

    return G4EvaluationResult(
        configs=tuple(configs),
        attack_classes=tuple(ATTACK_GENERATORS.keys()),
        asr_cells=tuple(asr_cells),
        fpr_cells=tuple(fpr_cells),
        n_attacks_per_class=n_attacks_per_class,
        n_benign=n_benign,
        rng_seed=rng_seed,
    )
