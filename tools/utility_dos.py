"""
Utility / DoS dashboard (W23.5).

Rev3-11 asks for utility/DoS metrics beyond FPR: task completion,
retries / reauths, attack-induced lockout, and sticky-floor DoS.

This tool synthesizes existing measurements from the benign control and
attack corpus into one operational dashboard. All measurements run through
the **deterministic fast-tier** ``full PALISADE`` config (the reported
operating point: trust scorer active, capability bound active, all gates
live, no Q-LLM).

**Benign side** (181-task W1 control):

- **Task completion** = benign utility actions allowed (= 1 − FPR).
- **Retryable block** = blocked but NOT SEV1 (the scientist can rephrase
  or adjust and resubmit; no session state is lost).
- **Lockout** = blocked AND SEV1 → the trust scorer would terminate the
  session; subsequent actions fail until re-authentication. This is the
  3.9% (7/181) benign-SEV1 number from Table E6.

**Attack side** (205-instance SIEGE corpus):

- **Attack blocked** = attack action denied by a live gate (the defense
  working as intended).
- **Utility under attack** = whether the instance's legitimate-task
  (``is_utility``) action was also blocked — the collateral cost of the
  defense.
- **Attack-induced lockout** = attack traces where a SEV1 denial locks
  the session, blocking ALL subsequent utility.

**Synthesis:**

- **Sticky-floor DoS amplification** = of the SEV1 blocks on attacks, how
  many also deny a utility action in the same trace (the lockout cost:
  catching the attacker blocks legitimate work too until re-auth).
- **Tie to the 3.9%** = the benign-SEV1 rate is the false-lockout floor
  the same sticky mechanism imposes on clean traffic.

Fully offline and deterministic.

    cd backend
    uv run python -m tools.utility_dos \\
        --report-out ../docs/palisade/utility_dos_w23.md
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

from siege.ablation_matrix import CUMULATIVE_CONFIGS
from siege.instance_loader import load_instances
from siege.session_runner import SessionRunner
from tools.benign_fpr import clopper_pearson, measure_benign_fpr
from siege.paths import CORPUS_DIR

__all__ = [
    "BenignUtilityReport",
    "AttackUtilityReport",
    "UtilityDosResult",
    "run_utility_dos",
]

_CORPUS_DIR = CORPUS_DIR
_FULL_CONFIG = CUMULATIVE_CONFIGS[-1]


@dataclass(frozen=True)
class BenignUtilityReport:
    total: int
    completed: int
    blocked_retryable: int
    blocked_lockout: int
    lockout_ci: tuple[float, float]
    lockout_signatures: dict[str, int]
    lockout_examples: list[tuple[str, str, str]]

    @property
    def completion_rate(self) -> float:
        return (self.completed / self.total) if self.total else 0.0

    @property
    def lockout_rate(self) -> float:
        return (self.blocked_lockout / self.total) if self.total else 0.0

    @property
    def retry_rate(self) -> float:
        return (self.blocked_retryable / self.total) if self.total else 0.0


@dataclass(frozen=True)
class AttackTraceOutcome:
    instance_id: str
    template: str
    attack_blocked: bool
    utility_blocked: bool
    has_sev1: bool
    blocking_gate: str
    blocking_reason: str


@dataclass(frozen=True)
class AttackUtilityReport:
    total: int
    attack_blocked: int
    utility_preserved: int
    utility_lost: int
    lockout_induced: int
    lockout_with_utility_loss: int
    outcomes: list[AttackTraceOutcome] = field(default_factory=list)

    @property
    def block_rate(self) -> float:
        return (self.attack_blocked / self.total) if self.total else 0.0

    @property
    def ua_rate(self) -> float:
        return (self.utility_preserved / self.total) if self.total else 0.0

    @property
    def lockout_rate(self) -> float:
        return (self.lockout_induced / self.total) if self.total else 0.0

    @property
    def dos_amplification(self) -> float:
        return (
            (self.lockout_with_utility_loss / self.lockout_induced)
            if self.lockout_induced
            else 0.0
        )


@dataclass
class UtilityDosResult:
    benign: BenignUtilityReport
    attack: AttackUtilityReport


async def _measure_benign() -> BenignUtilityReport:
    report = await measure_benign_fpr(config=_FULL_CONFIG)
    outcomes = report.outcomes
    total = len(outcomes)
    completed = sum(1 for o in outcomes if not o.blocked)
    blocked_lockout = sum(
        1 for o in outcomes if o.blocked and o.incident_level == 1
    )
    blocked_retryable = sum(
        1 for o in outcomes
        if o.blocked and o.incident_level != 1
    )
    import collections

    sigs = collections.Counter(
        o.signature for o in outcomes if o.blocked and o.incident_level == 1
    )
    examples = [
        (o.instance_id, o.gate or "—", o.reason)
        for o in outcomes
        if o.blocked and o.incident_level == 1
    ]
    return BenignUtilityReport(
        total=total,
        completed=completed,
        blocked_retryable=blocked_retryable,
        blocked_lockout=blocked_lockout,
        lockout_ci=clopper_pearson(blocked_lockout, total),
        lockout_signatures=dict(
            sorted(sigs.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        lockout_examples=examples,
    )


async def _measure_attack() -> AttackUtilityReport:
    instances = [
        i for i in load_instances(_CORPUS_DIR) if i.kind == "attack"
    ]
    outcomes: list[AttackTraceOutcome] = []
    for inst in instances:
        runner = SessionRunner()
        trace = await runner.run(inst, _FULL_CONFIG)
        attack_actions = [a for a in trace.actions if a.is_attack]
        utility_actions = [a for a in trace.actions if a.is_utility]
        attack_blocked = all(not a.allowed for a in attack_actions) if attack_actions else False
        utility_blocked = any(not a.allowed for a in utility_actions)
        has_sev1 = any(
            a.incident_level == 1 and not a.allowed for a in trace.actions
        )
        blocking = next(
            (a for a in trace.actions if not a.allowed and a.is_attack), None
        )
        outcomes.append(
            AttackTraceOutcome(
                instance_id=inst.instance_id,
                template=inst.template,
                attack_blocked=attack_blocked,
                utility_blocked=utility_blocked,
                has_sev1=has_sev1,
                blocking_gate=(blocking.gate or "—") if blocking else "—",
                blocking_reason=(blocking.reason or "") if blocking else "",
            )
        )
    total = len(outcomes)
    attack_blocked = sum(1 for o in outcomes if o.attack_blocked)
    utility_preserved = sum(1 for o in outcomes if not o.utility_blocked)
    utility_lost = sum(1 for o in outcomes if o.utility_blocked)
    lockout_induced = sum(1 for o in outcomes if o.has_sev1)
    lockout_with_utility_loss = sum(
        1 for o in outcomes if o.has_sev1 and o.utility_blocked
    )
    return AttackUtilityReport(
        total=total,
        attack_blocked=attack_blocked,
        utility_preserved=utility_preserved,
        utility_lost=utility_lost,
        lockout_induced=lockout_induced,
        lockout_with_utility_loss=lockout_with_utility_loss,
        outcomes=outcomes,
    )


def render(result: UtilityDosResult) -> str:
    b = result.benign
    a = result.attack
    lines = [
        "# Utility / DoS dashboard (W23.5)",
        "",
        "Operational framing of the defense's cost on legitimate work and the "
        "attack-side lockout dynamics. All measurements under the **deterministic "
        f"fast-tier `{_FULL_CONFIG.name}`** config (trust scorer active, capability "
        "bound active, all gates live, no Q-LLM). CIs are exact-binomial "
        "(Clopper-Pearson) 95%.",
        "",
        "## Benign side (W1 expanded control, n=181)",
        "",
        "| Outcome | Count | Rate | 95% CI |",
        "|---|---:|---:|---|",
        f"| Task completed | {b.completed}/{b.total} | "
        f"{b.completion_rate:.1%} | "
        f"{clopper_pearson(b.completed, b.total)[0]:.1%}–"
        f"{clopper_pearson(b.completed, b.total)[1]:.1%} |",
        f"| Blocked — retryable (non-SEV1) | {b.blocked_retryable}/{b.total} | "
        f"{b.retry_rate:.1%} | "
        f"{clopper_pearson(b.blocked_retryable, b.total)[0]:.1%}–"
        f"{clopper_pearson(b.blocked_retryable, b.total)[1]:.1%} |",
        f"| Blocked — lockout (SEV1) | {b.blocked_lockout}/{b.total} | "
        f"{b.lockout_rate:.1%} | "
        f"{b.lockout_ci[0]:.1%}–{b.lockout_ci[1]:.1%} |",
        "",
        "**Task completion = "
        f"{b.completion_rate:.1%}** (= 1 − FPR). Of the "
        f"{b.blocked_retryable + b.blocked_lockout} blocked benign tasks, "
        f"**{b.blocked_lockout}** are SEV1 (lockout: the sticky trust scorer "
        "terminates the session until re-authentication) and "
        f"**{b.blocked_retryable}** are lower-severity (retryable: the scientist "
        "rephrases or adjusts and resubmits without session state loss).",
    ]
    if b.lockout_signatures:
        lines += [
            "",
            "### Lockout signatures (benign SEV1)",
            "",
            "| Signature | Count |",
            "|---|---:|",
        ]
        for sig, n in b.lockout_signatures.items():
            lines.append(f"| `{sig}` | {n} |")
        lines += [
            "",
            "These are the same conservative signatures the FPR analysis names "
            "as tunable policy knobs — tightening them raises attack coverage "
            "at the cost of more benign lockouts; loosening them does the reverse. "
            "The lockout rate moves with the same knobs that set the FPR.",
        ]
    lines += [
        "",
        "## Attack side (SIEGE corpus, n=205)",
        "",
        "| Metric | Count | Rate |",
        "|---|---:|---:|",
        f"| Attack blocked (all attack actions denied) | {a.attack_blocked}/{a.total} | "
        f"{a.block_rate:.1%} |",
        f"| Utility preserved (legitimate task allowed) | {a.utility_preserved}/{a.total} | "
        f"{a.ua_rate:.1%} |",
        f"| Utility lost (legitimate task also blocked) | {a.utility_lost}/{a.total} | "
        f"{1 - a.ua_rate:.1%} |",
        f"| Attack-induced lockout (SEV1 on attack) | {a.lockout_induced}/{a.total} | "
        f"{a.lockout_rate:.1%} |",
        f"| Lockout + utility loss | {a.lockout_with_utility_loss}/{a.lockout_induced or 1} of lockouts | "
        f"{a.dos_amplification:.0%} |",
        "",
    ]
    if a.lockout_induced:
        lines += [
            f"**Attack-induced lockout: {a.lockout_induced}/{a.total} = "
            f"{a.lockout_rate:.1%}** of attack instances trigger a SEV1 "
            "that would terminate the session. Of those lockouts, "
            f"**{a.lockout_with_utility_loss}/{a.lockout_induced} = "
            f"{a.dos_amplification:.0%}** also block the instance's "
            "legitimate-task action — the DoS amplification: catching the "
            "attacker blocks legitimate work too until re-auth.",
            "",
        ]
    lines += [
        "## Synthesis: the operational tradeoff",
        "",
        f"- **Benign lockout floor: {b.lockout_rate:.1%}** "
        f"({b.blocked_lockout}/{b.total}) — the false-lockout rate the sticky "
        "mechanism imposes on clean traffic. This is the Table E6 number "
        "(7/181 = 3.9%).",
        f"- **Attack-induced lockout: {a.lockout_rate:.1%}** "
        f"({a.lockout_induced}/{a.total}) — the fraction of attacks that "
        "trigger a session-terminating SEV1.",
        f"- **Retries (non-lockout blocks): {b.retry_rate:.1%}** of benign "
        "tasks need a retry/adjustment (lower-severity blocks the scientist "
        "can recover from without re-auth).",
        "- **Re-authentication cost:** every lockout (benign or attack-induced) "
        "requires an explicit re-auth to restore capabilities. The lockout "
        "is intentionally expensive (it removes the probe-then-strike "
        "oscillation window), but it means a benign false-SEV1 carries the "
        "same re-auth overhead as a real attack.",
        "",
        "### DoS interpretation",
        "",
        "An adversary who triggers a SEV1 denial forces a session lockout — "
        "an **intended** consequence of the sticky trust scorer (blocking the "
        "oscillation attack). This is not a denial-of-service *vulnerability*; "
        "it is the cost of the sticky-floor guarantee. The operator trades "
        f"a {b.lockout_rate:.1%} benign-lockout floor for the property that "
        "no amount of benign probing can re-open a breached high-stakes "
        "capability within a session. The lockout rate is tunable via the same "
        "gate signatures that set the FPR (Section W1).",
        "",
    ]
    return "\n".join(lines)


def run_utility_dos() -> UtilityDosResult:
    benign = asyncio.run(_measure_benign())
    attack = asyncio.run(_measure_attack())
    return UtilityDosResult(benign=benign, attack=attack)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Utility / DoS dashboard (W23.5)."
    )
    parser.add_argument("--report-out", default=None, metavar="PATH")
    args = parser.parse_args(argv)

    result = run_utility_dos()
    md = render(result)
    if args.report_out:
        Path(args.report_out).write_text(md, encoding="utf-8")
        print(f"[utility-dos] wrote {args.report_out}")
    print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
