"""
SEV2 routing for lifecycle-hook denials (E4).

The manuscript calls this "the cheapest unmeasured change we can name". G5's
lifecycle-hook check is the only check that sees ``--prolog`` / ``--epilog``,
so it stays on, but its denial is raised at SEV1 -- and a SEV1 routed through
the incident manager terminates the session, which re-authentication does not
revive. Legitimate staging and archival jobs use the same directives, so the
five benign lifecycle-hook blocks end in an irrecoverable session. Routing the
same denial at SEV2 floors the session tier and forces re-authentication
instead, keeping the check while making its benign case recoverable.

This tool measures both halves of that trade on the recommended posture:

* **Benign (181 controls).** How many blocks fall in the irrecoverable class
  (SEV1) versus the recoverable class (SEV2/SEV3), with an exact-binomial CI.
  The total number of blocks must not move: SEV2 routing changes where a denial
  lands in the playbook, never whether the action is denied.

* **B5 attacks (55 instances).** Whether routing at SEV2 reopens any attack.
  Two readings, because they answer different questions:

  1. *Replayed admission.* Every action's disposition under both configs. The
     SIEGE runner records ``incident_level`` but never drives the incident
     manager (nothing calls ``TrustScorer.notify_incident`` in the replay
     path), so no replayed session terminates and severity cannot move an
     admission decision. This reading is therefore a **regression check** --
     it confirms the severity knob touches nothing else -- and not evidence
     about the deployed system.

  2. *Post-termination exposure.* The reading that actually answers the
     question. In deployment a SEV1 lifecycle-hook denial ends the session at
     that action, so every later action in the session never executes. Under
     SEV2 the session survives re-authentication and those actions do run.
     The attack surface SEV2 routing opens is exactly the set of actions that
     follow a lifecycle-hook SEV1 denial in the same session and are admitted.
     This tool enumerates them, and flags any that reaches a privileged sink.

Fully offline and deterministic; no served model.

    uv run python -m tools.sev2_routing
    uv run python -m tools.sev2_routing --report-out docs/palisade/sev2_routing.md
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

from siege import SessionRunner, load_instances
from siege.ablation_matrix import (
    RECOMMENDED_CONFIG,
    RECOMMENDED_SEV2_CONFIG,
    AblationConfig,
)
from siege.paths import CORPUS_DIR
from siege.schemas import HIGH_PRIV_SINKS
from siege.scorer import score_trace
from tools.benign_fpr import clopper_pearson, measure_benign_fpr
from tools.utility_dos import cut_by_termination, is_terminating

__all__ = [
    "BenignSeverityReport",
    "CollateralReport",
    "ExposedAction",
    "AttackRoutingReport",
    "Sev2RoutingResult",
    "run_sev2_routing",
    "to_markdown",
]

#: The lifecycle-hook denial reason prefix, as G5 emits it. Both the directive
#: route and the env route say "lifecycle-hook"; the directive-substitution
#: route is the third path the severity knob moves and says "substitution".
_LIFECYCLE_MARKERS = ("lifecycle-hook", "directive command substitution")


def _is_lifecycle_denial(reason: str) -> bool:
    lowered = (reason or "").lower()
    return any(m in lowered for m in _LIFECYCLE_MARKERS)


# -----------------------------------------------------------------
# Benign side: which class does each block land in
# -----------------------------------------------------------------


@dataclass(frozen=True)
class BenignSeverityReport:
    config_name: str
    total: int
    blocked: int
    irrecoverable: int  # blocked at SEV1 -> session terminated, no re-auth
    recoverable: int  # blocked at SEV2/SEV3 -> floored or logged
    irrecoverable_ci: tuple[float, float]
    examples: list[tuple[str, int, str]] = field(default_factory=list)

    @property
    def irrecoverable_rate(self) -> float:
        return self.irrecoverable / self.total if self.total else 0.0


async def _measure_benign(config: AblationConfig) -> BenignSeverityReport:
    report = await measure_benign_fpr(config=config)
    outcomes = report.outcomes
    blocked = [o for o in outcomes if o.blocked]
    irrecoverable = [o for o in blocked if o.incident_level == 1]
    return BenignSeverityReport(
        config_name=config.name,
        total=len(outcomes),
        blocked=len(blocked),
        irrecoverable=len(irrecoverable),
        recoverable=len(blocked) - len(irrecoverable),
        irrecoverable_ci=clopper_pearson(len(irrecoverable), len(outcomes)),
        examples=[
            (o.instance_id, o.incident_level or 0, o.reason) for o in blocked
        ],
    )


# -----------------------------------------------------------------
# Attack side: what SEV2 routing exposes
# -----------------------------------------------------------------


@dataclass(frozen=True)
class ExposedAction:
    """An action that a SEV1 termination would have prevented from running.

    It follows a lifecycle-hook SEV1 denial in the same session and is
    admitted, so under SEV2 routing it executes after re-authentication.
    """

    instance_id: str
    template: str
    session_id: str
    turn_index: int
    action_index: int
    is_attack: bool
    sink_tool: str | None

    @property
    def reaches_sink(self) -> bool:
        return self.sink_tool in HIGH_PRIV_SINKS


@dataclass(frozen=True)
class AttackRoutingReport:
    total: int
    #: Instances whose per-action admission differs between the two configs.
    admission_diffs: list[str] = field(default_factory=list)
    #: Instances whose soft/hard-win verdict differs between the two configs.
    verdict_diffs: list[str] = field(default_factory=list)
    #: Instances containing a lifecycle-hook SEV1 denial (the ones the knob
    #: touches at all).
    lifecycle_instances: list[str] = field(default_factory=list)
    #: Admitted actions after such a denial -- the surface SEV2 routing opens.
    exposed: list[ExposedAction] = field(default_factory=list)
    soft_wins: int = 0
    hard_wins: int = 0

    @property
    def exposed_reaching_sink(self) -> list[ExposedAction]:
        return [e for e in self.exposed if e.reaches_sink]

    @property
    def exposed_attack_actions(self) -> list[ExposedAction]:
        return [e for e in self.exposed if e.is_attack]


def _action_key(rec) -> tuple[str, int, int]:
    return (rec.session_id, rec.turn_index, rec.action_index)


def _is_lifecycle_sev1(rec) -> bool:
    return is_terminating(rec) and _is_lifecycle_denial(rec.reason)


def _exposed_after_lifecycle_sev1(trace, inst) -> list[ExposedAction]:
    """Admitted actions a deployed lifecycle-hook SEV1 termination cuts off.

    The cut is ``tools.utility_dos.cut_by_termination`` keyed on the first
    lifecycle-hook SEV1 in each session; of what it cuts, the admitted records
    are the ones SEV2 routing lets run after re-authentication.
    """
    return [
        ExposedAction(
            instance_id=inst.instance_id,
            template=inst.template,
            session_id=rec.session_id,
            turn_index=rec.turn_index,
            action_index=rec.action_index,
            is_attack=rec.is_attack,
            sink_tool=rec.sink_tool,
        )
        for rec in cut_by_termination(trace, _is_lifecycle_sev1)
        if rec.allowed
    ]


def _has_lifecycle_sev1(trace) -> bool:
    return any(_is_lifecycle_sev1(r) for r in trace.actions)


async def _measure_attacks(family_prefix: str = "b5_") -> AttackRoutingReport:
    instances = [
        i
        for i in load_instances(CORPUS_DIR)
        if i.kind == "attack" and i.template.startswith(family_prefix)
    ]
    admission_diffs: list[str] = []
    verdict_diffs: list[str] = []
    lifecycle_instances: list[str] = []
    exposed: list[ExposedAction] = []
    soft = hard = 0

    for inst in instances:
        t1 = await SessionRunner().run(inst, RECOMMENDED_CONFIG)
        t2 = await SessionRunner().run(inst, RECOMMENDED_SEV2_CONFIG)

        a1 = {_action_key(r): r.allowed for r in t1.actions}
        a2 = {_action_key(r): r.allowed for r in t2.actions}
        if a1 != a2:
            admission_diffs.append(inst.instance_id)

        s1, s2 = score_trace(t1, inst), score_trace(t2, inst)
        if (s1.attack_success, s1.hard_win) != (s2.attack_success, s2.hard_win):
            verdict_diffs.append(inst.instance_id)
        soft += bool(s1.attack_success)
        hard += bool(s1.hard_win)

        if _has_lifecycle_sev1(t1):
            lifecycle_instances.append(inst.instance_id)
            exposed.extend(_exposed_after_lifecycle_sev1(t1, inst))

    return AttackRoutingReport(
        total=len(instances),
        admission_diffs=admission_diffs,
        verdict_diffs=verdict_diffs,
        lifecycle_instances=lifecycle_instances,
        exposed=exposed,
        soft_wins=soft,
        hard_wins=hard,
    )


# -----------------------------------------------------------------
# Collateral: what a deployed SEV1 termination costs legitimate work
# -----------------------------------------------------------------


@dataclass(frozen=True)
class CollateralReport:
    """Utility lost on attack sessions that trigger a SEV1, under each routing.

    The gate allows each instance's legitimate action on its merits; what
    removes it in deployment is the session ending underneath it, which the
    replay harness never does. ``tools.utility_dos`` counts it the same way.
    SEV2 routing only reaches the lifecycle-hook check, so the SEV1 denials
    from every other check still terminate under it.
    """

    sev1_instances: list[str] = field(default_factory=list)
    lost_to_gate_denial: list[str] = field(default_factory=list)
    lost_to_termination: list[str] = field(default_factory=list)
    lost_to_termination_sev2: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.sev1_instances)

    @property
    def denial_rate(self) -> float:
        return len(self.lost_to_gate_denial) / self.total if self.total else 0.0

    @property
    def termination_rate(self) -> float:
        return len(self.lost_to_termination) / self.total if self.total else 0.0

    @property
    def recovered_by_sev2(self) -> list[str]:
        still = set(self.lost_to_termination_sev2)
        return [i for i in self.lost_to_termination if i not in still]


def _loses_utility_to_termination(trace) -> bool:
    return any(
        r.is_utility and r.allowed for r in cut_by_termination(trace)
    )


async def _measure_collateral() -> CollateralReport:
    sev1: list[str] = []
    by_denial: list[str] = []
    by_termination: list[str] = []
    by_termination_sev2: list[str] = []

    for inst in load_instances(CORPUS_DIR):
        if inst.kind != "attack":
            continue
        trace = await SessionRunner().run(inst, RECOMMENDED_CONFIG)
        if not any(is_terminating(r) for r in trace.actions):
            continue
        sev1.append(inst.instance_id)
        if any(r.is_utility and not r.allowed for r in trace.actions):
            by_denial.append(inst.instance_id)
        if _loses_utility_to_termination(trace):
            by_termination.append(inst.instance_id)
        trace_sev2 = await SessionRunner().run(inst, RECOMMENDED_SEV2_CONFIG)
        if _loses_utility_to_termination(trace_sev2):
            by_termination_sev2.append(inst.instance_id)

    return CollateralReport(
        sev1_instances=sev1,
        lost_to_gate_denial=by_denial,
        lost_to_termination=by_termination,
        lost_to_termination_sev2=by_termination_sev2,
    )


@dataclass
class Sev2RoutingResult:
    benign_sev1: BenignSeverityReport
    benign_sev2: BenignSeverityReport
    attack: AttackRoutingReport
    collateral: CollateralReport


async def run_sev2_routing() -> Sev2RoutingResult:
    return Sev2RoutingResult(
        benign_sev1=await _measure_benign(RECOMMENDED_CONFIG),
        benign_sev2=await _measure_benign(RECOMMENDED_SEV2_CONFIG),
        attack=await _measure_attacks(),
        collateral=await _measure_collateral(),
    )


def _attack_and_sink(a: "AttackRoutingReport") -> list[ExposedAction]:
    """Exposed actions that are both an attack and reach a privileged sink --
    the only cell of the exposure table that is a security cost."""
    return [e for e in a.exposed if e.is_attack and e.reaches_sink]


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def to_markdown(r: Sev2RoutingResult) -> str:
    b1, b2, a = r.benign_sev1, r.benign_sev2, r.attack
    out = [
        "<!-- Generated report. Source module: tools.sev2_routing. "
        "Regenerate with `uv run python -m tools.sev2_routing`. -->",
        "",
        "# SEV2 routing for lifecycle-hook denials (E4)",
        "",
        "G5's lifecycle-hook check is the only check that sees `--prolog` / "
        "`--epilog`, so it stays on. Its denial is raised at SEV1, and a SEV1 "
        "terminates the session in a way re-authentication does not revive -- "
        "which legitimate staging and archival jobs, using the same "
        "directives, pay. Routing the same denial at SEV2 floors the session "
        "and forces re-authentication instead. Measured on the recommended "
        "posture (`full`, G1 weaponization signature off, stickiness off).",
        "",
        "## Benign controls (181 tasks)",
        "",
        "| Routing | Blocked | Irrecoverable (SEV1) | Recoverable | Irrecoverable rate | 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for rep, label in ((b1, "SEV1 (current)"), (b2, "SEV2 (proposed)")):
        lo, hi = rep.irrecoverable_ci
        out.append(
            f"| {label} | {rep.blocked}/{rep.total} | {rep.irrecoverable} | "
            f"{rep.recoverable} | {_pct(rep.irrecoverable_rate)} | "
            f"[{_pct(lo)}, {_pct(hi)}] |"
        )
    same_blocks = b1.blocked == b2.blocked
    out += [
        "",
        f"Total blocks unchanged: **{'yes' if same_blocks else 'NO'}** "
        f"({b1.blocked} under both). SEV2 routing moves where a denial lands "
        "in the playbook, not whether the action is denied -- so the "
        "false-positive rate is identical and only the recoverability of "
        "those refusals changes.",
        "",
        f"Irrecoverable benign sessions: **{b1.irrecoverable}/{b1.total} "
        f"({_pct(b1.irrecoverable_rate)}) -> {b2.irrecoverable}/{b2.total} "
        f"({_pct(b2.irrecoverable_rate)})**.",
        "",
        "## B5 attacks (55 instances)",
        "",
        f"- Soft wins: **{a.soft_wins}/{a.total}**; hard wins: "
        f"**{a.hard_wins}/{a.total}** (unchanged by routing).",
        f"- Instances whose per-action admission differs between the two "
        f"routings: **{len(a.admission_diffs)}**.",
        f"- Instances whose soft/hard-win verdict differs: "
        f"**{len(a.verdict_diffs)}**.",
        "",
        "### Reading 1: replayed admission (regression check)",
        "",
        "The SIEGE runner records `incident_level` but never drives the "
        "incident manager -- nothing in the replay path calls "
        "`TrustScorer.notify_incident` -- so no replayed session terminates "
        "and severity cannot move an admission decision. The zero above "
        "confirms the knob touches nothing else; it is **not** evidence that "
        "the deployed system is unaffected.",
        "",
        "### Reading 2: post-termination exposure (the real question)",
        "",
        "In deployment a SEV1 lifecycle-hook denial ends the session at that "
        "action, so later actions never execute. Under SEV2 they run after "
        "re-authentication. The surface SEV2 routing opens is therefore the "
        "set of admitted actions ordered after a lifecycle-hook SEV1 denial "
        "in the same session.",
        "",
        f"- B5 instances containing a lifecycle-hook SEV1 denial: "
        f"**{len(a.lifecycle_instances)}/{a.total}**.",
        f"- Admitted actions after such a denial: **{len(a.exposed)}**.",
        f"- Of those, **attack** actions -- the security cost of the change: "
        f"**{len(a.exposed_attack_actions)}**"
        + (
            f", of which {len(_attack_and_sink(a))} reach a privileged sink."
            if a.exposed_attack_actions
            else "."
        ),
        f"- Of those, **legitimate** actions the SEV1 termination was "
        f"collaterally killing -- the utility the change recovers: "
        f"**{len(a.exposed) - len(a.exposed_attack_actions)}**.",
        "",
    ]
    if a.exposed and not a.exposed_attack_actions:
        out += [
            "**Re-authentication reopens no attack.** Every action the SEV1 "
            "termination was cutting off is the instance's own legitimate "
            "job, denied nothing by the gate and killed only by the session "
            "ending. On this corpus the change is free on security and "
            "positive on utility: the attack is still denied at the same "
            "action, and the benign work behind it completes.",
            "",
        ]
    if a.exposed:
        out += [
            "| instance | template | turn.action | role | sink |",
            "|---|---|---|---|---|",
        ]
        for e in a.exposed:
            role = "**attack**" if e.is_attack else "legitimate"
            out.append(
                f"| `{e.instance_id}` | `{e.template}` | "
                f"{e.turn_index}.{e.action_index} | {role} | "
                f"{e.sink_tool or '--'} |"
            )
        out.append("")
    else:
        out += [
            "**No admitted action follows a lifecycle-hook SEV1 denial in any "
            "B5 instance**, so re-authentication reopens no attack: the "
            "denial is the last thing that happens in every session it fires "
            "in. SEV2 routing is free on this corpus.",
            "",
        ]
    c = r.collateral
    out += [
        "## Collateral cost of the SEV1 path (all 205 attack instances)",
        "",
        "The same non-termination gap hides what SEV1 costs legitimate work on "
        "attack sessions. The gate allows each instance's legitimate action on "
        "its merits, so counting gate denials reads zero. What removes it in "
        "deployment is the session ending underneath it. `tools.utility_dos` "
        "counts it this way for the same reason.",
        "",
        "| Attack instances carrying a SEV1 denial | Legitimate action lost to a gate denial | Lost to session termination, SEV1 routing | Lost to session termination, SEV2 routing |",
        "|---|---|---|---|",
        f"| {c.total} | {len(c.lost_to_gate_denial)}/{c.total} "
        f"({_pct(c.denial_rate)}) | {len(c.lost_to_termination)}/{c.total} "
        f"({_pct(c.termination_rate)}) | "
        f"{len(c.lost_to_termination_sev2)}/{c.total} |",
        "",
        (
            "Every attack session that trips a SEV1 takes"
            if c.lost_to_termination and len(c.lost_to_termination) == c.total
            else f"{len(c.lost_to_termination)} of the {c.total} attack "
            "sessions that trip a SEV1 take"
        )
        + " the scientist's legitimate job with it, which makes a "
        "deliberately tripped SEV1 a denial-of-service primitive. SEV2 "
        f"routing recovers **{len(c.recovered_by_sev2)} of those "
        f"{len(c.lost_to_termination)}** -- they still wait on "
        "re-authentication, but the session no longer ends."
        + (
            f" It does not recover the other "
            f"**{len(c.lost_to_termination_sev2)}**: their SEV1 comes from a "
            "check the lifecycle-hook knob does not reach, so the primitive "
            "narrows rather than closes."
            if c.lost_to_termination_sev2
            else ""
        ),
        "",
    ]
    if c.lost_to_termination_sev2:
        out += [
            "| Still terminated under SEV2 routing |",
            "|---|",
        ]
        out += [f"| `{iid}` |" for iid in c.lost_to_termination_sev2]
        out.append("")
    out += [
        "## Scope",
        "",
        "- Both halves are deterministic and offline; no served model.",
        "- The exposure reading is over the authored B5 corpus. A session "
        "whose attack continues past its own lifecycle-hook denial is a "
        "shape the corpus would have to contain for this measurement to see "
        "it; the count above is a property of these instances, not of the "
        "threat class.",
        "- The benign figure counts controls blocked at SEV1 and applies the "
        "playbook's documented SEV1 rule offline. No session is actually "
        "terminated in the harness, here or anywhere else in SIEGE.",
    ]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-out", type=Path, default=None)
    args = ap.parse_args(argv)

    result = asyncio.run(run_sev2_routing())
    md = to_markdown(result)
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(md)
        print(f"wrote {args.report_out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
