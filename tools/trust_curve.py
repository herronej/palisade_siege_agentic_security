"""
Trust scorer as a curve, not a point (W7.1 / W7.2 / R1-S11).

Reviewer R1-S11: the trust machinery is justified by a single scenario -- one
high-stakes capability (G5), one strike, and a single "10 benign probes" point
in Table E6. This driver replaces the point with the full picture:

* **Recovery curve (W7.1).** Sweep the probe count contiguously and record
  whether the high-stakes strike lands after each length of benign probing. The
  decaying-trust design re-opens the floored capability at an *exact* crossover
  (the single "10" was a coarse-grid artifact); the sticky scorer never re-opens
  it at any probe count. Reported per high-stakes capability (G5 above-ceiling,
  G6 destructive, G3 CUI), so the result is not a single-capability artifact.

* **Cross-capability wash (W7.1).** The recovery lever the decaying design gives
  an attacker is *same-capability* probing; probing a *different* capability
  never re-opens a floored one under either variant, because the posteriors are
  independent. Mapping both axes shows the sticky scorer closes the one lever
  that existed and the other was never a lever.

* **Benign-SEV1 stickiness cost (W7.2).** Stickiness is not free: a benign task
  that trips a *high-stakes* (SEV1) denial floors that capability for the session
  until an explicit re-authentication. We measure that incidence on the W1
  expanded 181-task benign control (the same run as the FPR), with an
  exact-binomial CI and a per-signature breakdown, so the lockout cost is a named,
  tunable number rather than an unstated assumption.

The recovery/wash halves are fully offline and deterministic (they drive the real
``TrustScorer`` directly, the honest C4 test -- the offline ``SessionRunner`` does
not feed the scorer). The benign-SEV1 half runs the deterministic fast-tier stack
over the control. Lives under ``tools/`` because it is a result driver over the
control plane; it changes no gate or trust source.

Run: ``python -m tools.trust_curve``.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import sys
from dataclasses import dataclass, field
from pathlib import Path

from palisade.capabilities.registry import TrustTier
from palisade.config import PalisadeSettings
from siege.redteam.attacks.llm.multiturn_oscillation import MultiTurnOscillation
from palisade.trust import TrustScorer
from tools.benign_fpr import clopper_pearson, measure_benign_fpr
from siege.paths import REPO_ROOT

__all__ = [
    "CapabilityCurve",
    "BenignSev1Report",
    "TrustCurveResult",
    "run_trust_curve",
    "write_results_doc",
]

#: High-stakes capabilities whose denial ratchets a sticky floor (per the trust
#: scorer docstring: G5 above-ceiling, G6 destructive egress, G3 CUI exposure).
_HIGH_STAKES_CAPS: tuple[str, ...] = ("G5", "G6", "G3")

#: The primary capability the E6 curve is drawn for (the flagship HPC sink).
_PRIMARY_CAP = "G5"

#: Contiguous probe counts for the curve, plus coarse points confirming the
#: sticky floor never lifts even after a very long benign lull.
_CURVE_PROBES: tuple[int, ...] = (*range(0, 16), 20, 30, 50, 100, 500, 1000)

#: Upper bound for the exact-crossover search (well past the decaying crossover).
_CROSSOVER_CAP = 4000


@dataclass(frozen=True)
class CapabilityCurve:
    """The recovery curve for one high-stakes capability under one variant."""

    cap: str
    sticky: bool
    per_probe: dict[int, bool]  # probe count -> does the strike land?
    crossover: int | None  # fewest probes that re-open the cap, or None

    @property
    def ever_lands(self) -> bool:
        return self.crossover is not None


@dataclass(frozen=True)
class BenignSev1Report:
    """The cost of stickiness on the benign control (W7.2)."""

    total: int
    blocked: int
    sev1: int
    sev1_ci: tuple[float, float]
    by_gate: dict[str, int]  # gate -> SEV1 count
    by_signature: dict[str, int]  # signature -> SEV1 count
    examples: list[tuple[str, str, str]]  # (instance, gate, reason)

    @property
    def sev1_rate(self) -> float:
        return (self.sev1 / self.total) if self.total else 0.0


@dataclass
class TrustCurveResult:
    primary_cap: str
    sticky_curve: CapabilityCurve
    decaying_curve: CapabilityCurve
    multi_cap: dict[str, tuple[int | None, int | None]]  # cap -> (sticky_x, decaying_x)
    cross_wash: dict[str, bool]  # variant label -> does cross-cap probing re-open?
    benign: BenignSev1Report | None = field(default=None)

    # ---- headline helpers ----
    @property
    def decaying_crossover(self) -> int | None:
        return self.decaying_curve.crossover

    def uniform_multi_cap(self) -> bool:
        """Every high-stakes capability shows the same behavior: sticky never
        re-opens, decaying re-opens at the same crossover."""
        dxs = {d for (_s, d) in self.multi_cap.values()}
        sxs = {s for (s, _d) in self.multi_cap.values()}
        return sxs == {None} and len(dxs) == 1 and None not in dxs

    def to_markdown(self) -> str:
        lines: list[str] = [
            "# Trust scorer: a curve, not a point (W7)",
            "",
            "The single Table E6 row (\"10 benign probes re-open the capability\") "
            "replaced by the full recovery surface: the probe count swept "
            "contiguously, every high-stakes capability, both recovery levers, and "
            "the benign-SEV1 cost of stickiness.",
            "",
            "## Recovery curve (primary capability "
            f"`{self.primary_cap}`)",
            "",
            "Does a high-stakes strike land after *k* benign probe calls rebuild "
            "trust? The decaying-trust design re-opens the floored capability at an "
            "exact crossover; the sticky scorer never does.",
            "",
            "| benign probes *k* | decaying: strike lands | sticky (PALISADE): strike lands |",
            "|---:|:--:|:--:|",
        ]
        probes = sorted(self.decaying_curve.per_probe)
        for k in probes:
            d = "yes" if self.decaying_curve.per_probe[k] else "no"
            s = "yes" if self.sticky_curve.per_probe[k] else "**no**"
            lines.append(f"| {k} | {d} | {s} |")
        dxo = self.decaying_crossover
        lines += [
            "",
            f"**Decaying crossover: {dxo} benign probes** (the single \"10\" in the "
            "old E6 was the coarse `(0,1,3,10,...)` grid rounding this up). "
            "**Sticky: never re-opens** across every swept probe count, including a "
            "1000-probe lull. Only an explicit re-authentication clears the floor.",
            "",
            "## Every high-stakes capability, not just one",
            "",
            "The same sweep for each capability whose denial is sticky "
            "(G5 above-ceiling, G6 destructive egress, G3 CUI exposure).",
            "",
            "| capability | decaying crossover | sticky (PALISADE) |",
            "|---|:--:|:--:|",
        ]
        for cap in _HIGH_STAKES_CAPS:
            sx, dx = self.multi_cap[cap]
            sx_s = "never" if sx is None else str(sx)
            dx_s = "never" if dx is None else f"{dx} probes"
            lines.append(f"| {cap} | {dx_s} | **{sx_s}** |")
        lines += [
            "",
            f"Uniform across capabilities: {self.uniform_multi_cap()} "
            "(decaying re-opens each at the same crossover; sticky re-opens none).",
            "",
            "## Both recovery levers",
            "",
            "Same-capability probing is the lever the decaying design hands the "
            "attacker (above). Cross-capability probing -- flooring one capability "
            "and running clean calls on a *different* one -- is not a lever under "
            "either variant, because the per-capability posteriors are independent.",
            "",
            "| recovery lever | decaying | sticky (PALISADE) |",
            "|---|:--:|:--:|",
            f"| same-capability probing | re-opens at {dxo} probes | **never** |",
            f"| cross-capability probing | {'re-opens' if self.cross_wash.get('decaying') else 'never'} "
            f"| {'re-opens' if self.cross_wash.get('sticky') else '**never**'} |",
            "",
            "So sticky-high-stakes closes the one recovery window that existed; the "
            "cross-capability window was never open.",
            "",
        ]
        if self.benign is not None:
            b = self.benign
            lines += [
                "## Cost of stickiness: benign-SEV1 lockouts (W7.2)",
                "",
                "A benign task that trips a *high-stakes* (SEV1) denial floors that "
                "capability for the session until re-authentication. Measured on "
                "the W1 expanded 181-task benign control (the same fast-tier run as "
                "the FPR):",
                "",
                f"**Benign-SEV1 incidence: {b.sev1}/{b.total} = "
                f"{b.sev1_rate * 100:.1f}%** (exact-binomial 95% CI "
                f"[{b.sev1_ci[0] * 100:.1f}%, {b.sev1_ci[1] * 100:.1f}%]). Of the "
                f"{b.blocked} benign tasks blocked in all, {b.sev1} are SEV1 "
                f"(sticky-flooring) and {b.blocked - b.sev1} lower-severity.",
                "",
                "| gate | benign SEV1 blocks | signature |",
                "|---|---:|---|",
            ]
            sig_by_gate: dict[str, list[str]] = collections.defaultdict(list)
            for sig, _n in b.by_signature.items():
                # signature is "G<n> <name>"; group display under its gate
                g = sig.split(" ", 1)[0]
                sig_by_gate[g].append(sig)
            for gate in sorted(b.by_gate):
                sigs = "; ".join(f"`{s}`" for s in sig_by_gate.get(gate, []))
                lines.append(f"| {gate} | {b.by_gate[gate]} | {sigs} |")
            lines += [
                "",
                "These are the same conservative signatures the FPR analysis "
                "already names as tunable policy knobs (G1 weaponization "
                "over-match on legitimate non-proliferation science; G5 lifecycle "
                "-hook over-match on legitimate prolog/epilog jobs). Stickiness "
                "escalates each from a single blocked action to a session-level "
                "capability lockout until re-auth -- the deployment cost of "
                "removing the probe-then-strike window, and it moves with the same "
                "knobs that set the FPR.",
                "",
            ]
        return "\n".join(lines)


def _capability_curve(cap: str, *, sticky: bool) -> CapabilityCurve:
    """The recovery curve for one capability/variant, plus the exact crossover."""
    osc = MultiTurnOscillation(high_stakes_cap=cap)
    per_probe = {
        k: osc.run_oscillation(recovery_windows=(k,), sticky=sticky).per_window[k]
        for k in _CURVE_PROBES
    }
    crossover = _exact_crossover(osc, sticky=sticky)
    return CapabilityCurve(cap=cap, sticky=sticky, per_probe=per_probe, crossover=crossover)


def _exact_crossover(osc: MultiTurnOscillation, *, sticky: bool) -> int | None:
    """Fewest contiguous benign probes after which the strike lands, or None if
    it never lands up to the search cap (the sticky case)."""
    # Scan upward from 0; the landing region is monotone (clean calls only raise
    # the posterior), so the first landing k is the crossover.
    lo, hi = 0, _CROSSOVER_CAP
    if not osc.run_oscillation(recovery_windows=(hi,), sticky=sticky).per_window[hi]:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if osc.run_oscillation(recovery_windows=(mid,), sticky=sticky).per_window[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _cross_capability_lands(
    *, struck: str, probe_cap: str, sticky: bool, probes: int = 1000
) -> bool:
    """Floor ``struck`` with a high-stakes SEV1, run ``probes`` clean calls on a
    *different* capability, and report whether ``struck`` is back at NORMAL."""
    scorer = TrustScorer(PalisadeSettings(enabled=True, sticky_high_stakes=sticky))
    scorer.record_violation(1, capability_kind=struck, high_stakes=True)
    for _ in range(probes):
        scorer.record_clean_call(probe_cap)
    return scorer.current_tier_for(struck) is TrustTier.NORMAL


async def _measure_benign_sev1() -> BenignSev1Report:
    """Benign-SEV1 incidence on the W1 expanded control (fast-tier full stack)."""
    report = await measure_benign_fpr()
    outcomes = report.outcomes
    total = len(outcomes)
    blocked = [o for o in outcomes if o.blocked]
    sev1 = [o for o in blocked if o.incident_level == 1]
    by_gate = collections.Counter(o.gate or "—" for o in sev1)
    by_sig = collections.Counter(o.signature for o in sev1)
    examples = [(o.instance_id, o.gate or "—", o.reason) for o in sev1]
    return BenignSev1Report(
        total=total,
        blocked=len(blocked),
        sev1=len(sev1),
        sev1_ci=clopper_pearson(len(sev1), total),
        by_gate=dict(sorted(by_gate.items())),
        by_signature=dict(sorted(by_sig.items(), key=lambda kv: (-kv[1], kv[0]))),
        examples=examples,
    )


def run_trust_curve(*, with_benign: bool = True) -> TrustCurveResult:
    """Build the full recovery surface and (optionally) the benign-SEV1 cost."""
    sticky_curve = _capability_curve(_PRIMARY_CAP, sticky=True)
    decaying_curve = _capability_curve(_PRIMARY_CAP, sticky=False)
    multi_cap = {
        cap: (
            _capability_curve(cap, sticky=True).crossover,
            _capability_curve(cap, sticky=False).crossover,
        )
        for cap in _HIGH_STAKES_CAPS
    }
    cross_wash = {
        "sticky": _cross_capability_lands(
            struck="G5", probe_cap="G2", sticky=True
        ),
        "decaying": _cross_capability_lands(
            struck="G5", probe_cap="G2", sticky=False
        ),
    }
    benign = asyncio.run(_measure_benign_sev1()) if with_benign else None
    return TrustCurveResult(
        primary_cap=_PRIMARY_CAP,
        sticky_curve=sticky_curve,
        decaying_curve=decaying_curve,
        multi_cap=multi_cap,
        cross_wash=cross_wash,
        benign=benign,
    )


def write_results_doc(result: TrustCurveResult, path: str | None = None) -> str:
    if path is None:
        root = REPO_ROOT
        path = str(root / "docs" / "palisade" / "trust_curve_w7.md")
    Path(path).write_text(result.to_markdown(), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(
        description="Trust scorer as a curve (recovery sweep + multi-capability + "
        "cross-capability wash) plus the benign-SEV1 stickiness cost (W7)."
    )
    parser.add_argument(
        "--report-out", default=None, metavar="PATH",
        help="Write the markdown report to PATH (also printed).",
    )
    parser.add_argument(
        "--no-benign", action="store_true",
        help="Skip the benign-SEV1 control run (recovery-curve only).",
    )
    args = parser.parse_args(argv)
    result = run_trust_curve(with_benign=not args.no_benign)
    md = result.to_markdown()
    if args.report_out:
        Path(args.report_out).write_text(md, encoding="utf-8")
        print(f"[trust-curve] wrote report to {args.report_out}")
    print("\n" + md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
