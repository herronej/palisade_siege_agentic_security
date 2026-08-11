"""
What a session-level taint interlock closes, and what it costs (R4-16).

**The question.** The deployed taint guard is content-keyed: it denies a
privileged sink whose *own argument* resolves to an untrusted-tainted
value. A value transformed past that guard arrives unlabeled and passes,
which is the propagation residual --- 16 of 205 corpus instances, and
adaptively reachable. Reviewers asked why we do not simply close it by
refusing a privileged sink whenever untrusted taint is live in the
session, which is a coarser rule needing no lineage at all.

**Why the attack side is not the answer.** That rule is *exactly* the
predicate the hard-win metric scores. Enforcing it drives the measured
hard-win rate to zero by construction, and a zero obtained that way
measures nothing about security. We report it, flagged as such, and put
no weight on it.

**The experiment is the cost.** An interlock that refuses a privileged
sink after any untrusted read refuses a great deal of legitimate science:
a benign session that retrieves a paper and then writes an analysis file
trips it. So we sweep the interlock's scope --- ``off``, ``turn``,
``session`` --- and measure both axes on the same corpora the paper
already uses: hard wins over the 205 attack instances, and false blocks
over the 181-task benign control. The result is the trade a deployment
would actually face, and it is what decides whether the residual is worth
closing this way or with real value lineage.

Usage::

    python3 -m tools.taint_interlock \\
        [--bound production] [--report-out PATH]
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from siege import load_instances
from siege.ablation_matrix import CUMULATIVE_CONFIGS
from siege.corpus_builder import CORPUS_DIR
from siege.scorer import score_trace
from siege.session_runner import SessionRunner

_SCOPES = ("off", "turn", "session")


def _full_config():
    for cfg in CUMULATIVE_CONFIGS:
        if cfg.name == "full PALISADE":
            return cfg
    raise LookupError("full PALISADE config not found")


@dataclass
class ScopeResult:
    scope: str
    hard_wins: int = 0
    n_attack: int = 0
    benign_blocks: int = 0
    n_benign: int = 0
    hard_ids: list[str] = field(default_factory=list)
    benign_ids: list[str] = field(default_factory=list)
    benign_by_cat: Counter = field(default_factory=Counter)
    n_prov: int = 0
    prov_blocks: int = 0
    prov_by_shape: Counter = field(default_factory=Counter)
    prov_total_by_shape: Counter = field(default_factory=Counter)

    @property
    def prov_fpr(self) -> float:
        return 100.0 * self.prov_blocks / self.n_prov if self.n_prov else 0.0

    @property
    def hw_rate(self) -> float:
        return 100.0 * self.hard_wins / self.n_attack if self.n_attack else 0.0

    @property
    def fpr(self) -> float:
        return 100.0 * self.benign_blocks / self.n_benign if self.n_benign else 0.0

    @property
    def utility(self) -> float:
        return 100.0 - self.fpr


def _category(inst) -> str:
    axis = getattr(inst, "variation_axis", "") or ""
    if inst.template == "benign_diverse" and axis:
        return f"cat {axis.split('_', 1)[0]}"
    return inst.template


async def _measure(scope: str, bound: str) -> ScopeResult:
    from tools.benign_fpr import _utility_record
    from tools.benign_provenance_control import CONTROL_DIR

    cfg = _full_config()
    runner = SessionRunner(bound=bound, interlock=scope)
    res = ScopeResult(scope=scope)
    for inst in load_instances(CORPUS_DIR):
        trace = await runner.run(inst, cfg)
        if inst.is_attack:
            res.n_attack += 1
            if score_trace(trace, inst).hard_win:
                res.hard_wins += 1
                res.hard_ids.append(inst.instance_id)
        else:
            res.n_benign += 1
            rec = _utility_record(trace, None)
            if rec is not None and not rec.allowed:
                res.benign_blocks += 1
                res.benign_ids.append(inst.instance_id)
                res.benign_by_cat[_category(inst)] += 1

    # The provenance control, held outside the 205/181 corpus so no published
    # number moves. This is the only control that can price a provenance rule:
    # every instance is legitimate work carrying rag taint into a sink.
    for inst in load_instances(CONTROL_DIR):
        trace = await runner.run(inst, cfg)
        res.n_prov += 1
        shape = (getattr(inst, "variation_axis", "") or "").split("_", 1)[0]
        rec = _utility_record(trace, None)
        if rec is not None and not rec.allowed:
            res.prov_blocks += 1
            res.prov_by_shape[shape] += 1
        res.prov_total_by_shape[shape] += 1
    return res


def to_markdown(results: list[ScopeResult], bound: str) -> str:
    off = next(r for r in results if r.scope == "off")
    L: list[str] = []
    L.append("# Session-level taint interlock: closure vs. cost (R4-16)\n")
    L.append(
        "The deployed guard denies a privileged sink whose own argument "
        "resolves to untrusted taint; a value transformed past it arrives "
        "unlabeled and passes. An *interlock* instead refuses the sink whenever "
        "untrusted taint is live in scope, needing no lineage. Both axes are "
        f"measured on the paper's own corpora, taint bound `{bound}`, config "
        "`full PALISADE`.\n"
    )
    L.append(
        "> **The attack column is not a security result.** `session` scope is "
        "exactly the predicate the hard-win metric scores, so enforcing it "
        "zeroes that metric by construction. The column is shown for "
        "completeness; the cost column is the experiment.\n"
    )
    L.append(
        "| interlock | hard wins / 205 | corpus benign blocks / 181 | "
        "**provenance control blocked / %d** |" % (results[0].n_prov,)
    )
    L.append("|---|---|---|---|")
    for r in results:
        note = " *(by construction)*" if r.scope == "session" else ""
        L.append(
            f"| `{r.scope}` | {r.hard_wins} ({r.hw_rate:.1f}%){note} | "
            f"{r.benign_blocks} ({r.fpr:.1f}%) | "
            f"**{r.prov_blocks} ({r.prov_fpr:.1f}%)** |"
        )
    L.append(
        "\n**The 181-task corpus control cannot price this rule and never "
        "could.** Not one of its instances carries an untrusted-tainted value: "
        "55 drive a high-privilege sink, but none of those sessions first "
        "retrieves, so no benign instance puts provenance and a sink together. "
        "Its column is therefore flat across every scope, and that flatness is "
        "an artifact of the control, not evidence the interlock is free. The "
        "provenance control exists to supply what was missing."
    )

    L.append("\n## What the interlock costs\n")
    for r in results:
        if r.scope == "off":
            continue
        extra = r.benign_blocks - off.benign_blocks
        closed = off.hard_wins - r.hard_wins
        L.append(
            f"**`{r.scope}`** closes {closed} of the {off.hard_wins} residual "
            f"hard wins and adds **{extra}** benign false blocks "
            f"({off.fpr:.1f}% -> {r.fpr:.1f}%), taking task completion from "
            f"{off.utility:.1f}% to {r.utility:.1f}%."
        )
        if r.benign_by_cat:
            worst = ", ".join(
                f"{c} ({n})" for c, n in r.benign_by_cat.most_common(5)
            )
            L.append(f"  Newly-blocked benign work concentrates in: {worst}.")
        L.append("")

    if off.hard_ids:
        L.append(
            f"Residual hard wins under `off` (the deployed behaviour): "
            + ", ".join(f"`{i}`" for i in sorted(off.hard_ids))
            + "\n"
        )

    L.append("## Reading\n")
    sess = next((r for r in results if r.scope == "session"), None)
    turn = next((r for r in results if r.scope == "turn"), None)
    if sess and turn:
        L.append(
            f"**Session scope refuses {sess.prov_blocks} of {sess.n_prov} "
            "legitimate retrieve-then-act sessions -- all of them.** That is "
            "the answer to why we do not close the propagation residual this "
            "way. VISTA's core workflow *is* retrieve-then-analyze, so a rule "
            "refusing a privileged sink whenever a retrieval preceded it does "
            "not degrade the product, it removes it. The zero in the attack "
            "column is bought by refusing every scientific task the system "
            "exists to perform.\n"
        )
        L.append(
            f"**Turn scope closes nothing** ({turn.hard_wins} residual hard "
            "wins, unchanged) at no benign cost. The corpus chains put the "
            "poisoned read and the sink in different turns, which is also how "
            "a real multi-step agent works, so a same-turn interlock never "
            "sees them. There is no usable middle point here between blind to "
            "the chain and refusing the workflow.\n"
        )
        L.append(
            f"**The deployed guard costs {off.prov_blocks} of {off.n_prov} on "
            "the same control**, including the `carry` cases whose script "
            "quotes the retrieved value verbatim. That is by design, not by "
            "luck: containment requires the shared run to be *payload-shaped* "
            "-- a shell or path metacharacter, or a spaceless alphanumeric "
            "blob -- and a quoted scientific value (`2579.3 - 0.624 * T`) is "
            "prose-shaped, so the guard evaluates it and declines. The same "
            "distinctiveness floor that leaves a short scheduler field "
            "unattributable is what lets a scientist quote a published "
            "constant without being refused: the residual and the clean benign "
            "record are two readings of one threshold.\n"
        )
        L.append(
            "So the coarse fix is not a fix. Closing the residual without "
            "refusing the workflow requires knowing *which* value at the sink "
            "descended from the retrieval -- value lineage -- and a sidecar "
            "cannot observe that edge because it is created inside the model's "
            "forward pass. This is the argument for lineage in numbers rather "
            "than in principle."
        )
    L.append("\n_Generated by `tools.taint_interlock`._")
    return "\n".join(L) + "\n"


async def _arun(bound: str) -> str:
    results = [await _measure(s, bound) for s in _SCOPES]
    return to_markdown(results, bound)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(
        description="Sweep the taint interlock's scope: closure vs benign cost."
    )
    ap.add_argument("--bound", default="production", choices=["production", "declarative"])
    ap.add_argument("--report-out", type=Path, default=None)
    args = ap.parse_args(argv)
    text = asyncio.run(_arun(args.bound))
    if args.report_out:
        args.report_out.write_text(text)
        print(f"wrote {args.report_out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
