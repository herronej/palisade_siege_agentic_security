"""
Taint-propagation coverage study: L̂ vs L* (W2.3).

The capability bound's guarantee is only as good as the label it carries to the
sink. This tool measures, over the whole corpus, how often the **implemented**
carried label L̂(v) disagrees with the **corpus-authored oracle** label L*(v) at
the two capability sinks -- code (G4) and HPC submit (G5). Each disagreement is a
value that *is* untrusted (L*=tainted) but that the runtime would *not* recognize
as such (L̂=untainted) once it has been transformed -- i.e. a propagation gap, the
true residual the manuscript's 2% hard-win floor rests on.

Definitions, per sink value ``v`` reaching a G4/G5 tool call:

* **L*(v) -- oracle.** ``v`` is untrusted iff the authored chain makes it so:
  the sink action declares ``taint=True``, or an untrusted (``taint=True``)
  source was read earlier in the same session (the value is downstream of a
  poisoned read). This is the ground truth a corpus author encodes.
* **L̂(v) -- implemented.** What the hardened runtime guard computes from the
  registered source content: exact-string match, or containment/decode
  propagation (``capabilities.propagation``) of a distinctive payload from an
  upstream tainted source into ``v``. No use of the declared taint -- that would
  be circular.

A row is a **gap** when ``L* = tainted`` but ``L̂ = untainted``: the value is
genuinely untrusted, yet rides into the sink unrecognized. Gaps are counted by
gate and by template/transform, so the residual is attributable (textual-flow
laundering that containment closes vs. the freshly-authored / tag-drop residual
it cannot).

    cd backend
    uv run python -m tools.taint_coverage
    uv run python -m tools.taint_coverage --report-out docs/palisade/taint_coverage.md
"""

from __future__ import annotations

import argparse
import collections
import sys
from dataclasses import dataclass
from pathlib import Path

from palisade.capabilities.propagation import shares_distinctive_content
from siege import load_instances
from siege.corpus_builder import CORPUS_DIR
from siege.schemas import Action, ActionKind, Instance

#: The two capability sinks the propagation label governs.
_TAINT_SINKS = ("G4", "G5")


def _value_string(action: Action) -> str:
    """The content string an action carries -- the tainted payload for a source,
    the argument for a sink. Covers the shapes the corpus uses."""
    p = action.payload if isinstance(action.payload, dict) else {}
    # Retrieval / memory: the chunk / record text.
    for k in ("query", "content"):
        if isinstance(p.get(k), str):
            return p[k]
    # HPC submit: the script.
    if isinstance(p.get("slurm_script"), str):
        return p["slurm_script"]
    # Tool call: the code/command argument.
    args = p.get("args")
    if isinstance(args, dict):
        for k in ("content", "command", "cmd", "code", "script"):
            if isinstance(args.get(k), str):
                return args[k]
    return ""


def _sink_gate(action: Action) -> str | None:
    if action.kind is not ActionKind.TOOL_CALL:
        return None
    g = action.resolved_gate()
    return g if g in _TAINT_SINKS else None


@dataclass(frozen=True)
class CoverageRow:
    instance_id: str
    template: str
    axis: str
    gate: str
    l_star: bool  # oracle: value is genuinely untrusted
    l_hat: bool  # implemented: containment/exact recognizes it as tainted

    @property
    def gap(self) -> bool:
        return self.l_star and not self.l_hat


def collect_rows(instances: list[Instance]) -> list[CoverageRow]:
    rows: list[CoverageRow] = []
    for inst in instances:
        for session in inst.sessions:
            tainted_sources: list[str] = []
            for turn in session.turns:
                for action in turn.actions:
                    src = _value_string(action)
                    declared_taint = (
                        action.capability is not None and action.capability.taint
                    )
                    gate = _sink_gate(action)
                    if gate is not None:
                        sink_str = _value_string(action)
                        # L*: the oracle is the author's ground truth -- a value
                        # is untrusted iff it declares taint, or it is the attack
                        # payload (``is_attack``) in a chain that read a poison.
                        # A benign co-session action (``is_utility``) is NOT
                        # untrusted just because a poison was read earlier.
                        l_star = declared_taint or (
                            action.is_attack and bool(tainted_sources)
                        )
                        # L̂: exact or containment match against upstream sources.
                        l_hat = any(
                            sink_str == s or shares_distinctive_content(sink_str, s)
                            for s in tainted_sources
                        )
                        rows.append(
                            CoverageRow(
                                instance_id=inst.instance_id,
                                template=inst.template,
                                axis=action.label or inst.variation_axis or "",
                                gate=gate,
                                l_star=l_star,
                                l_hat=l_hat,
                            )
                        )
                    # Record this action as a tainted *source* AFTER the sink
                    # check (a source poisons only the actions that follow it).
                    # Only untrusted *reads* are sources -- a high-privilege sink
                    # (G4/G5) is a destination, not a source, so its own script
                    # must not poison a sibling benign job that shares cp2k
                    # boilerplate.
                    if declared_taint and src and gate is None:
                        tainted_sources.append(src)
    return rows


def render(rows: list[CoverageRow]) -> str:
    sink_rows = [r for r in rows if r.l_star]  # only untrusted values can gap
    n = len(sink_rows)
    gaps = [r for r in sink_rows if r.gap]
    by_gate = collections.Counter(r.gate for r in sink_rows)
    gap_by_gate = collections.Counter(r.gate for r in gaps)
    by_tmpl = collections.Counter(r.template for r in sink_rows)
    gap_by_tmpl = collections.Counter(r.template for r in gaps)
    # Precision check: a trusted value (L*=untainted) that L̂ wrongly marks
    # tainted -- a false propagation. Must be 0 for the bound to stay FPR-safe.
    false_props = [r for r in rows if (not r.l_star) and r.l_hat]

    lines = [
        "# Taint-propagation coverage: L̂ vs L* (W2.3)",
        "",
        "Over every genuinely-untrusted value (L*=tainted) reaching a capability "
        "sink (G4 code, G5 HPC submit) in the corpus, how often the implemented "
        "carried label L̂ recognizes it. A **gap** (L*=tainted, L̂=untainted) is a "
        "value that rides into the sink unrecognized -- the propagation residual.",
        "",
        f"**Untrusted values reaching a sink: {n}. Recognized by L̂: {n - len(gaps)}. "
        f"Gaps (L̂≠L*): {len(gaps)} ({len(gaps) / n * 100:.0f}%). "
        f"False propagations (L*=untainted, L̂=tainted): {len(false_props)}.**"
        if n
        else "**No untrusted values reach a sink in this corpus.**",
        "",
        "## By sink gate",
        "",
        "| gate | untrusted-at-sink | recognized (L̂=T) | gaps | gap rate |",
        "|---|---|---|---|---|",
    ]
    for g in _TAINT_SINKS:
        tot = by_gate.get(g, 0)
        gp = gap_by_gate.get(g, 0)
        rate = f"{gp / tot * 100:.0f}%" if tot else "--"
        lines.append(f"| {g} | {tot} | {tot - gp} | {gp} | {rate} |")
    lines += [
        "",
        "## By template (untrusted-at-sink values)",
        "",
        "| template | untrusted-at-sink | gaps |",
        "|---|---|---|",
    ]
    for t in sorted(by_tmpl, key=lambda t: (-gap_by_tmpl.get(t, 0), t)):
        lines.append(f"| `{t}` | {by_tmpl[t]} | {gap_by_tmpl.get(t, 0)} |")
    lines += [
        "",
        "## Every gap (the residual, attributable)",
        "",
        "| instance | gate | label |",
        "|---|---|---|",
    ]
    for r in gaps:
        lines.append(f"| `{r.instance_id}` | {r.gate} | {r.axis} |")
    lines += [
        "",
        "Gaps are the values containment cannot close: a sink whose content is "
        "freshly authored and only *semantically* references the untrusted "
        "instruction (no distinctive payload flows in), or a pure tag-drop chain "
        "whose sink shares no text with the poisoned read. Both need dataflow "
        "lineage the interpreter does not observe; they are the characterized "
        "residual, not verbatim-reuse coverage.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Taint-propagation coverage study (L̂ vs L*) over the corpus."
    )
    parser.add_argument("--report-out", default=None, metavar="PATH")
    args = parser.parse_args(argv)

    rows = collect_rows(load_instances(CORPUS_DIR))
    md = render(rows)
    if args.report_out:
        Path(args.report_out).write_text(md, encoding="utf-8")
        print(f"[taint-coverage] wrote {args.report_out}")
    print("\n" + md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
