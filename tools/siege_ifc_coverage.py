"""SIEGE vs runnable IFC/guardrail baselines: the sink-coverage money-plot (W4.3 / R1-M5).

Reviewer R1-M5 wants a head-to-head against a runnable IFC baseline. The decisive
comparison is not a utility bake-off on AgentDojo (where a frontier model already
resists the injection unaided) but a **structural** one: *which SIEGE sinks does
each baseline's published design even model?* A defense cannot close a hard win
at a sink it does not define.

This tool grounds that comparison in the corpus. It scores the corpus undefended,
collects every undefended hard win, and classifies the sink each one reaches into
a small set of **sink types**. It then crosses those sink types against the
published enforcement surface of each baseline -- the deployed/runnable IFC and
guardrail defenses the paper compares to -- and reports, per sink type, whether
**any** baseline models it.

The headline it produces: two SIEGE sink types -- the **HPC scheduler submission**
(``b5_11``: a tainted value setting an ``account``/``partition``/``qos``/
``reservation``/``dependency``/``output_path`` field of an otherwise-legitimate
``submit_hpc_job``) and the **scientific-citation / reference-value egress**
(``b3_4``: a forged citation reaching the answer, the G6 grounding sink) -- exist
in **no** baseline's environment. AgentDojo, the benchmark CaMeL, FIDES, and
Progent were built and evaluated on, has email/banking/travel/slack tool sinks and
no scheduler or citation sink at all. So the hard wins reaching those two sinks
fall outside every baseline's surface **by construction** (they pass without being
evaded), and PALISADE's G5 (HPC) and G6 (grounding) predicates are what close them.

The claim is deliberately narrow and checkable: it is about which sinks a
published design *defines*, not a runtime horse-race. On the sink types the
baselines do model (cross-boundary tool-call dataflow), a dataflow-IFC defense
like CaMeL is a structural peer, not a loser -- PALISADE's contribution there is to
carry the same discipline to the two sinks the baselines never modeled. The
runnable head-to-head that *does* need a served base model (PALISADE and a
least-privilege tool-filter baseline live in AgentDojo) is wired in
``tools.agentdojo_e1``; this module is its offline, model-independent complement.

    cd backend
    uv run python -m tools.siege_ifc_coverage
    uv run python -m tools.siege_ifc_coverage \
        --report-out ../docs/palisade/siege_ifc_coverage_w4.md
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from siege.instance_loader import load_instances
from siege.scorer import score_trace
from siege.session_runner import SessionRunner
from tools.baseline_hardwin import _OFF, _class_key
from palisade.paths import CORPUS_DIR, REPO_ROOT

__all__ = [
    "SinkType",
    "Baseline",
    "SINK_TYPES",
    "BASELINES",
    "CoverageResult",
    "run_ifc_coverage",
    "write_results_doc",
]

_CORPUS = CORPUS_DIR


# =================================================================
# SIEGE sink types (the class -> sink map is verified against the corpus at run
# time; ``agentdojo_has_it`` records whether the baselines' home benchmark even
# defines a sink of this type).
# =================================================================


@dataclass(frozen=True)
class SinkType:
    """A class of privileged sink a hard win can reach."""

    id: str
    label: str
    #: SIEGE hard-win classes that reach this sink (checked against the corpus).
    classes: tuple[str, ...]
    #: Does AgentDojo -- the benchmark CaMeL/FIDES/Progent were built on -- define
    #: a sink of this type at all? If not, no baseline can enforce at it.
    agentdojo_has_it: bool
    note: str


SINK_TYPES: tuple[SinkType, ...] = (
    SinkType(
        id="hpc_scheduler",
        label="HPC scheduler submission field",
        classes=("b5_11",),
        agentdojo_has_it=False,
        note="a tainted value sets a submit_hpc_job field (partition/qos/"
        "reservation/dependency/output_path); the submit tool is legitimately "
        "needed, so a tool-level allow/deny cannot close it -- only a field-level "
        "provenance read at the scheduler sink does (PALISADE G5).",
    ),
    SinkType(
        id="science_egress",
        label="scientific-citation / reference-value egress",
        classes=("b3_4",),
        agentdojo_has_it=False,
        note="a forged citation reaches the answer (the G6 grounding sink); no "
        "baseline environment defines a citation-provenance or reference-value "
        "contract sink.",
    ),
    SinkType(
        id="cross_boundary_write",
        label="cross-boundary tool-call dataflow to a write/exfil sink",
        classes=("xc_1", "xc_4", "b3_9"),
        agentdojo_has_it=True,
        note="a tainted value laundered across a boundary into a create_file / "
        "exfil-write sink; a dataflow-IFC defense (CaMeL/FIDES) models this sink "
        "type in its own environment.",
    ),
    SinkType(
        id="code_exec",
        label="untrusted code into a sandbox exec sink",
        classes=("b4_9",),
        agentdojo_has_it=True,
        note="corpus-supplied code copied into the sandbox and executed; IFC over "
        "the code tool models the data->exec flow.",
    ),
    SinkType(
        id="attached_content",
        label="forged instruction in an uploaded artifact -> tool sink",
        classes=("b1_9",),
        agentdojo_has_it=True,
        note="a forged instruction inside an uploaded file drives a credential-"
        "read tool call; a detection tier may flag the injection text.",
    ),
    SinkType(
        id="forged_context",
        label="forged prior-approval / trust context -> write sink",
        classes=("xc_5",),
        agentdojo_has_it=False,
        note="a forged prior approval (assistant turn / tool result / user "
        "consent) authorizes a write; closed by PALISADE's untrusted-context "
        "quarantine + sticky trust scorer, which the baselines do not model.",
    ),
)


# =================================================================
# The baselines: the enforcement surface of each published design. ``models`` is
# the set of sink-type ids the design *defines* a check at -- a narrow, checkable
# claim about the published mechanism, not a runtime result.
# =================================================================


@dataclass(frozen=True)
class Baseline:
    """A runnable/deployed baseline and the sink types its design models."""

    name: str
    family: str  # "capability/IFC" | "least-privilege" | "detection-only"
    cite: str
    runnable_here: str  # how/whether it runs in this repo
    models: frozenset[str]
    note: str


BASELINES: tuple[Baseline, ...] = (
    Baseline(
        name="CaMeL",
        family="capability/IFC (dataflow)",
        cite="Debenedetti et al. 2025",
        runnable_here="external repo + served base model (not wired; see agentdojo_e1 for the runnable slot)",
        models=frozenset(
            {"cross_boundary_write", "code_exec", "attached_content", "forged_context"}
        ),
        note="tracks value provenance through the agent's tool calls and refuses "
        "a sink fed by untrusted data (a forged approval is untrusted data too, so "
        "xc_5 is in scope). But its sinks are its environment's: no HPC-scheduler "
        "submission field and no citation/reference-value sink exist to enforce at.",
    ),
    Baseline(
        name="FIDES",
        family="capability/IFC (deterministic labels)",
        cite="Debenedetti et al. 2025",
        runnable_here="external repo + served base model (not wired)",
        models=frozenset(
            {"cross_boundary_write", "code_exec", "attached_content", "forged_context"}
        ),
        note="deterministic information-flow labels on tool data; same "
        "environment scoping as CaMeL -- no scheduler or citation sink.",
    ),
    Baseline(
        name="Progent",
        family="least-privilege (tool-call policy DSL)",
        cite="Shi et al. 2025",
        runnable_here="external repo + served base model (not wired)",
        models=frozenset(),
        note="allow/deny at TOOL (and, by hand-authored argument predicate, "
        "argument) granularity. But every SIEGE hard win rides a legitimately-"
        "needed tool with a distribution-valid poisoned value (an injected qos, a "
        "laundered create_file argument, a forged citation), and Progent reads no "
        "provenance, so it cannot separate the injected value from the intended "
        "one; it lowers soft-win ASR by denying unneeded tools, but closes none of "
        "these hard wins by construction.",
    ),
    Baseline(
        name="LlamaFirewall / Prompt-Guard",
        family="detection-only",
        cite="Meta 2025",
        runnable_here="RUN (real Prompt-Guard-86M; see detector_baselines_w4)",
        models=frozenset(),
        note="scores text; models no sink at all. Any hard win whose text reads "
        "benign passes, and at a sensitivity that catches the laundered ones its "
        "benign FPR is 67% (W4.1).",
    ),
    Baseline(
        name="Denylist (this work's reimpl.)",
        family="detection-only",
        cite="this work",
        runnable_here="RUN (offline; see detector_baselines_w4)",
        models=frozenset(),
        note="9-pattern surface denylist; models no sink. The 14% baseline the "
        "manuscript reports is this reimplementation.",
    ),
)

#: The IFC/least-privilege baselines (detection-only excluded): the "runnable IFC
#: baseline" family R1-M5 asks for. A sink type outside ALL of these is outside
#: the IFC comparison surface by construction.
_IFC_BASELINES = tuple(b for b in BASELINES if b.family != "detection-only")


# =================================================================
# Result
# =================================================================


@dataclass
class CoverageResult:
    """Undefended hard-win counts per class + the sink-coverage matrix."""

    hw_by_class: dict[str, int] = field(default_factory=dict)
    total_hw: int = 0
    sink_of_class: dict[str, str] = field(default_factory=dict)  # class -> sink id
    unmapped_hw_classes: tuple[str, ...] = ()

    # ---- derived ----
    def hw_for_sink(self, sink_id: str) -> int:
        return sum(
            self.hw_by_class.get(cls, 0)
            for cls in self.sink_of_class
            if self.sink_of_class[cls] == sink_id
        )

    def ifc_models(self, sink_id: str) -> bool:
        return any(sink_id in b.models for b in _IFC_BASELINES)

    def out_of_scope_sinks(self) -> list[SinkType]:
        """Sink types no IFC baseline models -- outside the surface by construction."""
        return [s for s in SINK_TYPES if not self.ifc_models(s.id)]

    def by_construction_hw(self) -> int:
        return sum(self.hw_for_sink(s.id) for s in self.out_of_scope_sinks())

    def to_markdown(self) -> str:
        oos = self.out_of_scope_sinks()
        oos_ids = {s.id for s in oos}
        by_con = self.by_construction_hw()
        lines = [
            "# SIEGE vs runnable IFC/guardrail baselines: sink-coverage money-plot (W4.3)",
            "",
            "A defense cannot close a hard win at a sink its design does not "
            "define. This crosses every undefended SIEGE hard win, grouped by the "
            "sink it reaches, against the published enforcement surface of the "
            "IFC/guardrail baselines. Hard-win counts are scored from the corpus "
            "undefended (seed 42); sink modeling is a checkable property of each "
            "baseline's published mechanism.",
            "",
            f"**Headline: {by_con} of {self.total_hw} undefended hard wins reach a "
            "sink no runnable IFC baseline models** -- the HPC scheduler submission "
            "field and the scientific-citation egress. They fall outside the IFC "
            "comparison surface *by construction* (they pass without being evaded); "
            "PALISADE's G5 and G6 predicates are what close them.",
            "",
            "## Sink types and who models them",
            "",
            "| Sink type | SIEGE classes | hard wins | in AgentDojo? | modeled by an IFC baseline? |",
            "|---|---|---|---|---|",
        ]
        for s in SINK_TYPES:
            n = self.hw_for_sink(s.id)
            ad = "yes" if s.agentdojo_has_it else "**no**"
            mod = "yes" if s.id not in oos_ids else "**no (by construction)**"
            classes = ", ".join(f"`{c}`" for c in s.classes)
            lines.append(f"| {s.label} | {classes} | {n} | {ad} | {mod} |")
        lines += [
            "",
            "## Coverage matrix (does the baseline's design model the sink?)",
            "",
            "| Baseline | family | " + " | ".join(s.id for s in SINK_TYPES) + " |",
            "|---|---|" + "|".join("---" for _ in SINK_TYPES) + "|",
        ]
        # PALISADE row (models every SIEGE sink -- that is the point of the gate set)
        palisade_cells = " | ".join("closes" for _ in SINK_TYPES)
        lines.append(f"| **PALISADE** | capability bound + contracts | {palisade_cells} |")
        for b in BASELINES:
            cells = []
            for s in SINK_TYPES:
                if s.id in b.models:
                    cells.append("models")
                elif b.family == "detection-only":
                    cells.append("text-only")
                else:
                    cells.append("—")
            lines.append(f"| {b.name} | {b.family} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "_`models` = the published design defines a check at this sink type; "
            "`—` = out of scope for that design; `text-only` = a content detector "
            "with no sink model (may still flag on surface text). PALISADE `closes` "
            "every row because G1–G6 define a predicate at each sink; its residual "
            "(the xc_1 propagation gap, W2) is within-scope, not a coverage gap._",
            "",
            "## Sinks outside every IFC baseline (by construction)",
            "",
        ]
        for s in oos:
            lines.append(
                f"- **{s.label}** (`{'`, `'.join(s.classes)}`, "
                f"{self.hw_for_sink(s.id)} hard wins): {s.note}"
            )
        lines += [
            "",
            "## Per-baseline enforcement surface",
            "",
            "| Baseline | family | cite | runnable here | closes (of the SIEGE sinks) |",
            "|---|---|---|---|---|",
        ]
        for b in BASELINES:
            closes = sum(self.hw_for_sink(sid) for sid in b.models)
            closes_str = "text-only (see W4.1)" if b.family == "detection-only" else f"{closes} hard wins"
            lines.append(
                f"| {b.name} | {b.family} | {b.cite} | {b.runnable_here} | {closes_str} |"
            )
        for b in BASELINES:
            lines += ["", f"- **{b.name}** ({b.cite}): {b.note}"]
        if self.unmapped_hw_classes:
            lines += [
                "",
                "> ⚠ Unmapped hard-win classes (add to `SINK_TYPES`): "
                + ", ".join(f"`{c}`" for c in self.unmapped_hw_classes),
            ]
        lines.append("")
        return "\n".join(lines)


async def _score_undefended() -> dict[str, int]:
    """Undefended hard-win count per class, scored from the corpus (seed 42)."""
    runner = SessionRunner()
    hw: dict[str, int] = defaultdict(int)
    for inst in load_instances(_CORPUS):
        if not inst.is_attack:
            continue
        trace = await runner.run(inst, _OFF)
        if score_trace(trace, inst).hard_win:
            hw[_class_key(inst.template)] += 1
    return dict(hw)


def run_ifc_coverage() -> CoverageResult:
    """Score the corpus undefended and build the sink-coverage matrix."""
    hw_by_class = asyncio.run(_score_undefended())
    sink_of_class: dict[str, str] = {}
    for s in SINK_TYPES:
        for cls in s.classes:
            sink_of_class[cls] = s.id
    unmapped = tuple(sorted(c for c in hw_by_class if c not in sink_of_class))
    return CoverageResult(
        hw_by_class=hw_by_class,
        total_hw=sum(hw_by_class.values()),
        sink_of_class=sink_of_class,
        unmapped_hw_classes=unmapped,
    )


def write_results_doc(result: CoverageResult, path: str | None = None) -> str:
    if path is None:
        root = REPO_ROOT
        path = str(root / "docs" / "palisade" / "siege_ifc_coverage_w4.md")
    Path(path).write_text(result.to_markdown(), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(
        description="W4.3: SIEGE vs IFC-baseline sink-coverage money-plot."
    )
    parser.add_argument("--report-out", default=None, metavar="PATH")
    args = parser.parse_args(argv)
    result = run_ifc_coverage()
    if args.report_out:
        print(f"wrote {write_results_doc(result, args.report_out)}")
    print(result.to_markdown())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
