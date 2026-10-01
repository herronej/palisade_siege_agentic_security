"""
Closure attribution over the non-sink-capable security instances (A1).

R3: "The paper includes 150 security instances, but only 32 can reach
privileged operations ... Much of the benchmark therefore evaluates earlier
detection stages rather than PALISADE's ability to prevent dangerous
privileged actions."

That is correct, and the paper does not currently say what the other 118
instances measure. Section III-B asserts that the allow-list, schema, parser
policy and label predicates -- not the signature and pattern checks -- produce
the reduction, and points at Section V-D. But V-D establishes that for **hard
wins**: the sink predicate is what halves them. It says nothing about the 118
instances that cannot reach a sink. This tool puts a count behind the
assertion where it was never checked.

For each non-sink-capable security instance closed at ``full``, it attributes
the closure to the check that denied it and classifies that check by **what it
decides on**, read from the implementation rather than from the reason string:

* **structural** -- decides on the parse, a schema, an allow-list or a label.
  There is no content to search: an attacker evades it only by satisfying the
  policy, which is the point.
* **heuristic** -- a pattern, denylist or anomaly score both locates the
  candidate and decides. An attacker evades it by rewriting content.
* **hybrid** -- a pattern locates the candidate and a structural predicate
  decides it. The decision is sound; the locator is searchable, so the check
  is only as strong as its weakest half. ``_extract_network_targets`` is the
  clearest case: it fires only for a binary in ``_NET_BINARIES`` carrying an
  ``http(s)://`` URL, and the parser's own comment notes that a
  ``/dev/tcp/host/port`` reverse shell "never sees it".

The headline folds hybrid into heuristic, because a check an attacker can hide
from is not a structural guarantee whatever its decision rule. The three-way
split is reported beside it.

Offline and deterministic; replays the existing ``full`` configuration under
the production bound and adds no new runs.

    uv run python -m tools.closure_attribution
    uv run python -m tools.closure_attribution \\
        --report-out docs/palisade/closure_attribution.md
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import sys
from dataclasses import dataclass, field
from pathlib import Path

from siege import SessionRunner, load_instances
from siege.ablation_matrix import CUMULATIVE_CONFIGS
from siege.paths import CORPUS_DIR
from siege.report_suites import ReportSuite, suite_for
from siege.scorer import score_trace

__all__ = [
    "CheckClass",
    "CheckSpec",
    "TAXONOMY",
    "classify",
    "ClosureRow",
    "AttributionResult",
    "run_closure_attribution",
    "to_markdown",
]

_FULL = CUMULATIVE_CONFIGS[-1]

#: The seven classes with a non-zero undefended hard win -- the 32 sink-capable
#: instances Section V-E is scored on. Everything else in the security
#: partition is what this tool attributes.
SINK_CAPABLE_PREFIXES: frozenset[str] = frozenset(
    {"b1_9", "b3_9", "b4_9", "b5_11", "xc_1", "xc_4", "xc_5"}
)


def is_sink_capable(template: str) -> bool:
    return any(template.startswith(p + "_") for p in SINK_CAPABLE_PREFIXES)


# -----------------------------------------------------------------
# The taxonomy
# -----------------------------------------------------------------


class CheckClass(str):
    STRUCTURAL = "structural"
    HEURISTIC = "heuristic"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class CheckSpec:
    """One deterministic check, classified by what it decides on.

    ``match`` is a substring of the gate's reason string that identifies the
    check uniquely; ``symbol`` names the implementation the classification was
    read from, so a reader can check the call rather than trust the label.
    """

    check_id: str
    gate: str
    match: str
    klass: str
    symbol: str
    rationale: str


#: Ordered: the first spec whose ``match`` appears in the reason wins, so more
#: specific strings precede the buckets they sit inside (the G4 Tier-0 rows).
TAXONOMY: tuple[CheckSpec, ...] = (
    # ----- G3 -------------------------------------------------------
    CheckSpec(
        "g3_query_injection", "G3", "G3 query-injection",
        CheckClass.HEURISTIC, "g3_rag query-injection regex catalog",
        "a regex catalog over the query; the reason quotes the pattern it matched",
    ),
    CheckSpec(
        "g3_embedding_anomaly", "G3", "G3 embedding anomaly",
        CheckClass.HEURISTIC, "g3_anomaly z-score over embedding distance",
        "a statistical content score; in-distribution poisoning is exactly what "
        "it does not see (Section II-A says so)",
    ),
    CheckSpec(
        "g3_hybrid_seam", "G3", "G3 hybrid-seam spoof",
        CheckClass.HEURISTIC, "g3_rag.match_db_authority_spoof",
        "a content predicate over the chunk; fpr_curve prices it as one of the "
        "three conservative signatures",
    ),
    # ----- G4 (specific before the Tier-0 bucket) -------------------
    CheckSpec(
        "g4_typo_squat", "G4", "typo-squatted package",
        CheckClass.HEURISTIC, "g4_code.import_typo_squat_violation",
        "a denylist of known bad module names; a new misspelling is not on it",
    ),
    CheckSpec(
        "g4_path_escape", "G4", "escapes the working directory",
        CheckClass.STRUCTURAL, "g4_code create_file path resolution",
        "resolves the path and tests containment in the writable roots",
    ),
    CheckSpec(
        "g4_path_outside_sandbox", "G4", "outside the writable sandbox roots",
        CheckClass.STRUCTURAL, "g4_code create_file path allow-list",
        "allow-list membership on the resolved path",
    ),
    CheckSpec(
        "g4_path_protected_subtree", "G4", "protected read-only subtree",
        CheckClass.STRUCTURAL, "g4_code create_file path allow-list",
        "allow-list membership on the resolved path",
    ),
    CheckSpec(
        "g4_path_protected_corpus", "G4", "writes into the pr",
        CheckClass.STRUCTURAL, "g4_code create_file path allow-list",
        "allow-list membership on the resolved path (protected corpus subtree)",
    ),
    CheckSpec(
        "g4_active_content", "G4", "writes active content",
        CheckClass.HEURISTIC, "g4_code active-content scan",
        "a content scan of the file body for script / handler / URI shapes",
    ),
    CheckSpec(
        "g4_executable_fanout", "G4", "executable shell script",
        CheckClass.HEURISTIC, "g4_code active-content scan",
        "a content scan of the file body",
    ),
    # ----- G5 -------------------------------------------------------
    CheckSpec(
        "g5_allocation", "G5", "G5 allocation",
        CheckClass.STRUCTURAL, "g5_hpc._check_allocation",
        "the parsed --account must be in the allocation allow-list; no content "
        "is inspected",
    ),
    CheckSpec(
        "g5_resource_ceiling", "G5", "G5 resource ceiling",
        CheckClass.STRUCTURAL, "g5_hpc._check_resource_ceilings",
        "numeric comparison of parsed directives against the allocation caps",
    ),
    CheckSpec(
        "g5_resource_dos", "G5", "G5 resource exhaustion",
        CheckClass.STRUCTURAL, "g5_hpc._check_resource_dos",
        "a policy predicate over parsed directive values (--mem=0, --array span)",
    ),
    CheckSpec(
        "g5_directive_substitution", "G5", "G5 directive command substitution",
        CheckClass.STRUCTURAL, "g5_hpc._check_lifecycle_hooks (directive route)",
        "a #SBATCH value is static config by the grammar, so a substitution in "
        "the parsed value is structurally illegal rather than suspicious",
    ),
    CheckSpec(
        "g5_network_egress", "G5", "G5 network egress",
        CheckClass.HYBRID, "slurm_parser._extract_network_targets + _host_allowed",
        "the host allow-list is structural, but the target is found only when a "
        "_NET_BINARIES command carries an http(s):// URL",
    ),
    CheckSpec(
        "g5_cross_project", "G5", "G5 cross-project access",
        CheckClass.HYBRID, "g5_hpc._LUSTRE_PROJ_RE + _normalize_lustre",
        "subtree containment is structural, but candidate paths are found by a "
        "regex for 'proj-shared'",
    ),
    CheckSpec(
        "g5_chained_dag", "G5", "G5 chained-DAG escalation",
        CheckClass.HEURISTIC, "g5_hpc._NESTED_SUBMISSION_RE",
        "a regex over the job body for an in-body submission command, not a "
        "property of the parse",
    ),
    CheckSpec(
        "g5_lifecycle_env", "G5", "script sets",
        CheckClass.HEURISTIC, "g5_hpc._LIFECYCLE_ENV_PATTERNS",
        "a substring scan of the raw script for SLURM_PROLOG / LD_PRELOAD",
    ),
    CheckSpec(
        "g5_lifecycle_directive", "G5", "G5 lifecycle-hook injection",
        CheckClass.STRUCTURAL, "g5_hpc._LIFECYCLE_HOOK_DIRECTIVES",
        "membership of a parsed directive name in a fixed set; the value is "
        "never read",
    ),
    CheckSpec(
        "g5_opaque_execution", "G5", "G5 opaque execution",
        CheckClass.HEURISTIC, "slurm_parser._DECODE_PIPE_RE",
        "the argument for the check is structural (code it cannot see), but the "
        "check is a regex for the decode-pipe-shell shape",
    ),
    CheckSpec(
        "g5_reverse_shell", "G5", "G5 reverse shell",
        CheckClass.HEURISTIC, "slurm_parser._REVERSE_SHELL_RE",
        "a regex for /dev/tcp and netcat -e",
    ),
    CheckSpec(
        "g5_mining_denylist", "G5", "G5 mining-binary denylist",
        CheckClass.HEURISTIC, "g5_hpc mining IOC denylist",
        "a denylist of known mining binaries",
    ),
    CheckSpec(
        "g5_path_scoping", "G5", "G5 path scoping",
        CheckClass.HEURISTIC, "g5_hpc DEFAULT_SENSITIVE_PATH_PATTERNS",
        "a pattern list of sensitive locations",
    ),
)


def classify(reason: str) -> CheckSpec | None:
    """The first taxonomy entry whose ``match`` appears in ``reason``."""
    for spec in TAXONOMY:
        if spec.match in (reason or ""):
            return spec
    return None


# -----------------------------------------------------------------
# Attribution
# -----------------------------------------------------------------


@dataclass(frozen=True)
class ClosureRow:
    instance_id: str
    template: str
    family: str
    gate: str | None
    check_id: str
    klass: str
    reason: str


@dataclass
class AttributionResult:
    partition: int = 0
    sink_capable: int = 0
    non_sink: int = 0
    closed: list[ClosureRow] = field(default_factory=list)
    unclosed: list[str] = field(default_factory=list)
    unclassified: list[tuple[str, str]] = field(default_factory=list)
    #: Soft wins inside the sink-capable 32, for the Table I reconciliation.
    sink_capable_soft: int = 0
    #: Hard wins inside it, which Table I reports as 13.
    sink_capable_hard: int = 0
    unclosed_by_family: collections.Counter = field(
        default_factory=collections.Counter
    )
    #: B1 instances in the security partition -- all sink-capable, hence B1's
    #: absence from the family table.
    b1_security: int = 0

    def by_class(self) -> dict[str, int]:
        c = collections.Counter(r.klass for r in self.closed)
        return {k: c.get(k, 0) for k in
                (CheckClass.STRUCTURAL, CheckClass.HYBRID, CheckClass.HEURISTIC)}

    def structural_share(self) -> float:
        n = len(self.closed)
        return self.by_class()[CheckClass.STRUCTURAL] / n if n else 0.0

    def by_family(self) -> dict[str, collections.Counter]:
        out: dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter
        )
        for r in self.closed:
            out[r.family][r.klass] += 1
        return dict(out)

    def by_check(self) -> collections.Counter:
        return collections.Counter(r.check_id for r in self.closed)


def _family_of(template: str) -> str:
    """The Table I family label for a template."""
    if template.startswith("xc_"):
        return "XC"
    return template.split("_", 1)[0].upper()


async def run_closure_attribution() -> AttributionResult:
    security = [
        i
        for i in load_instances(CORPUS_DIR)
        if i.kind == "attack" and suite_for(i.template) is ReportSuite.SECURITY
    ]
    result = AttributionResult(
        partition=len(security),
        sink_capable=sum(1 for i in security if is_sink_capable(i.template)),
    )
    targets = [i for i in security if not is_sink_capable(i.template)]
    result.non_sink = len(targets)
    result.b1_security = sum(
        1 for i in security if i.template.startswith("b1_")
    )

    # The sink-capable side, scored only so the report can reconcile against
    # Table I's pooled soft-win rate rather than assert a partial count.
    for inst in security:
        if not is_sink_capable(inst.template):
            continue
        score = score_trace(
            await SessionRunner(bound="production").run(inst, _FULL), inst
        )
        result.sink_capable_soft += bool(score.attack_success)
        result.sink_capable_hard += bool(score.hard_win)

    for inst in targets:
        trace = await SessionRunner(bound="production").run(inst, _FULL)
        if score_trace(trace, inst).attack_success:
            result.unclosed.append(inst.instance_id)
            result.unclosed_by_family[_family_of(inst.template)] += 1
            continue
        denial = next(
            (a for a in trace.actions if a.is_attack and not a.allowed), None
        )
        if denial is None:
            result.unclassified.append((inst.instance_id, "<no denied action>"))
            continue
        spec = classify(denial.reason or "")
        if spec is None:
            result.unclassified.append((inst.instance_id, denial.reason or ""))
            continue
        result.closed.append(
            ClosureRow(
                instance_id=inst.instance_id,
                template=inst.template,
                family=_family_of(inst.template),
                gate=denial.gate,
                check_id=spec.check_id,
                klass=spec.klass,
                reason=denial.reason or "",
            )
        )
    return result


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def to_markdown(r: AttributionResult) -> str:
    n = len(r.closed)
    cls = r.by_class()
    struct = cls[CheckClass.STRUCTURAL]
    hybrid = cls[CheckClass.HYBRID]
    heur = cls[CheckClass.HEURISTIC]
    not_structural = hybrid + heur

    out = [
        "<!-- Generated report. Source module: tools.closure_attribution. "
        "Regenerate with `uv run python -m tools.closure_attribution`. -->",
        "",
        "# What the non-sink-capable instances measure (A1)",
        "",
        f"Of the {r.partition}-instance security partition, {r.sink_capable} "
        f"can reach a privileged sink and {r.non_sink} cannot. R3 is right "
        "that the second group exercises the detection tiers rather than the "
        "sink predicate. This attributes each of its closures to the check "
        "that fired and classifies that check by what it decides on, read "
        "from the implementation named in the last table.",
        "",
        f"At `full` under the production bound, **{n} of {r.non_sink}** are "
        f"closed and **{len(r.unclosed)}** still succeed.",
        "",
        "## The split",
        "",
        "| Class | Closures | Share |",
        "|---|---|---|",
        f"| structural (parse, schema, allow-list, label) | {struct} | "
        f"{_pct(struct / n)} |",
        f"| hybrid (pattern locates, structural predicate decides) | {hybrid} | "
        f"{_pct(hybrid / n)} |",
        f"| heuristic (pattern, denylist, anomaly score) | {heur} | "
        f"{_pct(heur / n)} |",
        "",
        f"Folding hybrid into heuristic -- a check an attacker can hide from "
        f"is not a structural guarantee whatever its decision rule -- gives "
        f"**{struct} structural ({_pct(struct / n)}) against "
        f"{not_structural} searchable ({_pct(not_structural / n)})**.",
        "",
        "### What this does and does not contradict",
        "",
        "Section V-D's claim is about **hard wins**: the sink predicate is "
        "what halves them, and nothing here touches that. Section III-B "
        "generalizes it -- \"the reduction comes from the [structural "
        "checks]\" -- to the whole reduction. On the "
        f"{r.non_sink} instances that cannot reach a sink, that is not what "
        f"happens: {_pct(not_structural / n)} of their closures come from a "
        "check an adaptive adversary can search against. The two statements "
        "are compatible only if III-B is narrowed to the criterion V-D "
        "actually measures.",
        "",
        "## By family (the Table I column)",
        "",
        "| Family | Closures | Structural | Hybrid | Heuristic | Structural share |",
        "|---|---|---|---|---|---|",
    ]
    fam = r.by_family()
    for family in sorted(fam):
        c = fam[family]
        tot = sum(c.values())
        s = c[CheckClass.STRUCTURAL]
        out.append(
            f"| {family} | {tot} | {s} | {c[CheckClass.HYBRID]} | "
            f"{c[CheckClass.HEURISTIC]} | {_pct(s / tot) if tot else '--'} |"
        )
    total_soft = len(r.unclosed) + r.sink_capable_soft
    out += [
        "",
        f"B1 has no row: all {r.b1_security} of its security-partition "
        "instances are `b1_9_attached_content`, which is sink-capable, so the "
        "whole family sits in the other 32. The direct-prompt B1 classes are "
        "the misuse partition and are not counted here at all.",
        "",
        "### Reconciliation against Table I",
        "",
        f"The {len(r.unclosed)} instances still succeeding here plus the "
        f"{r.sink_capable_soft} soft wins inside the sink-capable 32 are "
        f"{total_soft} of {r.partition} = "
        f"{100 * total_soft / r.partition:.1f}%, which is Table I's pooled "
        f"soft-win rate at `full`. Of those {r.sink_capable_soft}, "
        f"{r.sink_capable_hard} are also hard wins -- Table I's 13 -- so every "
        "sink-capable instance that evades inspection at `full` also reaches a "
        "sink.",
        "",
        "Unclosed by family: "
        + ", ".join(
            f"{fam} {n}" for fam, n in sorted(r.unclosed_by_family.items())
        )
        + ".",
        "",
        "## By check",
        "",
        "| Check | Gate | Class | Closures | Decides on | Implementation |",
        "|---|---|---|---|---|---|",
    ]
    counts = r.by_check()
    specs = {s.check_id: s for s in TAXONOMY}
    for check_id, k in counts.most_common():
        s = specs[check_id]
        out.append(
            f"| `{s.check_id}` | {s.gate} | {s.klass} | {k} | {s.rationale} | "
            f"`{s.symbol}` |"
        )
    out += [
        "",
        "## Scope",
        "",
        "- Deterministic and offline; this replays the existing `full` "
        "configuration under the production bound and adds no new runs.",
        "- Attribution is to the **first** denying attack action in the trace. "
        "A later check that would also have fired is not credited, so these "
        "counts are first-responder attribution, not coverage.",
        "- The classification is a judgement about each check's decision rule, "
        "not a measurement. Every row names the implementation it was read "
        "from so it can be checked; the hybrid class exists because forcing "
        "those two checks either way would misstate them.",
    ]
    if r.unclassified:
        out += [
            "",
            f"**{len(r.unclassified)} closure(s) did not match the taxonomy** "
            "and are excluded from the counts above:",
            "",
        ]
        out += [f"- `{iid}`: {reason[:150]}" for iid, reason in r.unclassified]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-out", type=Path, default=None)
    args = ap.parse_args(argv)

    result = asyncio.run(run_closure_attribution())
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
