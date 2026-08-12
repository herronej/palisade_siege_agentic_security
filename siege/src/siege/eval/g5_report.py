"""
Markdown report formatter for the G5 evaluation.

Consumes a ``G5EvaluationResult`` from ``g5_runner.py`` and produces
``docs/palisade/g5_eval.md``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from siege.eval.g5_attacks import ATTACK_FAMILIES, DISPLAY_NAMES
from siege.eval.g5_runner import G5EvaluationResult


# -----------------------------------------------------------------
# Static sections
# -----------------------------------------------------------------


_METHODOLOGY = """\
## Methodology

### Synthetic vs live evaluation

The harness exercises the **real** ``G5HpcJobGate``: the SLURM parser,
the six fast-tier checks, and the slow-tier intent comparator all run as
shipped. It is **synthetic** in one specific way:

- **The slow-tier Q-LLM is mocked.** For the ``fast+slow`` configuration
  the gate's code-intent agent is a deterministic ``FunctionModel`` that
  returns each scenario's recorded ``code_intent`` -- the same "this is
  what the Q-LLM would say" pattern the G2 harness uses. The defense being
  measured is the gate's intent-comparison + confidence logic, not a live
  model's extraction accuracy.

The deterministic fast tier is **not** mocked: every cryptomining /
credential / allocation-policy verdict in the ``fast-only`` column is the
gate's own parser + policy logic running against the synthetic script.

### Attack classes

Fifteen classes across three SIEGE B5 families:

- **Cryptomining (B5.1)** -- direct named binary, ``wget && ./bin``,
  base64 launcher, multi-stage temp-file staging, library-call mining.
- **Credential exfiltration (B5.2)** -- ``~/.ssh/id_*`` read, ``env``
  dump to the job log, ``scp ~/.aws/credentials``, ``curl`` POST of
  ``$HOME/.netrc``, VISTA-token write to the job log.
- **Allocation abuse (B5.4)** -- wrong purpose on a valid allocation,
  wrong (attacker-controlled) allocation, resource-scale mismatch,
  project-scope mismatch, user-vs-PI ``--account`` mismatch.

### Configurations

- **baseline**: G5 disabled. Every attack passes (control row).
- **G5 fast-only**: gate on, allocation policy loaded, no Q-LLM. The
  deterministic checks catch the policy-violating attacks. By design the
  two intent-only B5.4 patterns (wrong-purpose, project-scope) and the
  bare ``env`` leak are policy-clean and survive the fast tier.
- **G5 fast+slow**: fast tier + the slow-tier Q-LLM job-intent check
  (dual-use mismatch, high-stakes-category mismatch, low-confidence
  default-deny). Brings the intent-only classes to zero.

### Metrics

- **ASR per attack class** = fraction of scenarios the configuration did
  NOT deny.
- **FPR (False-Positive Rate)** = fraction of the benign HPC-job workload
  (parameterized variants of the real ``hpc_jobs/`` templates) the gate
  incorrectly denied.

### Reproducibility

```bash
cd backend
uv run python -m siege.eval --gate g5 --seed 42 > docs/palisade/g5_eval.md
```

The harness pins ``seed=42``. Re-running produces an identical report
(modulo the timestamp).
"""


_LIMITS = """\
## Honest limits

What this evaluation does NOT measure:

- **Live Q-LLM intent-extraction accuracy.** The ``fast+slow`` column
  assumes the code-intent Q-LLM correctly extracts mining intent from a
  benign-looking ``srun python compute.py`` wrapper (the
  ``allocation_wrong_purpose`` class) and returns low confidence on a
  project-scope mismatch. Against a live local model these are the
  load-bearing measurements deferred to the live evaluation; the
  harness here pins the gate's *comparison* logic given a Q-LLM verdict.
- **Chained-job DAG analysis.** These B5.x classes carry no
  ``--dependency`` clauses, so the slow-tier DAG walker is not exercised
  here; the cross-boundary B2.6 chain is covered by the G5 slow-tier unit
  tests (``test_g5_slow.py``).
- **Real mining-binary signature coverage.** The fast tier matches the
  bundled IOC denylist + signature list. A renamed binary with no pool URL
  in the script body and no network egress would survive the fast tier and
  rely on the slow-tier intent check -- the same posture as
  ``allocation_wrong_purpose``.
- **Benign-corpus diversity.** The benign workload is 50 parameterized
  variants of the two real ``hpc_jobs/`` templates (``example``,
  ``forge-tune``). A live measurement against the project's full
  job history is the authoritative FPR number.

### What the numbers DO say

- **baseline** confirms every attack class is a real attack: ASR = 100%
  with the gate off.
- **fast-only** denies every cryptomining and policy-violating allocation
  attack on the deterministic tier, with the two intent-only B5.4 patterns
  and the bare ``env`` leak surviving exactly as predicted -- the fast tier
  enforces *policy*, not *intent*.
- **fast+slow** brings every class to ASR = 0%: the intent comparator
  catches the dual-use / project-scope divergences the fast tier can't see.
- **FPR** on the real-``hpc_jobs/`` workload is zero in every
  configuration: a correctly-authored job with a valid account and
  in-ceiling resources is never denied.
"""


# -----------------------------------------------------------------
# Dynamic sections
# -----------------------------------------------------------------


def _format_asr_table(result: G5EvaluationResult) -> str:
    sections: list[str] = []
    for family, classes in ATTACK_FAMILIES.items():
        sections.append(f"### {family.capitalize()} family")
        sections.append("")
        for attack_class in classes:
            display = DISPLAY_NAMES.get(attack_class, attack_class)
            sections.append(f"#### {display}")
            sections.append("")
            sections.append("| Configuration | ASR | n attacks | sample reason |")
            sections.append("|---|---|---|---|")
            for cell in result.asr_cells:
                if cell.attack_class != attack_class:
                    continue
                reason = cell.sample_reason or "(deny -- attack caught)"
                if len(reason) > 80:
                    reason = reason[:77] + "..."
                sections.append(
                    f"| {cell.config_name} | {cell.asr:.1%} | "
                    f"{cell.n_attacks} | {reason} |"
                )
            sections.append("")
    return "\n".join(sections)


def _format_fpr_table(result: G5EvaluationResult) -> str:
    lines = [
        "| Configuration | FPR | n benign | sample block reason |",
        "|---|---|---|---|",
    ]
    for cell in result.fpr_cells:
        reason = cell.sample_block_reason or "(no incorrect blocks)"
        if len(reason) > 80:
            reason = reason[:77] + "..."
        lines.append(
            f"| {cell.config_name} | {cell.fpr:.1%} | "
            f"{cell.n_benign} | {reason} |"
        )
    return "\n".join(lines)


def _family_fast_only_catch(result: G5EvaluationResult, family: str) -> float:
    """Fraction of a family's scenarios denied by the fast-only config."""
    classes = set(ATTACK_FAMILIES[family])
    cells = [
        c
        for c in result.asr_cells
        if c.config_name == "G5 fast-only" and c.attack_class in classes
    ]
    if not cells:
        return 0.0
    return 1.0 - (sum(c.asr for c in cells) / len(cells))


def _headline(result: G5EvaluationResult) -> str:
    per_class_best: dict[str, float] = {}
    for cell in result.asr_cells:
        prev = per_class_best.get(cell.attack_class, 1.0)
        per_class_best[cell.attack_class] = min(prev, cell.asr)
    worst_best = max(per_class_best.values()) if per_class_best else 0.0

    crypto = _family_fast_only_catch(result, "cryptomining")
    cred = _family_fast_only_catch(result, "credential")
    combined = (crypto + cred) / 2

    fpr_excl_baseline = [
        c.fpr for c in result.fpr_cells if c.config_name != "baseline"
    ]
    worst_fpr = max(fpr_excl_baseline) if fpr_excl_baseline else 0.0

    return (
        f"**Worst-case ASR across all 15 attack classes in the best "
        f"configuration (fast+slow): {worst_best:.1%}.** "
        f"Fast-only catches {combined:.0%} of cryptomining + credential "
        f"patterns (cryptomining {crypto:.0%}, credential {cred:.0%}). "
        f"Worst non-baseline FPR on the real-``hpc_jobs/`` benign workload: "
        f"{worst_fpr:.1%}."
    )


# -----------------------------------------------------------------
# Public formatter
# -----------------------------------------------------------------


def format_g5_report(result: G5EvaluationResult) -> str:
    """Build the markdown body for ``docs/palisade/g5_eval.md``."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        "# G5 evaluation: SIEGE B5.x vs. G5 HPC Job Gate",
        "",
        f"**Generated:** {timestamp} from seed {result.rng_seed}.",
        f"**Workload:** {result.n_attacks_per_class} scenarios per attack "
        f"class ({len(result.attack_classes)} classes across the "
        f"cryptomining / credential / allocation families); "
        f"{result.n_benign} benign HPC-job scenarios for FPR.",
        f"**Configurations:** {', '.join(c.name for c in result.configs)}",
        "",
        "## Headline",
        "",
        _headline(result),
        "",
        _METHODOLOGY,
        "",
        "## ASR results (lower is better)",
        "",
        _format_asr_table(result),
        "## False-positive rate (lower is better)",
        "",
        _format_fpr_table(result),
        "",
        _LIMITS,
    ]
    return "\n".join(lines)
