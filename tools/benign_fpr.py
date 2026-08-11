"""
Benign false-positive-rate measurement over the expanded control (W1.3 / W1.4).

Reviewer R1-M1: a 0% FPR on a 24-task control is weak evidence. This tool runs
the benign false-positive control (``benign_workload`` + the W1 expanded
``benign_diverse``) through the **deterministic fast-tier** ``full`` gate stack --
the reported operating point (the abstract's "deterministic tier … 0% FPR") --
and reports the benign FPR:

* **pooled** and **per gate** (G1/G2/G3/G5), each with an **exact-binomial
  (Clopper-Pearson) 95% CI** -- the interval the manuscript already quotes for
  the 0/24 cell, now over ~181 tasks so it actually tightens;
* **per category** (a: imperative prompts, b: instruction-quoting retrievals,
  c: dual-use-adjacent science, d: edge-of-policy HPC), from the
  ``benign_diverse`` variation-axis prefix;
* **per signature** (W1.4): for every false block, the gate + reason prefix that
  fired (e.g. ``G5 lifecycle-hook injection``), so a nonzero FPR is attributable
  to a named signature rather than an opaque rate.

The fast tier is deterministic and offline, so this run is fully reproducible.
The ``+both`` (Q-LLM slow-tier) FPR needs a served quarantine model; pass
``quarantine_agents`` (or run under the full-ablation harness with a key) to
measure it -- the fast-tier result is the reported operating point regardless.

    cd backend
    uv run python -m tools.benign_fpr
    uv run python -m tools.benign_fpr --report-out docs/palisade/benign_fpr.md
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from siege import SessionRunner, load_instances
from siege.ablation_matrix import (
    AUGMENTED_CONFIGS,
    CUMULATIVE_CONFIGS,
    DECOMPOSITION_CONFIGS,
    FAIL_CLOSED_CONFIG,
    LEAVE_ONE_OUT_CONFIGS,
    AblationConfig,
)
from siege.corpus_builder import CORPUS_DIR
from siege.report_suites import BENIGN_TEMPLATES
from palisade.paths import CORPUS_DIR

#: The augmented ``full +all`` config (all gates + Semgrep + Q-LLM slow tier).
_FULL_BOTH: AblationConfig = next(c for c in AUGMENTED_CONFIGS if c.name == "full +all")

#: High-stakes sink actions, for the ``--fail-closed`` coverage caveat. A
#: SIEGE scheduler submission is a ``slurm_script`` payload adjudicated by G5
#: rather than a ``submit_hpc_job`` tool call, so both spellings count.
_SINK_TOOLS = frozenset({"run_bash", "create_file", "submit_hpc_job", "cancel_hpc_job"})


def _controls_reaching_a_sink(instances_dir: Path | None = None) -> int:
    """How many benign controls reach a high-stakes sink at all.

    The denominator behind the ``--fail-closed`` caveat: a sink-side rule can
    only fire where the control exercises a sink, so this number bounds what
    the control set is able to measure about one.
    """
    import yaml

    base = instances_dir or (
        CORPUS_DIR
    )
    count = 0
    for family in ("benign_workload", "benign_diverse"):
        for path in sorted((base / family).glob("*.yaml")):
            spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            reaches = any(
                (payload.get("tool_name") in _SINK_TOOLS) or ("slurm_script" in payload)
                for session in spec.get("sessions", [])
                for turn in session.get("turns", [])
                for action in turn.get("actions", [])
                for payload in ((action.get("payload") or {}),)
            )
            count += int(reaches)
    return count

#: The reported operating point: the deterministic fast-tier ``full`` config
#: (all gates live + capability bound + trust; Semgrep/Q-LLM off when no
#: quarantine agents are wired).
_FAST_FULL: AblationConfig = CUMULATIVE_CONFIGS[-1]


# -----------------------------------------------------------------
# Exact-binomial (Clopper-Pearson) interval
# -----------------------------------------------------------------


def clopper_pearson(k: int, n: int, *, alpha: float = 0.05) -> tuple[float, float]:
    """Exact-binomial (Clopper-Pearson) ``(lo, hi)`` for ``k``/``n`` at
    ``1-alpha`` confidence. Uses the Beta-quantile form; returns ``(0.0, 1.0)``
    for ``n == 0``. This is the interval the manuscript quotes for the benign
    FPR (vs. the Wilson interval the per-cell ASR table uses)."""
    if n <= 0:
        return (0.0, 1.0)
    from scipy.stats import beta  # scipy is a backend dependency

    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


# -----------------------------------------------------------------
# Per-instance benign outcome
# -----------------------------------------------------------------


@dataclass(frozen=True)
class BenignOutcome:
    """One benign instance's fate under a config."""

    instance_id: str
    template: str
    category: str  # a / b / c / d for benign_diverse; "" otherwise
    gate: str | None  # the is_utility action's routed gate
    blocked: bool
    signature: str  # "" when allowed; else "G<n> <name>" (reason prefix)
    reason: str  # full reason string when blocked
    incident_level: int | None = None  # SEV of the block (1=high-stakes/sticky)


def _category_of(instance) -> str:
    """The benign_diverse category (a/b/c/d) from the variation axis, else ''."""
    axis = instance.variation_axis or ""
    if instance.template == "benign_diverse" and axis[:1] in ("a", "b", "c", "d") and axis[1:2] == "_":
        return axis[0]
    return ""


def _signature(reason: str) -> str:
    """Stable signature key: the gate + name before the first ':' in the reason
    (e.g. 'G5 lifecycle-hook injection'). Falls back to the whole reason."""
    head = reason.split(":", 1)[0].strip()
    return head or reason.strip()


def _utility_record(trace, gate_hint: str | None):
    """The record for the benign task's is_utility action (there is exactly one
    per benign instance in this control)."""
    utils = [a for a in trace.actions if a.is_utility]
    return utils[0] if utils else None


# -----------------------------------------------------------------
# Report
# -----------------------------------------------------------------


@dataclass
class FprReport:
    config_name: str
    outcomes: list[BenignOutcome] = field(default_factory=list)

    # ---- aggregate helpers ----
    def _rate(self, subset: list[BenignOutcome]) -> tuple[int, int, float, tuple[float, float]]:
        n = len(subset)
        k = sum(1 for o in subset if o.blocked)
        fpr = (k / n) if n else 0.0
        return k, n, fpr, clopper_pearson(k, n)

    def pooled(self):
        return self._rate(self.outcomes)

    def by_gate(self) -> dict[str, tuple[int, int, float, tuple[float, float]]]:
        groups: dict[str, list[BenignOutcome]] = collections.defaultdict(list)
        for o in self.outcomes:
            groups[o.gate or "—"].append(o)
        return {g: self._rate(v) for g, v in sorted(groups.items())}

    def by_template(self) -> dict[str, tuple[int, int, float, tuple[float, float]]]:
        groups: dict[str, list[BenignOutcome]] = collections.defaultdict(list)
        for o in self.outcomes:
            groups[o.template].append(o)
        return {t: self._rate(v) for t, v in sorted(groups.items())}

    def by_category(self) -> dict[str, tuple[int, int, float, tuple[float, float]]]:
        groups: dict[str, list[BenignOutcome]] = collections.defaultdict(list)
        for o in self.outcomes:
            if o.category:
                groups[o.category].append(o)
        return {c: self._rate(v) for c, v in sorted(groups.items())}

    def by_signature(self) -> dict[str, int]:
        sigs = collections.Counter(o.signature for o in self.outcomes if o.blocked)
        return dict(sorted(sigs.items(), key=lambda kv: (-kv[1], kv[0])))

    def blocked(self) -> list[BenignOutcome]:
        return [o for o in self.outcomes if o.blocked]

    def to_markdown(self) -> str:
        def pct(x: float) -> str:
            return f"{x * 100:.1f}%"

        def ci(lohi: tuple[float, float]) -> str:
            lo, hi = lohi
            return f"[{lo * 100:.1f}%, {hi * 100:.1f}%]"

        k, n, fpr, cint = self.pooled()
        lines = [
            "# Benign false-positive rate — expanded control (W1.3 / W1.4)",
            "",
            f"Config: **{self.config_name}** (deterministic fast tier; the reported "
            "operating point). Benign control: "
            f"{', '.join(sorted(BENIGN_TEMPLATES))}. FPR = fraction of benign tasks "
            "whose legitimate action was blocked. CIs are exact-binomial "
            "(Clopper-Pearson) 95%.",
            "",
            f"**Pooled benign FPR: {k}/{n} = {pct(fpr)}, 95% CI {ci(cint)}.**",
            "",
            "## Per gate",
            "",
            "| gate | false blocks | n | FPR | 95% CI |",
            "|---|---|---|---|---|",
        ]
        for g, (gk, gn, gf, gci) in self.by_gate().items():
            lines.append(f"| {g} | {gk} | {gn} | {pct(gf)} | {ci(gci)} |")
        lines += [
            "",
            "## Per benign suite",
            "",
            "| suite | false blocks | n | FPR | 95% CI |",
            "|---|---|---|---|---|",
        ]
        for t, (tk, tn, tf, tci) in self.by_template().items():
            lines.append(f"| `{t}` | {tk} | {tn} | {pct(tf)} | {ci(tci)} |")
        cats = self.by_category()
        if cats:
            _labels = {
                "a": "imperative-but-benign prompts",
                "b": "instruction-quoting retrievals",
                "c": "dual-use-adjacent science",
                "d": "edge-of-policy HPC jobs",
            }
            lines += [
                "",
                "## Per benign_diverse category",
                "",
                "| category | false blocks | n | FPR | 95% CI |",
                "|---|---|---|---|---|",
            ]
            for c, (ck, cn, cf, cci) in cats.items():
                lines.append(
                    f"| {c} — {_labels.get(c, c)} | {ck} | {cn} | {pct(cf)} | {ci(cci)} |"
                )
        sigs = self.by_signature()
        lines += [
            "",
            "## Per signature (every false block)",
            "",
        ]
        if not sigs:
            lines.append("_No false blocks — the fast tier passed the entire control._")
        else:
            lines += ["| signature | false blocks |", "|---|---|"]
            for s, c in sigs.items():
                lines.append(f"| `{s}` | {c} |")
            lines += [
                "",
                "### Blocked instances",
                "",
                "| instance | gate | signature | reason |",
                "|---|---|---|---|",
            ]
            for o in self.blocked():
                lines.append(
                    f"| `{o.instance_id}` | {o.gate} | `{o.signature}` | {o.reason} |"
                )
        lines.append("")
        return "\n".join(lines)


# -----------------------------------------------------------------
# Measurement
# -----------------------------------------------------------------


async def measure_benign_fpr(
    *,
    config: AblationConfig = _FAST_FULL,
    instances_dir: str | Path | None = None,
    quarantine_agents: dict[str, Any] | None = None,
    judge: Any | None = None,
) -> FprReport:
    """Run the benign control through ``config`` and collect per-instance
    outcomes. Fast tier is offline; pass ``quarantine_agents`` to also engage the
    slow tier (the ``+both`` posture), and ``judge`` to engage the slow-tier
    judge stage (only consulted when ``config.judge_active``). The judge's
    benign cost has to be measured on the payloads a gate actually hands it,
    which is not the same text a standalone detector baseline is scored on."""
    base = instances_dir if instances_dir is not None else CORPUS_DIR
    instances = [i for i in load_instances(base) if i.kind == "benign"]
    report = FprReport(config_name=config.name)
    for inst in instances:
        runner = SessionRunner(quarantine_agents=quarantine_agents, judge=judge)
        trace = await runner.run(inst, config)
        rec = _utility_record(trace, None)
        gate = rec.gate if rec is not None else None
        blocked = bool(rec is not None and not rec.allowed)
        reason = rec.reason if (rec is not None and blocked) else ""
        report.outcomes.append(
            BenignOutcome(
                instance_id=inst.instance_id,
                template=inst.template,
                category=_category_of(inst),
                gate=gate,
                blocked=blocked,
                signature=_signature(reason) if blocked else "",
                reason=reason,
                incident_level=(rec.incident_level if (rec is not None and blocked) else None),
            )
        )
    return report


async def per_config_fpr(
    configs: tuple[AblationConfig, ...],
    *,
    instances_dir: str | Path | None = None,
) -> list[tuple[str, int, int, float, tuple[float, float]]]:
    """Pooled fast-tier benign FPR over the control for each config in
    ``configs`` (offline; the deterministic per-config column of Table E3).
    Returns ``[(config_name, k, n, fpr, (lo, hi)), ...]``."""
    rows: list[tuple[str, int, int, float, tuple[float, float]]] = []
    for cfg in configs:
        report = await measure_benign_fpr(config=cfg, instances_dir=instances_dir)
        k, n, fpr, cint = report.pooled()
        rows.append((cfg.name, k, n, fpr, cint))
    return rows


def _print_per_config() -> None:
    """Emit the per-config deterministic benign FPR for E3 (cumulative +
    leave-one-out + decomposition; the augmented +Q-LLM/+both rows equal full at
    the fast tier, their slow-tier addition pending a served model)."""
    seen: set[str] = set()
    configs: list[AblationConfig] = []
    for cfg in (*CUMULATIVE_CONFIGS, *LEAVE_ONE_OUT_CONFIGS, *DECOMPOSITION_CONFIGS):
        if cfg.name not in seen:
            seen.add(cfg.name)
            configs.append(cfg)
    rows = asyncio.run(per_config_fpr(tuple(configs)))
    print(f"{'config':<34} {'k/n':>8} {'FPR':>7}  95% CI")
    for name, k, n, fpr, (lo, hi) in rows:
        print(
            f"{name:<34} {f'{k}/{n}':>8} {fpr * 100:>6.1f}%  "
            f"[{lo * 100:.1f}%, {hi * 100:.1f}%]"
        )


async def measure_plus_both_fpr(
    model: Any, *, instances_dir: str | Path | None = None
) -> tuple[FprReport, dict[str, int]]:
    """Measure the benign FPR at the augmented ``full +all`` posture (W1.3b):
    the deterministic fast tier plus the live Q-LLM slow tier and Semgrep. Builds
    the same per-gate caching Q-LLM agents ``full_ablation`` uses (so identical
    benign prompts are queried once), and returns ``(report, qllm_stats)``.
    Needs a served quarantine model reachable via ``model``."""
    from siege.full_ablation import _build_cached_agents

    stats = {"model_calls": 0, "cache_hits": 0, "errors": 0}
    agents = _build_cached_agents(model, stats)
    report = await measure_benign_fpr(
        config=_FULL_BOTH, instances_dir=instances_dir, quarantine_agents=agents
    )
    return report, stats


def _resolve_qllm_model(args: argparse.Namespace) -> tuple[Any, str]:
    """Resolve the Q-LLM model spec + a human label (mirrors full_ablation)."""
    from siege.smoke_static_qllm import DEFAULT_QLLM_MODEL

    if args.stub:
        from pydantic_ai.models.test import TestModel

        return TestModel(), "TestModel (offline stub)"
    model = args.model or DEFAULT_QLLM_MODEL
    label = str(model)
    if isinstance(model, str) and model.startswith("ollama:"):
        base = args.ollama_base_url or "http://localhost:11434/v1"
        os.environ["OLLAMA_BASE_URL"] = base
        label = f"{model} @ {base}"
    elif isinstance(model, str) and model.startswith("openai:"):
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url
        label = f"{model} @ {args.openai_base_url}"
    return model, label


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the benign false-positive rate over the expanded control at "
            "the deterministic fast-tier full posture, with exact-binomial CIs "
            "and a per-signature breakdown (W1.3 / W1.4)."
        )
    )
    parser.add_argument(
        "--report-out", default=None, metavar="PATH",
        help="Write the markdown report to PATH (also printed).",
    )
    parser.add_argument(
        "--diverse-only", action="store_true",
        help="Measure only benign_diverse (exclude the original benign_workload).",
    )
    parser.add_argument(
        "--per-config", action="store_true",
        help="Emit the per-config deterministic benign FPR (the Table E3 column).",
    )
    parser.add_argument(
        "--plus-both", action="store_true",
        help="Measure at the augmented full +all posture (live Q-LLM slow tier); "
             "the W1.3b run. Needs a served quarantine model.",
    )
    parser.add_argument(
        "--model", default=None,
        help="Q-LLM model spec for --plus-both (default: the deployed gpt-oss-120b). "
             "Use the UNFILTERED endpoint for the deployed-posture number.",
    )
    parser.add_argument(
        "--stub", action="store_true",
        help="With --plus-both: use an offline TestModel stub (wiring smoke test, "
             "not a real FPR).",
    )
    parser.add_argument(
        "--ollama-base-url", default=os.environ.get("OLLAMA_BASE_URL"),
        help="Base URL for an `ollama:` model (default: $OLLAMA_BASE_URL).",
    )
    parser.add_argument(
        "--openai-base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="Base URL for an `openai:` model (default: $OPENAI_BASE_URL).",
    )
    parser.add_argument(
        "--fail-closed",
        action="store_true",
        help=(
            "Measure the benign cost of the 'full +failclosed' posture (G2 "
            "denies an unlabeled argument at a high-stakes sink while "
            "untrusted content is live). Pairs with full_ablation "
            "--fail-closed. NOTE: this control set is sink-light and "
            "non-chaining -- read the caveat this run prints before quoting "
            "the number."
        ),
    )
    args = parser.parse_args(argv)

    if args.per_config:
        _print_per_config()
        return 0

    if args.plus_both:
        model, label = _resolve_qllm_model(args)
        print(f"[benign-fpr] +both posture | Q-LLM via {label}")
        report, stats = asyncio.run(measure_plus_both_fpr(model))
        print(
            f"[benign-fpr] Q-LLM: {stats['model_calls']} call(s), "
            f"{stats['cache_hits']} cache hit(s), {stats['errors']} error(s)"
        )
        if not args.stub and stats["model_calls"] == 0 and stats["errors"] > 0:
            from siege.smoke_static_qllm import _qllm_hint

            print(
                "[benign-fpr] FAIL: every Q-LLM call errored -- the model is "
                "unreachable (calls default-denied, so this is not a valid FPR)."
            )
            print(_qllm_hint())
            return 1
    elif args.fail_closed:
        report = asyncio.run(measure_benign_fpr(config=FAIL_CLOSED_CONFIG))
        n_sink = _controls_reaching_a_sink()
        print(
            "\n[benign-fpr] CAVEAT -- read before quoting this number.\n"
            f"  Only {n_sink} of {len(report.outcomes)} benign controls reach a "
            "high-stakes sink at all, and no control both ingests untrusted\n"
            "  content and reaches a sink in the same session. The "
            "fail-closed rule can only fire on a session that does both, so\n"
            "  this control set cannot price it: a low number here measures "
            "the absence of the workload, not the absence of cost. A\n"
            "  retrieve-then-write / retrieve-then-submit benign category is "
            "the missing control."
        )
    else:
        report = asyncio.run(measure_benign_fpr())
    if args.diverse_only:
        report.outcomes = [o for o in report.outcomes if o.template == "benign_diverse"]

    md = report.to_markdown()
    if args.report_out:
        # Create the parent rather than discarding a completed measurement on a
        # path typo -- the live +both / --fail-closed runs are not cheap.
        out = Path(args.report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"[benign-fpr] wrote report to {args.report_out}")
    print("\n" + md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
