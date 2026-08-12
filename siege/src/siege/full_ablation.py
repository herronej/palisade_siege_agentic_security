"""
Full SIEGE ablation in one command.

Runs the 6-config B4-first cumulative gate sweep (``baseline`` -> ``full``,
deterministic / offline) AND the three augmented optional-tier columns --
``full +semgrep``, ``full +slow``, ``full +all`` -- then merges them into a
single per-(boundary, template) report with all nine columns.

**The Q-LLM is held fixed across the two Q-LLM columns.** ``+Q-LLM`` and
``+both`` share ONE set of *caching* Q-LLM agents: the model is queried once
per distinct prompt and the same decision is replayed for the second column.
So the ``+both`` - ``+Q-LLM`` delta is a clean measurement of what Semgrep adds
*on top of the Q-LLM*, not Semgrep + a fresh (re-sampled) model pass -- the
confound the live ``smoke_static_qllm`` has.

Determinism: the gate columns (``baseline``..``full``, ``+Semgrep``) are fully
deterministic. The Q-LLM columns ride one live model pass (held fixed across
the two via the cache). ``--stub`` makes the whole run deterministic.

    cd backend
    # real gpt-oss-120b (needs OPENAI_API_KEY) + Semgrep (needs the CLI):
    uv run --extra palisade-g4 python -m \\
        siege.full_ablation \\
        --max 216 --report-out full_ablation.md

    # offline, fully deterministic (stub Q-LLM; Semgrep still real if installed):
    uv run --extra palisade-g4 python -m \\
        siege.full_ablation --stub --max 216
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from siege.eval.siege_report import format_siege_report
from siege.eval.siege_runner import (
    SIEGEResult,
    run_siege_evaluation,
)
from palisade.quarantine import (
    build_code_intent_extraction_agent,
    build_intent_extraction_agent,
    build_quarantine_agent,
)
from siege.ablation_matrix import (
    AUGMENTED_CONFIGS,
    CUMULATIVE_CONFIGS,
    FAIL_CLOSED_CONFIG,
    JUDGE_ABLATED_CONFIG,
    AblationConfig,
)
from siege.scorer import aggregate_cells
from siege.smoke_static_qllm import (
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_QLLM_MODEL,
    _qllm_hint,
)


class _CachingAgent:
    """Memoizes ``.run(prompt)`` by the prompt string.

    Repeated identical Q-LLM queries (the same authored action evaluated under
    two configs) return the SAME decision without re-calling the model -- which
    is what holds the Q-LLM fixed across the ``+Q-LLM`` and ``+both`` columns so
    their delta is purely Semgrep. ``stats`` is a shared dict with
    ``model_calls`` / ``cache_hits`` / ``errors``. A call that raises is not
    cached; the session runner's fail-closed handler default-denies."""

    def __init__(self, inner: Any, stats: dict[str, int]) -> None:
        self._inner = inner
        self._stats = stats
        self._cache: dict[str, Any] = {}

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        key = str(args[0]) if args else ""
        if key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[key]
        try:
            result = await self._inner.run(*args, **kwargs)
        except Exception as exc:
            self._stats["errors"] += 1
            import logging

            logging.getLogger(__name__).warning(
                "Q-LLM structured-output failed (%s); "
                "session runner will default-deny",
                type(exc).__name__,
            )
            raise
        self._stats["model_calls"] += 1
        self._cache[key] = result
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _build_cached_agents(model: Any, stats: dict[str, int]) -> dict[str, Any]:
    """Per-gate caching Q-LLM agents from one ``model`` spec (mirrors the
    smoke's ``build_qllm_agents`` but memoizing, with retries=3 for
    structured-output resilience)."""
    intent = build_intent_extraction_agent(model)  # G1
    quarantine = build_quarantine_agent(model)  # G2 + G3
    code_intent = build_code_intent_extraction_agent(model)  # G4 + G5
    for agent in (intent, quarantine, code_intent):
        agent._max_result_retries = 3
    return {
        "G1": _CachingAgent(intent, stats),
        "G2": _CachingAgent(quarantine, stats),
        "G3": _CachingAgent(quarantine, stats),
        "G4": _CachingAgent(code_intent, stats),
        "G5": _CachingAgent(code_intent, stats),
    }


def _augmented(name: str) -> AblationConfig:
    return next(c for c in AUGMENTED_CONFIGS if c.name == name)


async def run_full_ablation(
    model: Any,
    *,
    max_instances: int | None = None,
    qllm_samples: int = 1,
    bound: str = "declarative",
    slow_tier_judge: Any | None = None,
    judge_ablation: bool = False,
    fail_closed: bool = False,
) -> tuple[SIEGEResult, dict[str, int]]:
    """Run the full ablation and return ``(merged_result, qllm_stats)``.

    Four grouped evaluations -- cumulative (offline), ``+Semgrep`` (offline),
    ``+Q-LLM`` (live), ``+both`` -- merged into one ``SIEGEResult``
    whose ``configs`` run ``baseline`` -> ``full`` -> ``+Semgrep`` ->
    ``+Q-LLM`` -> ``+both``.

    ``slow_tier_judge`` is the slow tier's judge stage, so it runs in the
    ``+Q-LLM`` and ``+both`` columns alongside the quarantined model rather than
    as a separate configuration; pass None to ablate it from the whole sweep.
    ``judge_ablation`` additionally appends a ``full +all -judge`` column whose
    delta against ``+both`` isolates the judge's marginal contribution over the
    rest of the slow tier.

    ``fail_closed`` appends a ``full +failclosed`` column: ``full`` with the G2
    sink guard denying an argument that resolves to no capability label in a
    session holding live untrusted content. It is deterministic and offline, so
    it costs one extra sweep and no model calls. Its delta against ``full`` is
    the hard-win closure; report it only alongside the paired benign-control
    delta from ``tools.benign_fpr``, which is what it costs.

    ``qllm_samples`` (k) runs the two Q-LLM columns over k **independent** live
    draws and pools them: each sample builds a fresh caching-agent set (so the
    model is re-sampled), while within a sample ``+Q-LLM`` and ``+both`` share
    that sample's cache (so their delta stays a clean Semgrep measurement). The
    pooled cells' n -- and the report's Wilson CI -- then reflect
    ``k × instances``, smoothing the single-live-pass variance that makes an
    optional-tier cell wobble above ``full``. ``k=1`` is the original behavior.
    """
    full_sem = _augmented("full +semgrep")
    full_qllm = _augmented("full +slow")
    full_both = _augmented("full +all")

    print("[full-ablation] (1/4) cumulative gate sweep (baseline -> full), offline ...")
    cum = await run_siege_evaluation(
        configs=CUMULATIVE_CONFIGS, max_instances=max_instances, bound=bound
    )
    print("[full-ablation] (2/4) full +semgrep (Q-LLM off), offline ...")
    sem = await run_siege_evaluation(
        configs=(full_sem,), max_instances=max_instances, bound=bound
    )

    fc_cells: tuple = ()
    fc_scores: tuple = ()
    fc_configs: tuple = ()
    if fail_closed:
        print("[full-ablation] (2b) full +failclosed (unlabeled-sink guard), offline ...")
        fc = await run_siege_evaluation(
            configs=(FAIL_CLOSED_CONFIG,), max_instances=max_instances, bound=bound
        )
        fc_cells, fc_scores, fc_configs = fc.cells, fc.scores, (FAIL_CLOSED_CONFIG,)

    stats = {"model_calls": 0, "cache_hits": 0, "errors": 0, "salvaged": 0}
    k = max(1, int(qllm_samples))
    qllm_scores: list = []
    both_scores: list = []
    judge_scores: list = []
    for s in range(k):
        # Fresh caching agents per sample -> an independent live draw. Within
        # the sample, +both replays the sample's cache (held-fixed delta).
        agents = _build_cached_agents(model, stats)
        print(
            f"[full-ablation] (3/4) full +slow sample {s + 1}/{k} "
            f"(live Q-LLM) ..."
        )
        q = await run_siege_evaluation(
            configs=(full_qllm,),
            quarantine_agents=agents,
            slow_tier_judge=slow_tier_judge,
            max_instances=max_instances,
            bound=bound,
        )
        print(
            f"[full-ablation] (4/4) full +all sample {s + 1}/{k} "
            f"(Q-LLM replayed) ..."
        )
        b = await run_siege_evaluation(
            configs=(full_both,),
            quarantine_agents=agents,
            slow_tier_judge=slow_tier_judge,
            max_instances=max_instances,
            bound=bound,
        )
        qllm_scores.extend(q.scores)
        both_scores.extend(b.scores)
        if judge_ablation:
            print(
                f"[full-ablation] (5/5) full +all -judge sample {s + 1}/{k} "
                f"(Q-LLM replayed, judge ablated) ..."
            )
            j = await run_siege_evaluation(
                configs=(JUDGE_ABLATED_CONFIG,),
                quarantine_agents=agents,
                slow_tier_judge=slow_tier_judge,
                max_instances=max_instances,
                bound=bound,
            )
            judge_scores.extend(j.scores)
    print(
        f"[full-ablation]       Q-LLM: {k} sample(s), {stats['model_calls']} "
        f"model call(s), {stats['cache_hits']} cache hit(s), "
        f"{stats['errors']} error(s)"
    )

    # Pool the k samples per Q-LLM column -> the multi-sample mean ASR per cell,
    # with n (and the Wilson CI) reflecting k × instances.
    qllm_cells = tuple(aggregate_cells(qllm_scores))
    both_cells = tuple(aggregate_cells(both_scores))
    judge_cells = tuple(aggregate_cells(judge_scores)) if judge_scores else ()
    extra_configs = (JUDGE_ABLATED_CONFIG,) if judge_scores else ()

    merged = SIEGEResult(
        configs=cum.configs + (full_sem,) + fc_configs + (full_qllm, full_both)
        + extra_configs,
        cells=cum.cells + sem.cells + fc_cells + qllm_cells + both_cells + judge_cells,
        scores=cum.scores
        + sem.scores
        + fc_scores
        + tuple(qllm_scores)
        + tuple(both_scores)
        + tuple(judge_scores),
        n_instances=cum.n_instances,
        n_attack=cum.n_attack,
        n_benign=cum.n_benign,
        seed=cum.seed,
        metric_definitions=cum.metric_definitions,
        contract_coverage=cum.contract_coverage,
        n_claims=cum.n_claims,
        n_claims_covered=cum.n_claims_covered,
        n_claims_violated=cum.n_claims_violated,
        science_contract=cum.science_contract,
        qllm_samples=k,
    )
    return merged, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Full SIEGE ablation: the 6-config cumulative gate sweep "
            "plus the +Semgrep / +Q-LLM / +both augmented columns (Q-LLM held "
            "fixed across the two Q-LLM columns)."
        )
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Q-LLM model spec (default: {DEFAULT_QLLM_MODEL}).",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Offline: use a TestModel stub for the Q-LLM (fully deterministic).",
    )
    parser.add_argument(
        "--max", type=int, default=None, help="Instances to run (id-sorted subset)."
    )
    parser.add_argument(
        "--qllm-samples",
        type=int,
        default=1,
        metavar="K",
        help=(
            "Independent live Q-LLM draws to pool for the +Q-LLM / +both "
            "columns (default: 1). K>1 averages out the single-live-pass "
            "variance; the pooled cells' n and CI reflect K×instances."
        ),
    )
    parser.add_argument(
        "--ollama-base-url",
        default=os.environ.get("OLLAMA_BASE_URL"),
        help="Base URL for an `ollama:` model (default: $OLLAMA_BASE_URL).",
    )
    parser.add_argument(
        "--openai-base-url",
        default=os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL,
        help=(
            "Base URL for an `openai:` model (default: $OPENAI_BASE_URL, else "
            f"{DEFAULT_OPENAI_BASE_URL})."
        ),
    )
    parser.add_argument(
        "--bound",
        choices=("declarative", "production"),
        default="declarative",
        help=(
            "Which label the capability bound reads. 'declarative' (default) "
            "trusts each corpus instance's authored capability.taint field -- "
            "the oracle label L*, which the manuscript reports only as the "
            "recorded-label row of the adaptive table. The published ablation "
            "(the cumulative ladder, both figures, and the hard-win counts "
            "13/12/4) is scored under 'production' -- pass it explicitly to "
            "reproduce those. 'production' reconstructs taint the way the "
            "deployed runtime does, by content-matching sink arguments against "
            "registered untrusted values, so a value transformed past the "
            "content guard arrives tag-dropped as it would in deployment. The "
            "gap between the two is the label-propagation residual."
        ),
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help=(
            "Ablate the slow-tier judge from the whole sweep. The judge is a "
            "stage of the slow tier, so by default it runs in the +Q-LLM and "
            "+both columns alongside the quarantined model."
        ),
    )
    parser.add_argument(
        "--fail-closed",
        action="store_true",
        help=(
            "Append a 'full +failclosed' column: full with the G2 sink guard "
            "denying an argument that resolves to no capability label while "
            "untrusted content is live in the session. Deterministic and "
            "offline. Pair its delta with tools.benign_fpr --fail-closed; the "
            "closure alone is half the result."
        ),
    )
    parser.add_argument(
        "--judge-ablation",
        action="store_true",
        help=(
            "Append a tenth 'full +all -judge' column whose delta against "
            "'full +all' isolates the judge's marginal contribution over the "
            "rest of the slow tier."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-oss-120b",
        help="Model id for --judge (default: gpt-oss-120b).",
    )
    parser.add_argument(
        "--report", action="store_true", help="Print the full report to stdout."
    )
    parser.add_argument(
        "--report-out",
        default=None,
        metavar="PATH",
        help="Write the full report (markdown) to PATH.",
    )
    args = parser.parse_args(argv)

    # Resolve the Q-LLM model (mirrors smoke_static_qllm.main).
    if args.stub:
        from pydantic_ai.models.test import TestModel

        model: Any = TestModel()
        label = "TestModel (offline stub)"
    else:
        model = args.model or DEFAULT_QLLM_MODEL
        label = str(model)
        if isinstance(model, str) and model.startswith("ollama:"):
            base_url = args.ollama_base_url or "http://localhost:11434/v1"
            os.environ["OLLAMA_BASE_URL"] = base_url
            label = f"{model} @ {base_url}"
        elif isinstance(model, str) and model.startswith("openai:"):
            os.environ["OPENAI_BASE_URL"] = args.openai_base_url
            label = f"{model} @ {args.openai_base_url}"

    slow_tier_judge = None
    if not args.no_judge:
        if args.stub:
            from tools.judge_slow_tier import _StubJudge

            slow_tier_judge = _StubJudge()
            print("[full-ablation] slow-tier judge: offline keyword stub (plumbing only)")
        else:
            from siege.redteam.baselines.detectors import LlmJudgeDetector

            slow_tier_judge = LlmJudgeDetector(model=args.judge_model)
            if not slow_tier_judge.available():
                print(
                    "[full-ablation] FAILED: the slow-tier judge needs "
                    "OPENAI_BASE_URL and OPENAI_API_KEY (read from the repo .env "
                    "via the backend config). Pass --no-judge to ablate it."
                )
                return 1
            print(f"[full-ablation] slow-tier judge: {args.judge_model}")

    if shutil.which("semgrep") is None:
        print(
            "[full-ablation] WARNING: the 'semgrep' CLI is not on PATH; the "
            "+Semgrep and +both columns will degrade to Tier-0 (install: uv "
            "sync --extra palisade-g4)."
        )
    print(
        f"[full-ablation] {9 + args.judge_ablation + args.fail_closed} configs "
        f"(baseline..full + Semgrep/Q-LLM/both"
        f"{'/failclosed' if args.fail_closed else ''}"
        f"{'/-judge' if args.judge_ablation else ''}) | "
        f"judge {'OFF' if args.no_judge else 'in slow tier'} | "
        f"Q-LLM via {label} | {args.qllm_samples} Q-LLM sample(s) | "
        f"max={args.max}"
    )

    try:
        result, stats = asyncio.run(
            run_full_ablation(
                model,
                max_instances=args.max,
                qllm_samples=args.qllm_samples,
                bound=args.bound,
                slow_tier_judge=slow_tier_judge,
                judge_ablation=args.judge_ablation,
                fail_closed=args.fail_closed,
            )
        )
    except Exception as exc:  # noqa: BLE001 -- report any failure
        print(f"[full-ablation] FAILED: {type(exc).__name__}: {exc}")
        print(_qllm_hint())
        return 1

    if not args.stub and stats["model_calls"] == 0 and stats["errors"] > 0:
        print(
            "[full-ablation] FAIL: every Q-LLM call errored -- the model is "
            "unreachable (calls default-denied)."
        )
        print(_qllm_hint())
        return 1

    # The judge's call-error rate is not optional bookkeeping: a detector whose
    # calls fail returns "not flagged" for each one, so a dead endpoint produces
    # a judge column identical to +both and reads as "the judge added nothing".
    # Surface it, and refuse to present the column as a measurement above the
    # same 2% threshold tools.judge_slow_tier enforces.
    if slow_tier_judge is not None:
        err = float(getattr(slow_tier_judge, "error_rate", 0.0) or 0.0)
        calls = int(getattr(slow_tier_judge, "_calls", 0) or 0)
        print(f"[full-ablation]       judge: {calls} call(s), {err:.1%} errored")
        if calls == 0:
            print(
                "[full-ablation] FAIL: the judge column ran but made no calls -- "
                "it is a copy of 'full +all', not a measurement."
            )
            return 1
        if err > 0.02:
            print(
                f"[full-ablation] FAIL: the judge errored on {err:.1%} of calls "
                "(>2%). Failed calls score as 'not flagged', so the judge column "
                "understates a healthy judge. Re-run against a healthy endpoint."
            )
            return 1

    markdown = format_siege_report(result)
    if args.report_out:
        Path(args.report_out).write_text(markdown, encoding="utf-8")
        print(f"[full-ablation] wrote report to {args.report_out}")
    if args.report or not args.report_out:
        print("\n" + markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
