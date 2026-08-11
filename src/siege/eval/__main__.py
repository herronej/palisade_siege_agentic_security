"""
CLI entry point for the PALISADE evaluation harnesses.

"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from siege.eval.corpus import generate_corpus
from siege.eval.g1_report import format_g1_report
from siege.eval.g1_runner import run_g1_evaluation
from siege.eval.g2_report import format_g2_report
from siege.eval.g2_runner import run_g2_evaluation
from siege.eval.g4_report import format_g4_report
from siege.eval.g4_runner import run_g4_evaluation
from siege.eval.g5_report import format_g5_report
from siege.eval.g5_runner import run_g5_evaluation
from siege.eval.report import format_report
from siege.eval.runner import run_evaluation
from siege.eval.siege_report import format_siege_report
from siege.eval.siege_runner import run_siege_evaluation


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="siege.eval",
        description=(
            "Run a PALISADE synthetic evaluation and write the "
            "markdown report to stdout."
        ),
    )
    p.add_argument(
        "--gate",
        choices=("g1", "g2", "g3", "g4", "g5", "siege"),
        default="g3",
        help=(
            "Which gate's evaluation to run. g3 (default) measures "
            "the G3 PoisonedRAG/AgentPoison/MemoryGraft "
            "defenses; g2 measures the G2 AgentDojo + "
            "SIEGE A3 defenses; g1 measures the G1 "
            "jailbreak-corpus defenses; g4 measures the G4 "
            "malicious-code-corpus defenses; g5 measures the "
            "G5 SIEGE B5.x HPC-job defenses."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for scenario/corpus generation (default: 42)",
    )
    # G3-specific knobs (no-ops when --gate != g3).
    p.add_argument(
        "--n-chunks",
        type=int,
        default=100,
        help="[g3 only] Synthetic corpus size (default: 100)",
    )
    p.add_argument(
        "--n-queries",
        type=int,
        default=20,
        help="[g3 only] Benign query workload size (default: 20)",
    )
    # G1/G2/G4-shared knobs.
    p.add_argument(
        "--n-attacks-per-class",
        type=int,
        default=20,
        help="[g1/g2/g4] Scenarios per attack class (default: 20)",
    )
    p.add_argument(
        "--n-benign",
        type=int,
        default=None,
        help=(
            "[g1/g2/g4/g5] Benign workload size. Defaults: g1=200 (matches "
            "AC), g2=20, g4=50, g5=50."
        ),
    )
    # SIEGE-specific knobs (no-ops when --gate != siege).
    p.add_argument(
        "--instances-dir",
        type=str,
        default=None,
        help=(
            "[siege] Directory of instance YAML/JSON files. "
            "Defaults to the bundled fixtures."
        ),
    )
    p.add_argument(
        "--mode",
        choices=("authored", "live"),
        default="authored",
        help=(
            "[siege] 'authored' (default, offline replay) or 'live' "
            "(model-in-the-loop via a real ProjectAgent; requires a configured "
            "model + MCP servers)."
        ),
    )
    p.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="[siege] Cost guard: run only the first N instances.",
    )
    p.add_argument(
        "--qllm",
        action="store_true",
        help=(
            "[siege] Wire the Q-LLM slow tier (G1 intent, G2/G3 "
            "quarantine, G4/G5 code-intent) into the ablation, so every live "
            "gate runs its slow tier. Default model is the unfiltered "
            "gpt-oss-120b endpoint -- set OPENAI_API_KEY (and optionally "
            "OPENAI_BASE_URL). Omit for the fast-tier-only ablation."
        ),
    )
    p.add_argument(
        "--qllm-model",
        default=None,
        help=(
            "[siege] Q-LLM model spec used with --qllm "
            "(default: openai:gpt-oss-120b; e.g. ollama:llama3.2:3b)."
        ),
    )
    p.add_argument(
        "--qllm-stub",
        action="store_true",
        help=(
            "[siege] With --qllm, use an offline TestModel stub instead "
            "of a served model -- exercises the full slow-tier path with no "
            "endpoint (wiring check)."
        ),
    )
    p.add_argument(
        "--configs",
        choices=("cumulative", "all"),
        default="cumulative",
        help=(
            "[siege] 'cumulative' (default): the 6 Tier-0 / fast-tier "
            "B4-first configs. 'all': also runs the 3 augmented configs "
            "(full +Semgrep, full +Q-LLM, full +both) so the report shows the "
            "optional-tier ceilings. The +Semgrep columns need the `semgrep` "
            "CLI on PATH; the +Q-LLM columns need --qllm to be meaningful."
        ),
    )
    return p.parse_args(argv)


def _run_g3(args: argparse.Namespace) -> str:
    corpus = generate_corpus(
        seed=args.seed,
        n_chunks=args.n_chunks,
        n_queries=args.n_queries,
    )
    result = run_evaluation(corpus, rng_seed=args.seed)
    return format_report(result)


def _run_g2(args: argparse.Namespace) -> str:
    """G2 evaluation is async (the gate's check methods are
    coroutines)."""
    n_benign = args.n_benign if args.n_benign is not None else 20
    result = asyncio.run(
        run_g2_evaluation(
            rng_seed=args.seed,
            n_attacks_per_class=args.n_attacks_per_class,
            n_benign=n_benign,
        )
    )
    return format_g2_report(result)


def _run_g1(args: argparse.Namespace) -> str:
    """G1 evaluation. Default benign workload is 200 to match the AC."""
    n_benign = args.n_benign if args.n_benign is not None else 200
    result = asyncio.run(
        run_g1_evaluation(
            rng_seed=args.seed,
            n_attacks_per_class=args.n_attacks_per_class,
            n_benign=n_benign,
        )
    )
    return format_g1_report(result)


def _run_g4(args: argparse.Namespace) -> str:
    """G4 evaluation. Default benign workload is 50."""
    n_benign = args.n_benign if args.n_benign is not None else 50
    result = asyncio.run(
        run_g4_evaluation(
            rng_seed=args.seed,
            n_attacks_per_class=args.n_attacks_per_class,
            n_benign=n_benign,
        )
    )
    return format_g4_report(result)


def _run_g5(args: argparse.Namespace) -> str:
    """G5 evaluation. Defaults: 15 attacks/class, 50 benign scripts."""
    n_benign = args.n_benign if args.n_benign is not None else 50
    n_attacks = (
        args.n_attacks_per_class
        if args.n_attacks_per_class != 20
        else 15
    )
    result = asyncio.run(
        run_g5_evaluation(
            rng_seed=args.seed,
            n_attacks_per_class=n_attacks,
            n_benign=n_benign,
        )
    )
    return format_g5_report(result)


def _resolve_qllm_model(args: argparse.Namespace):
    """Resolve --qllm-model into a model spec (or an offline TestModel),
    wiring the OpenAI/Ollama base URL the same way the static smoke does so the
    default gpt-oss-120b endpoint resolves with only OPENAI_API_KEY set."""
    if args.qllm_stub:
        from pydantic_ai.models.test import TestModel

        return TestModel()
    from siege.smoke_static_qllm import (
        DEFAULT_OPENAI_BASE_URL,
        DEFAULT_QLLM_MODEL,
    )

    model = args.qllm_model or DEFAULT_QLLM_MODEL
    if model.startswith("openai:"):
        os.environ.setdefault("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    elif model.startswith("ollama:"):
        os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    return model


def _make_progress_cb():
    """Build a ``progress_cb(done, total)`` that renders a progress bar to
    stderr — so it never mixes into the report on stdout. Uses tqdm when
    available; otherwise a terse carriage-return counter on a TTY (and
    silence when redirected, to keep logs clean). Lazily initialized:
    ``total`` is only known once the first tick arrives.
    """
    state: dict = {"init": False, "bar": None, "tqdm": False}

    def cb(done: int, total: int) -> None:
        if not state["init"]:
            state["init"] = True
            try:
                from tqdm import tqdm

                # disable=None -> tqdm auto-disables when stderr isn't a TTY.
                state["bar"] = tqdm(
                    total=total, desc="siege", unit="cell", disable=None
                )
                state["tqdm"] = True
            except Exception:  # noqa: BLE001 — progress is best-effort
                state["tqdm"] = False
        if state["tqdm"]:
            bar = state["bar"]
            bar.update(done - bar.n)
            if done >= total:
                bar.close()
        elif sys.stderr.isatty():
            end = "\n" if done >= total else "\r"
            print(
                f"siege: {done}/{total} cells",
                end=end, file=sys.stderr, flush=True,
            )

    return cb


def _run_siege(args: argparse.Namespace) -> str:
    """SIEGE ablation (async -- gate checks are coroutines)."""
    agent_driver = None
    if args.mode == "live":
        agent_driver = _build_live_agent_driver()

    quarantine_agents = None
    counter = [0, 0]  # [attempted, succeeded] slow-tier Q-LLM calls
    if args.qllm:
        # Reuse the static smoke's gate->agent mapping (single source of truth)
        # plus its call counter, so the report can confirm the slow tier fired.
        from siege.smoke_static_qllm import build_qllm_agents

        quarantine_agents = build_qllm_agents(_resolve_qllm_model(args), counter)
        print(
            "siege: --qllm runs the Q-LLM slow tier with live LLM "
            "calls across the full ablation (configs x instances) — this is "
            "a long run, not a hang. The bar below shows pace + ETA; use "
            "--max-instances N and/or --qllm-stub for a quick pass.",
            file=sys.stderr,
        )

    from siege.ablation_matrix import ALL_CONFIGS, CUMULATIVE_CONFIGS

    configs = ALL_CONFIGS if args.configs == "all" else CUMULATIVE_CONFIGS
    result = asyncio.run(
        run_siege_evaluation(
            instances_dir=args.instances_dir,
            seed=args.seed,
            mode=args.mode,
            configs=configs,
            agent_driver=agent_driver,
            quarantine_agents=quarantine_agents,
            max_instances=args.max_instances,
            progress_cb=_make_progress_cb(),
        )
    )
    report = format_siege_report(result)
    if args.qllm:
        from siege.smoke_static_qllm import DEFAULT_QLLM_MODEL

        label = (
            "TestModel (stub)"
            if args.qllm_stub
            else (args.qllm_model or DEFAULT_QLLM_MODEL)
        )
        report += (
            f"\n\n_Q-LLM slow tier on via {label}: "
            f"{counter[1]}/{counter[0]} calls succeeded._"
        )
        if counter[0] and not counter[1]:
            report += (
                "\n\n> **Warning:** every Q-LLM call failed (model unreachable "
                "or no `OPENAI_API_KEY`) and default-denied, so these ASR "
                "numbers are artificially low. Set the key, or use `--qllm-stub`."
            )
    return report


def _build_live_agent_driver():
    """Build the live ProjectAgent driver, or fail with a clear message.

    Live mode (WI17) needs a configured model + MCP servers; in CI / offline
    the tested path is the Python API with a mocked AgentDriver.
    """
    raise SystemExit(
        "siege --mode live requires a configured ProjectAgent "
        "(model + MCP servers). Drive LiveSessionRunner via the Python API "
        "with a ProjectAgentDriver (see siege/README.md "
        "'Running with models in the loop')."
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runners = {
        "g1": _run_g1,
        "g2": _run_g2,
        "g3": _run_g3,
        "g4": _run_g4,
        "g5": _run_g5,
        "siege": _run_siege,
    }
    report = runners[args.gate](args)
    sys.stdout.write(report)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
