"""
Static SIEGE smoketest with the Q-LLM slow tiers ENABLED.

The default ``run_siege_evaluation`` is fast-tier-only. This
smoketest wires the slow-tier Q-LLM agents into the authored-replay
(static) harness -- G1 intent-extraction, G2/G3 quarantine, G4/G5
code-intent -- and runs a subset through ``full PALISADE`` so the slow
tiers actually fire. It proves the static harness runs end-to-end with the
Q-LLMs on, and a call counter confirms the slow tier was genuinely engaged
(not silently skipped).

Run against the real Q-LLM. The default is the i2-core OpenAI-compatible
``gpt-oss-120b`` deployment -- an unfiltered endpoint that reads the
adversarial corpus -- so no model flags are needed:

    cd backend
    uv run python -m siege.smoke_static_qllm --max 40

Point it elsewhere with ``--model`` (plus ``--openai-base-url`` /
``--ollama-base-url`` for the endpoint), e.g. a local Ollama:

    uv run python -m siege.smoke_static_qllm \\
        --model ollama:llama3.2:3b --max 40

Offline (no served model -- ``TestModel`` synthesizes the structured
output, so the slow-tier code path runs anywhere):

    uv run python -m siege.smoke_static_qllm --stub

**Use ``--stub`` when no unfiltered Q-LLM is reachable** -- the recommended
smoke in a locked-down environment. A *local* model (Ollama / vLLM) may be
blocked, and a hosted, *safety-filtered* model (e.g. ``azure:gpt-5``)
400-rejects the adversarial corpus (``invalid_prompt``), so neither serves
as a Q-LLM there. The default i2-core ``gpt-oss-120b`` endpoint is unfiltered
and works when reachable; ``--stub`` still validates the harness wiring and
the slow-tier code path end-to-end when it isn't.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from siege.eval.siege_runner import run_siege_evaluation
from palisade.quarantine import (
    build_code_intent_extraction_agent,
    build_intent_extraction_agent,
    build_quarantine_agent,
)
from siege.ablation_matrix import AUGMENTED_CONFIGS, CUMULATIVE_CONFIGS

# The static smoke's default Q-LLM: the i2-core OpenAI-compatible
# ``gpt-oss-120b`` deployment. Unlike a hosted, safety-filtered model (which
# 400-rejects the adversarial corpus), this endpoint reads the attack content,
# so it serves as a real Q-LLM. Override with --model / --openai-base-url.
DEFAULT_QLLM_MODEL = "openai:gpt-oss-120b"
DEFAULT_OPENAI_BASE_URL = "https://api.i2-core.american-science-cloud.org"


class _CountingAgent:
    """Thin proxy that counts attempted and **successful** ``.run()`` calls.

    A call that raises (e.g. the Q-LLM is unreachable -> the run-helper
    catches it and default-denies) counts as an attempt but NOT a success.
    That lets the smoke distinguish "slow tier wired and firing" from
    "Q-LLM actually answered" -- without the split, an unreachable Q-LLM
    would still look like a pass. ``counter`` is ``[attempts, successes]``.
    The gate only ever calls ``agent.run(prompt)``; everything else
    delegates to the wrapped agent."""

    def __init__(self, inner: Any, counter: list[int]) -> None:
        self._inner = inner
        self._counter = counter

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        self._counter[0] += 1
        result = await self._inner.run(*args, **kwargs)
        self._counter[1] += 1  # only reached if the call did not raise
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def build_qllm_agents(model: Any, counter: list[int]) -> dict[str, Any]:
    """Build the per-gate slow-tier Q-LLM agents from one ``model`` spec.

    ``model`` may be a model spec string (``"ollama:llama3.2:3b"``) or a
    model object (``TestModel()`` for offline). Each agent is wrapped in a
    counter so the caller can confirm the slow tier engaged.
    """
    intent = build_intent_extraction_agent(model)  # G1
    quarantine = build_quarantine_agent(model)  # G2 + G3
    code_intent = build_code_intent_extraction_agent(model)  # G4 + G5
    return {
        "G1": _CountingAgent(intent, counter),
        "G2": _CountingAgent(quarantine, counter),
        "G3": _CountingAgent(quarantine, counter),
        "G4": _CountingAgent(code_intent, counter),
        "G5": _CountingAgent(code_intent, counter),
    }


def _render_progress(done: int, total: int, counter: list[int]) -> None:
    """In-place text progress bar: instances completed + live Q-LLM tally
    (``succeeded/attempted``). Carriage-return updates one line."""
    width = 28
    filled = int(width * done / total) if total else width
    bar = "#" * filled + "." * (width - filled)
    pct = (100 * done // total) if total else 100
    sys.stdout.write(
        f"\r[smoke] [{bar}] {done}/{total} ({pct:3d}%) | "
        f"Q-LLM {counter[1]}/{counter[0]} ok "
    )
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n")


async def run_smoke(
    model: Any,
    *,
    max_instances: int = 40,
    show_progress: bool = False,
    semgrep: bool = False,
) -> tuple[Any, int, int]:
    """Run the static harness over ``max_instances`` through ``full
    PALISADE`` with every slow tier wired. Returns
    ``(result, n_attempted, n_succeeded)`` -- the second/third count the
    slow-tier Q-LLM calls attempted vs. those that actually returned.
    ``show_progress`` renders an in-place progress bar (CLI use).

    ``semgrep=True`` swaps in the ``full +Semgrep`` augmented config so the
    Tier-1 Semgrep tier runs *together* with the Q-LLM slow tier (the
    combined-defense column). It needs the ``semgrep`` CLI on PATH
    (``uv sync --extra palisade-g4``); without it G4 degrades to Tier-0."""
    counter = [0, 0]
    agents = build_qllm_agents(model, counter)
    config = (
        next(c for c in AUGMENTED_CONFIGS if c.semgrep_active)
        if semgrep
        else CUMULATIVE_CONFIGS[-1]
    )
    progress_cb = None
    if show_progress:

        def progress_cb(done: int, total: int) -> None:
            _render_progress(done, total, counter)

    result = await run_siege_evaluation(
        mode="authored",
        quarantine_agents=agents,
        configs=(config,),
        max_instances=max_instances,
        progress_cb=progress_cb,
    )
    return result, counter[0], counter[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Static SIEGE smoketest with the Q-LLM slow tier on."
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Q-LLM model spec (default: {DEFAULT_QLLM_MODEL}).",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Offline: use a TestModel stub instead of a served Q-LLM.",
    )
    parser.add_argument(
        "--max", type=int, default=40, help="Instances to run (id-sorted subset)."
    )
    parser.add_argument(
        "--ollama-base-url",
        default=os.environ.get("OLLAMA_BASE_URL"),
        help=(
            "Base URL for an `ollama:` model (default: $OLLAMA_BASE_URL, else "
            "http://localhost:11434/v1). pydantic-ai's Ollama provider requires "
            "it."
        ),
    )
    parser.add_argument(
        "--openai-base-url",
        default=os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL,
        help=(
            "Base URL for an `openai:` model (default: $OPENAI_BASE_URL, else "
            f"{DEFAULT_OPENAI_BASE_URL}). pydantic-ai's OpenAI provider reads "
            "it from $OPENAI_BASE_URL."
        ),
    )
    parser.add_argument(
        "--semgrep",
        action="store_true",
        help=(
            "Also enable the Tier-1 Semgrep tier -- Q-LLM + Semgrep together "
            "(the combined-defense column). Needs the 'semgrep' CLI on PATH "
            "(uv sync --extra palisade-g4); without it G4 degrades to Tier-0."
        ),
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print the full per-suite ASR / UA / BU / hard-win report after the run.",
    )
    parser.add_argument(
        "--report-out",
        default=None,
        metavar="PATH",
        help="Write the full report (markdown) to PATH.",
    )
    args = parser.parse_args(argv)

    if args.stub:
        from pydantic_ai.models.test import TestModel

        model: Any = TestModel()
        label = "TestModel (offline stub)"
    else:
        model = args.model or DEFAULT_QLLM_MODEL
        label = str(model)
        # pydantic-ai's Ollama provider needs a base URL; set it from the flag
        # / env so an `ollama:` spec resolves without extra setup.
        if isinstance(model, str) and model.startswith("ollama:"):
            base_url = args.ollama_base_url or "http://localhost:11434/v1"
            os.environ["OLLAMA_BASE_URL"] = base_url
            label = f"{model} @ {base_url}"
        # pydantic-ai's OpenAI provider reads OPENAI_BASE_URL from the env;
        # point an `openai:` spec (the default gpt-oss-120b) at the i2-core
        # endpoint without relying on .env being loaded.
        elif isinstance(model, str) and model.startswith("openai:"):
            os.environ["OPENAI_BASE_URL"] = args.openai_base_url
            label = f"{model} @ {args.openai_base_url}"

    if args.semgrep and shutil.which("semgrep") is None:
        print(
            "[smoke] WARNING: --semgrep set but the 'semgrep' CLI is not on "
            "PATH; G4 will degrade to Tier-0 (install: uv sync --extra "
            "palisade-g4)."
        )
    tiers = "Q-LLM slow tier ON" + (" + Semgrep ON" if args.semgrep else "")
    print(
        f"[smoke] static SIEGE | full PALISADE | {tiers} "
        f"via {label} | max={args.max}"
    )
    try:
        result, attempted, succeeded = asyncio.run(
            run_smoke(
                model,
                max_instances=args.max,
                show_progress=True,
                semgrep=args.semgrep,
            )
        )
    except Exception as exc:  # noqa: BLE001 -- smoketest reports any failure
        print(f"[smoke] FAILED: {type(exc).__name__}: {exc}")
        print(_qllm_hint())
        return 1

    print(
        f"[smoke] instances={result.n_instances} "
        f"({result.n_attack} attack, {result.n_benign} benign) | "
        f"cells={len(result.cells)} | slow-tier Q-LLM calls: "
        f"{attempted} attempted, {succeeded} succeeded"
    )
    if attempted > 0 and succeeded == 0:
        # Wired and firing, but the model never answered -> unreachable.
        print(
            "[smoke] FAIL: the slow tier fired but EVERY Q-LLM call failed -- "
            "the model is unreachable (calls default-denied)."
        )
        print(_qllm_hint())
        return 1

    ok = succeeded > 0 and len(result.cells) > 0
    print(
        f"[smoke] {'PASS' if ok else 'FAIL'}: "
        f"Q-LLM slow tier {'answered' if succeeded > 0 else 'never answered'} "
        f"and {'cells populated' if result.cells else 'no cells produced'}."
    )

    # Full per-suite ASR/UA/BU/hard-win report (the smoke runs `full` only,
    # so it's a single-config snapshot with the slow tier on).
    if args.report or args.report_out:
        from siege.eval.siege_report import format_siege_report

        markdown = format_siege_report(result)
        if args.report_out:
            Path(args.report_out).write_text(markdown, encoding="utf-8")
            print(f"[smoke] wrote report to {args.report_out}")
        if args.report:
            print("\n" + markdown)

    return 0 if ok else 1


def _qllm_hint() -> str:
    return (
        "[smoke] hint: the Q-LLM must be reachable AND able to read adversarial "
        "content. Common failures:\n"
        "  - 400 'invalid_prompt' (content filter): a hosted, safety-filtered "
        "model (e.g. azure:gpt-5) refuses the attack corpus -- use a LOCAL / "
        "unfiltered Q-LLM.\n"
        "  - 400 'exceeds the available context size': raise the server context "
        "window, e.g. `llama-server -c 32768` (Qwen2.5-3B supports 32k).\n"
        "  - Connection error (ollama): nothing served at --ollama-base-url / "
        "$OLLAMA_BASE_URL (default http://localhost:11434/v1).\n"
        "  - Connection / auth error (openai): check --openai-base-url / "
        "$OPENAI_BASE_URL and that $OPENAI_API_KEY is set for the endpoint.\n"
        "  - Or run offline with --stub (no model needed)."
    )


if __name__ == "__main__":
    sys.exit(main())
