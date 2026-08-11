"""
PALISADE/PALISADE as a live AgentDojo defense -- paper table E1.

Wires the real gate stack into AgentDojo's pipeline as a two-tier defense and
runs the actual benchmark, so PALISADE sits next to CaMeL / Progent on the *same*
benchmark:

- **Inspection tier** -- `PalisadeDetector` is a `PromptInjectionDetector`
  inserted into the `ToolsExecutionLoop` exactly like AgentDojo's
  `transformers_pi_detector`; it screens each tool output through the real G3
  injection patterns and redacts a flagged one.
- **Structural tier** -- `PalisadeToolGate` gates the agent's *proposed tool
  calls* before execution: a privileged sink whose arguments carry a span lifted
  from an injection-flagged tool output is a `taint:T` value reaching a
  high-privilege sink (the §8.2 bound) and is aborted. This is the capability
  contribution a detection-only defense cannot make.

`agentdojo` is an **optional dependency** (`uv sync --extra palisade-agentdojo`);
this module imports without it (the pipeline/runner then raise a clear error) so
the rest of `tools/` is unaffected. The benchmark itself needs a live LLM
(`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`).

Run (from `backend/`, needs the extra + a key):

    OPENAI_API_KEY=... uv run --extra palisade-agentdojo python -m \\
        tools.agentdojo_e1 \\
        --model gpt-4o-2024-05-13 --suites banking --report-out ../docs/palisade/agentdojo_e1.md
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tools.palisade_screen import (
    PalisadeCallPolicy,
    PalisadeScreen,
    looks_like_injection,
    shares_span,
)
from palisade.paths import REPO_ROOT

try:  # optional dependency
    from agentdojo.agent_pipeline import (
        AbortAgentError,
        AgentPipeline,
        BasePipelineElement,
        PromptInjectionDetector,
        ToolsExecutionLoop,
    )
    from agentdojo.agent_pipeline.agent_pipeline import PipelineConfig, load_system_message
    from agentdojo.attacks import load_attack
    from agentdojo.benchmark import (
        benchmark_suite_with_injections,
        benchmark_suite_without_injections,
        get_suite,
    )
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.logging import OutputLogger
    from agentdojo.types import get_text_content_as_str

    _HAVE_AGENTDOJO = True
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only without the extra
    _HAVE_AGENTDOJO = False
    _IMPORT_ERROR = exc
    BasePipelineElement = object  # type: ignore  (so the class body parses)
    PromptInjectionDetector = object  # type: ignore

__all__ = [
    "AGENTDOJO_SUITES",
    "DEFAULT_BENCHMARK_VERSION",
    "is_high_stakes_tool",
    "E1Row",
    "E1Result",
    "CITED_BASELINES",
    "run_agentdojo_e1",
    "write_results_doc",
]

AGENTDOJO_SUITES: tuple[str, ...] = ("workspace", "banking", "travel", "slack")
DEFAULT_BENCHMARK_VERSION = "v1.2.1"

#: Verbs that mark a state-changing (privileged) tool -- the sinks that matter.
#: AgentDojo read tools (get_/read_/list_/search_/retrieve_) are benign.
_STATE_CHANGING = (
    "send", "transfer", "pay", "create", "delete", "update", "post", "write",
    "book", "share", "add", "remove", "schedule", "cancel", "move", "set",
    "reply", "forward", "invite", "withdraw", "deposit", "append", "edit",
    "reserve", "rent", "order", "purchase", "authorize", "grant", "publish",
    "upload", "download", "execute", "install", "submit", "wire", "charge",
    "refund", "assign", "enable", "disable", "revoke",
)
_READ_PREFIXES = ("get_", "read_", "list_", "search_", "retrieve_", "check_", "find_", "query_")


def is_high_stakes_tool(name: str) -> bool:
    """Heuristic: a state-changing sink (not an obvious read-only accessor).

    Deliberately conservative and name-based so it is deterministic and
    inspectable; a deployment can pass its own predicate to the tool gate.
    """
    n = (name or "").lower()
    if any(n.startswith(p) for p in _READ_PREFIXES):
        return False
    return any(verb in n for verb in _STATE_CHANGING)


def _arg_values(value):
    """Yield the leaf scalar values of a (possibly nested) tool-call args mapping."""
    if isinstance(value, Mapping):
        for v in value.values():
            yield from _arg_values(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _arg_values(v)
    else:
        yield value


# -----------------------------------------------------------------
# Endpoint compatibility: some OpenAI-compatible endpoints are backed by
# litellm -> AWS Bedrock, which rejects a tool_use `input` that is not a JSON
# object. A no-argument tool call (e.g. `get_balance()`) round-trips to
# `arguments="{}"` normally, but `null`/empty serializations slip through and
# Bedrock 400s on replay. Normalize outgoing request messages so every assistant
# tool-call carries a JSON-object `arguments`. Pure request-side fix-up; it never
# alters arguments that already carry data.
# -----------------------------------------------------------------


def _normalize_tool_args(args):
    """Return a Bedrock-safe JSON-object string for a tool call's `arguments`.

    Coerces missing / empty / ``null`` / non-object arguments to ``"{}"``, and
    drops empty-string keys -- the model sometimes emits ``{"": {}}`` for a
    no-argument tool call, which is a valid JSON object but which litellm->Bedrock
    rejects as an invalid tool-use ``input``. Real arguments pass through byte-for-
    byte.
    """
    if not (isinstance(args, str) and args.strip()):
        return "{}"
    try:
        obj = json.loads(args)
    except ValueError:
        return "{}"
    if not isinstance(obj, dict):
        return "{}"
    cleaned = {k: v for k, v in obj.items() if isinstance(k, str) and k.strip()}
    return args if cleaned == obj else json.dumps(cleaned)


#: Bedrock validates `toolUse.name` against this pattern and rejects the whole
#: request when any message in the history violates it. A planner that
#: hallucinates a namespaced call (`banking.get_balance`) therefore kills the
#: run several turns later, when the bad name is replayed as context.
_BEDROCK_TOOL_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")

_log = logging.getLogger(__name__)


def _normalize_tool_name(name):
    """Recover a usable tool name and coerce it into Bedrock's `[a-zA-Z0-9_-]+`.

    Two distinct cases, and conflating them corrupts the measurement:

    * A **harmony control marker** leaking into the name
      (``search_files<|channel|>commentary``). gpt-oss emits its channel
      delimiters inline and the OpenAI-compat shim folds them into the function
      name. The model did call ``search_files``, so the real name is the text
      before the first delimiter and recovering it lets the episode proceed.
      Mangling it instead would fail the call as tool-not-found and *understate
      utility* in every arm, which is a silent scoring error rather than a crash.
    * A genuinely **malformed or hallucinated** name, which cannot be recovered.
      Those are made Bedrock-safe so the request is accepted and the episode
      fails as tool-not-found, which is the correct outcome for a call that was
      never real, and far better than a 400 that aborts the whole run.

    Well-formed names pass through unchanged.
    """
    if not isinstance(name, str):
        return name
    # Harmony/ChatML control markers: the tool name is everything before the
    # first delimiter. Covers `<|channel|>`, `<|message|>`, `<|end|>`, ...
    head = re.split(r"<\|", name, maxsplit=1)[0].strip()
    if head and head != name:
        recovered = _BEDROCK_TOOL_NAME_RE.sub("_", head)
        if recovered == head:
            return recovered
    fixed = _BEDROCK_TOOL_NAME_RE.sub("_", name)
    return fixed or "invalid_tool_name"


def _sanitize_messages_for_bedrock(messages):
    """Normalize every assistant tool-call's `name` and `arguments` for Bedrock."""
    if not messages:
        return messages
    out = []
    for m in messages:
        tool_calls = m.get("tool_calls") if isinstance(m, Mapping) else None
        if not tool_calls:
            out.append(m)
            continue
        fixed = []
        for tc in tool_calls:
            fn = tc.get("function") if isinstance(tc, Mapping) else None
            if not isinstance(fn, Mapping):
                fixed.append(tc)
                continue
            new_args = _normalize_tool_args(fn.get("arguments"))
            new_name = _normalize_tool_name(fn.get("name"))
            if new_args == fn.get("arguments") and new_name == fn.get("name"):
                fixed.append(tc)
                continue
            if new_name != fn.get("name"):
                orig = fn.get("name")
                recovered = isinstance(orig, str) and orig.startswith(new_name)
                _log.warning(
                    "Tool name %r -> %r (%s).", orig, new_name,
                    "harmony marker stripped, call proceeds" if recovered
                    else "unrecoverable; episode will fail as tool-not-found",
                )
            fixed.append(
                {**tc, "function": {**fn, "name": new_name, "arguments": new_args}}
            )
        out.append({**m, "tool_calls": fixed})
    return out


#: Transient gateway failures (`503 Bedrock is unable to process your request`)
#: are common on a shared endpoint over a multi-hour benchmark. AgentDojo's own
#: tenacity wrapper gives up quickly and the exception then aborts the entire
#: run, so we retry more patiently here. Only 5xx and explicit
#: service-unavailable bodies are retried; a 4xx is a real error and propagates.
_TRANSIENT_ATTEMPTS = 6
_TRANSIENT_BACKOFF_S = 4.0


def _is_transient(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status >= 500:
        return True
    text = repr(exc).lower()
    return any(
        m in text
        for m in ("serviceunavailable", "unable to process", "throttl", "timeout")
    )


class _BedrockSafeCompletions:
    def __init__(self, inner) -> None:
        self._inner = inner

    def create(self, *args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _sanitize_messages_for_bedrock(kwargs["messages"])
        delay = _TRANSIENT_BACKOFF_S
        for attempt in range(1, _TRANSIENT_ATTEMPTS + 1):
            try:
                return self._inner.create(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 -- re-raised unless transient
                if attempt == _TRANSIENT_ATTEMPTS or not _is_transient(exc):
                    raise
                _log.warning(
                    "Transient endpoint failure (%s), attempt %d/%d; retrying in "
                    "%.0fs. A run that dies here is resumable from the log dir.",
                    type(exc).__name__, attempt, _TRANSIENT_ATTEMPTS, delay,
                )
                time.sleep(delay)
                delay *= 2

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _BedrockSafeChat:
    def __init__(self, inner) -> None:
        self.completions = _BedrockSafeCompletions(inner.completions)
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _BedrockSafeClient:
    """OpenAI-client proxy that normalizes tool-call args on `chat.completions.create`."""

    def __init__(self, inner) -> None:
        self.chat = _BedrockSafeChat(inner.chat)
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)


# -----------------------------------------------------------------
# The E1 table (agentdojo-free: pure data, so it formats/tests without the extra)
# -----------------------------------------------------------------


@dataclass(frozen=True)
class E1Row:
    """One row of the E1 headline comparison."""

    system: str
    benign_utility: float | None
    asr: float | None
    utility_under_attack: float | None
    benign_fpr: float | None
    cited: bool = False
    note: str = ""


#: Baseline rows we do NOT run here -- cited from their papers on AgentDojo.
#: Anchors are the values in the outline; leave a cell None and fill from the
#: paper rather than invent a number.
CITED_BASELINES: tuple[E1Row, ...] = (
    E1Row(
        system="Undefended [ref]",
        benign_utility=0.84, asr=None, utility_under_attack=None, benign_fpr=0.0,
        cited=True, note="CaMeL reports ~84% undefended AgentDojo utility [Debenedetti et al. 2025]",
    ),
    E1Row(
        system="CaMeL [Debenedetti et al. 2025]",
        benign_utility=0.77, asr=None, utility_under_attack=0.77, benign_fpr=None,
        cited=True, note="77% utility with provable security vs 84% undefended; fill ASR from the paper",
    ),
    E1Row(
        system="Progent [Shi et al. 2025]",
        benign_utility=None, asr=None, utility_under_attack=None, benign_fpr=None,
        cited=True, note="least-privilege DSL; ~6% of policy updates need approval; fill utility/ASR from the paper",
    ),
    E1Row(
        system="LlamaFirewall [Meta 2025]",
        benign_utility=None, asr=None, utility_under_attack=None, benign_fpr=None,
        cited=True, note="detection/guardrail baseline; fill from the paper",
    ),
)


@dataclass(frozen=True)
class E1Result:
    """The measured PALISADE/Undefended rows + the cited baselines."""

    measured: tuple[E1Row, ...]
    model_id: str
    suites: tuple[str, ...]
    attack: str
    benchmark_version: str
    n_benign_tasks: int
    n_injection_pairs: int
    cited: tuple[E1Row, ...] = CITED_BASELINES

    def _row(self, r: E1Row) -> str:
        def pct(v: float | None) -> str:
            return "--" if v is None else f"{v:.0%}"

        tag = " _(cited)_" if r.cited else ""
        return (
            f"| {r.system}{tag} | {pct(r.benign_utility)} | {pct(r.asr)} | "
            f"{pct(r.utility_under_attack)} | {pct(r.benign_fpr)} |"
        )

    def to_markdown(self) -> str:
        rows = [self._row(r) for r in self.measured] + [
            self._row(r) for r in self.cited
        ]
        notes = [f"- **{r.system}** -- {r.note}" for r in self.cited if r.note]
        return "\n".join(
            [
                "# AgentDojo headline comparison (E1)",
                "",
                f"PALISADE wired into AgentDojo as a live two-tier defense (G3 "
                f"injection screening on tool outputs + the capability tool-gate "
                f"on proposed calls), run on the real benchmark so it sits on the "
                f"**same** surface as the capability defenses it is compared to.",
                "",
                f"- **model:** `{self.model_id}`  ·  **suites:** "
                f"{', '.join(self.suites)}  ·  **attack:** `{self.attack}`  ·  "
                f"**benchmark:** {self.benchmark_version}",
                f"- **coverage:** {self.n_benign_tasks} benign user-task run(s), "
                f"{self.n_injection_pairs} (user-task × injection-task) attack pair(s)",
                "",
                "| System | Benign utility | Attack success (ASR) "
                "| Utility under attack | FPR on benign |",
                "|---|---|---|---|---|",
                *rows,
                "",
                "_ASR = fraction of (user-task × injection-task) pairs where the "
                "injection goal was executed (AgentDojo `security=True`). Benign "
                "utility is over the no-injection runs; FPR on benign = benign "
                "tasks the undefended agent completed that the defense broke. "
                "**PALISADE enters AgentDojo as a two-element proxy**, not the "
                "full six-gate stack: G3's injection patterns screening tool "
                "output, and the capability rule on proposed calls. AgentDojo "
                "has no retrieval-provenance, scheduler or egress boundary and "
                "exposes no value-level dataflow, so there is no G1/G5/G6, no "
                "capability registry and no contract library here, and Semgrep "
                "(a G4 tier over generated code) has nothing to analyse. These "
                "rows therefore measure a different configuration from the "
                "SIEGE `full` and `+all` columns and must not be read across._",
                "",
                "## Cited baselines (not re-run here)",
                "",
                "These rows are from the cited papers on AgentDojo, not measured "
                "in this run -- fill the `--` cells from the source before "
                "publishing:",
                "",
                *notes,
                "",
            ]
        )


def _require_agentdojo() -> None:
    if not _HAVE_AGENTDOJO:
        raise ImportError(
            "agentdojo is not installed. Install the optional extra:\n"
            "    uv sync --extra palisade-agentdojo\n"
            f"(original import error: {_IMPORT_ERROR!r})"
        )


def _register_custom_model(model_id: str, display_name: str = "AI assistant") -> None:
    """Make a custom (non-AgentDojo) model id resolvable by the attack machinery.

    The `important_instructions` attack addresses the model by name in the
    injection (`get_model_name_from_pipeline` over `MODEL_NAMES`) and raises on
    an unknown id. Register an OpenAI-compatible custom id (e.g.
    ``gpt-oss-120b``) as the neutral ``"AI assistant"`` -- the same display name
    AgentDojo already uses for the open Llama-3 model -- so the attack is built
    faithfully without mis-branding the served model as GPT-4. A model AgentDojo
    already knows is left untouched.
    """
    if not _HAVE_AGENTDOJO:
        return
    from agentdojo import models as _adm

    try:
        _adm.ModelsEnum(model_id)
    except ValueError:
        _adm.MODEL_NAMES.setdefault(model_id, display_name)


# -----------------------------------------------------------------
# The two PALISADE pipeline elements (defined only when agentdojo is present)
# -----------------------------------------------------------------

if _HAVE_AGENTDOJO:

    class PalisadeDetector(PromptInjectionDetector):
        """PALISADE's inspection tier as an AgentDojo `PromptInjectionDetector`.

        Screens each tool output through the real G3 injection patterns; a flag
        redacts the tool output (or aborts, if `raise_on_injection`) exactly like
        AgentDojo's built-in `transformers_pi_detector`.
        """

        def __init__(
            self,
            screen: PalisadeScreen | None = None,
            *,
            mode: str = "message",
            raise_on_injection: bool = False,
        ) -> None:
            super().__init__(mode=mode, raise_on_injection=raise_on_injection)
            self._screen = screen or PalisadeScreen()

        def detect(self, tool_output: str) -> tuple[bool, float]:
            v = self._screen.screen(tool_output)
            return v.flagged, v.confidence

    class PalisadeToolGate(BasePipelineElement):
        """PALISADE's structural tier: deny a privileged sink under an injected instruction.

        Placed *before* the `ToolsExecutor` in the loop, so it sees each round's
        proposed calls before they run. A tool return is tainted (taint:T) when it
        *structurally reads as an injected instruction* (`looks_like_injection` ---
        untrusted content addressing the agent and issuing a directive), not merely
        because it is a tool return: benign tool DATA carries no directive, so it is
        not tainted and legitimate tool-derived flows are not over-blocked. Once an
        injected instruction is present in the untrusted context, a privileged
        (state-changing) sink is denied --- the action may be injection-driven,
        whether or not its arguments overlap the injection text (so a parameterless
        `delete_file`/DoS is caught too). This is a proxy for the real system's
        quarantine of untrusted instructions plus its capability bound; AgentDojo
        exposes no value-level dataflow, so the airtight taint of the core model
        (Section~sec:principle, the SIEGE results) is approximated here.
        """

        def __init__(
            self,
            policy: PalisadeCallPolicy | None = None,
            screen: PalisadeScreen | None = None,
        ) -> None:
            super().__init__()
            self._screen = screen or PalisadeScreen()
            self._policy = policy or PalisadeCallPolicy(is_high_stakes_tool)

        def _injection_texts(self, messages: Sequence[Mapping]) -> list[str]:
            """Tool returns that structurally read as injected instructions (taint:T)."""
            out: list[str] = []
            for m in messages:
                if m.get("role") != "tool":
                    continue
                try:
                    text = get_text_content_as_str(m.get("content") or [])
                except Exception:
                    text = str(m.get("content") or "")
                # Via the screen, not `looks_like_injection` directly: the screen
                # applies the same structural signature and then escalates a pass
                # to the quarantined model when one is wired, so the tool gate
                # sees the slow tier too. Fast-only behaviour is unchanged.
                if text and self._screen.injection_flag(text):
                    out.append(text)
            return out

        def query(self, query, runtime, env=None, messages=(), extra_args=None):
            messages = list(messages)
            extra_args = dict(extra_args or {})
            if not messages or messages[-1].get("role") != "assistant":
                return query, runtime, env, messages, extra_args
            tool_calls = messages[-1].get("tool_calls") or []
            if not tool_calls:
                return query, runtime, env, messages, extra_args
            injections = self._injection_texts(messages)
            if not injections:  # no untrusted directive in context -> nothing to gate
                return query, runtime, env, messages, extra_args
            for call in tool_calls:
                name = getattr(call, "function", None) or ""
                if not is_high_stakes_tool(name):
                    continue
                args = getattr(call, "args", None) or {}
                serialized = " ".join(str(v) for v in _arg_values(args))
                span = next(
                    (s for s in (shares_span(serialized, inj) for inj in injections) if s),
                    None,
                )
                detail = (
                    f"argument span {span!r} lifted from an injected instruction"
                    if span is not None
                    else "an injected instruction is present in the untrusted tool context"
                )
                raise AbortAgentError(
                    f"PALISADE blocked tool call: taint:T value reaching privileged "
                    f"sink `{name}` ({detail})",
                    messages,
                    env,
                )
            return query, runtime, env, messages, extra_args

    __all__ += ["PalisadeDetector", "PalisadeToolGate", "build_pipelines"]

    def _resolve_llm(model_id: str):
        """Resolve a model id to an AgentDojo LLM element.

        A model in AgentDojo's ``ModelsEnum`` is passed through as a string so
        ``from_config`` builds it. Any other id -- e.g. an open-weights model
        served at an OpenAI-compatible endpoint (``gpt-oss-120b`` on the science
        cloud) -- is built here as an ``OpenAILLM`` against the env-configured
        client (``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``), bypassing
        ``ModelsEnum``, which would ``ValueError`` on an unlisted id.
        """
        import openai
        from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
        from agentdojo.models import ModelsEnum

        try:
            ModelsEnum(model_id)
            return model_id
        except ValueError:
            llm = OpenAILLM(_BedrockSafeClient(openai.OpenAI()), model_id)
            llm.name = model_id
            return llm

    def build_pipelines(
        model_id: str, *, include_tool_filter: bool = False,
        screen: PalisadeScreen | None = None,
    ) -> dict[str, "AgentPipeline"]:
        """Build the `undefended` and `palisade` pipelines sharing components.

        The undefended pipeline is AgentDojo's default (`from_config`, no
        defense); PALISADE reuses its system message / init-query / llm /
        tools-executor and inserts the two PALISADE elements into the tools loop.

        With ``include_tool_filter`` the dict also carries ``tool_filter`` --
        AgentDojo's own runnable **least-privilege** baseline (the ``tool_filter``
        defense: an LLM prunes the tool set to those the task needs before each
        round). This is the runnable IFC-family comparator R1-M5 asks for --
        least-privilege at tool granularity, the Progent design point -- so the E1
        head-to-head runs PALISADE beside it on the *same* base model. AgentDojo
        only supports the tool filter for OpenAI(-compatible) models with a name,
        which the science-cloud endpoint is.
        """
        config = PipelineConfig(
            llm=_resolve_llm(model_id),
            model_id=model_id,
            defense=None,
            system_message_name=None,
            system_message=load_system_message(None),
        )
        base = AgentPipeline.from_config(config)
        system_message, init_query, llm, tools_loop = base.elements
        tools_executor = tools_loop.elements[0]

        screen = screen or PalisadeScreen()
        palisade_loop = ToolsExecutionLoop(
            [
                PalisadeToolGate(screen=screen),  # gate proposed calls first
                tools_executor,                     # then execute the survivors
                PalisadeDetector(screen=screen),  # then screen the tool outputs
                llm,
            ]
        )
        palisade = AgentPipeline([system_message, init_query, llm, palisade_loop])
        # The pipeline name is the *cache key* AgentDojo logs and resumes
        # under, so it must encode the tier. Without the suffix a fast-only run
        # started after a slow-tier run silently replays the slow run's cached
        # task results and reports them under the fast label, which is a
        # cross-configuration contamination that looks like a real measurement.
        suffix = ""
        if screen is not None and getattr(screen, "slow_enabled", False):
            suffix = "-slow" if getattr(screen, "_judge", None) is not None else "-qllm"
        palisade.name = f"{base.name}-palisade{suffix}"
        pipelines: dict[str, "AgentPipeline"] = {"undefended": base, "palisade": palisade}
        if include_tool_filter:
            tf_config = PipelineConfig(
                llm=_resolve_llm(model_id),
                model_id=model_id,
                defense="tool_filter",
                system_message_name=None,
                system_message=load_system_message(None),
            )
            pipelines["tool_filter"] = AgentPipeline.from_config(tf_config)
        return pipelines


def _summarize(
    system: str,
    benign_util: dict,
    atk_security: dict,
    atk_util: dict,
    *,
    undefended_benign: dict | None,
) -> E1Row:
    """Roll per-task AgentDojo results into one E1 row."""
    def mean(d: dict) -> float | None:
        vals = list(d.values())
        return (sum(1 for v in vals if v) / len(vals)) if vals else None

    fpr: float | None = None
    if undefended_benign is not None:
        # False block: a benign task the undefended agent completed but this
        # defense broke. Denominator = tasks undefended actually completed.
        base_ok = [k for k, v in undefended_benign.items() if v]
        if base_ok:
            broke = sum(1 for k in base_ok if not benign_util.get(k, False))
            fpr = broke / len(base_ok)
    return E1Row(
        system=system,
        benign_utility=mean(benign_util),
        asr=mean(atk_security),  # security=True => injection executed => attack succeeded
        utility_under_attack=mean(atk_util),
        benign_fpr=(0.0 if undefended_benign is None else fpr),
    )


#: Display label per pipeline key (the tool_filter row is the least-privilege baseline).
_PIPELINE_LABEL = {
    "undefended": "Undefended (measured)",
    # Overwritten per run by `_palisade_label`: a fast-only run and a slow-tier
    # run must not emit the same row label, or two artifacts are
    # indistinguishable after the fact.
    "palisade": "PALISADE proxy (deterministic tier)",
    "tool_filter": "Tool-filter (least-privilege baseline)",
}


def _palisade_label(screen: "PalisadeScreen | None") -> str:
    """The row label naming the configuration actually run.

    Deployed PALISADE's slow tier is the quarantined model *and* the judge, so a
    run missing either is named for what it had. The alternative, one static
    "full" label, made a fast-only artifact and a slow-tier artifact
    indistinguishable.
    """
    if screen is None or not getattr(screen, "slow_enabled", False):
        return "PALISADE proxy (deterministic tier)"
    has_q = getattr(screen, "_agent", None) is not None
    has_j = getattr(screen, "_judge", None) is not None
    if has_q and has_j:
        return "PALISADE proxy (+ Q-LLM + judge)"
    return f"PALISADE proxy (+ {'Q-LLM only' if has_q else 'judge only'})"


def run_agentdojo_e1(
    model_id: str,
    suites: Sequence[str] = ("banking",),
    *,
    attack: str = "important_instructions",
    benchmark_version: str = DEFAULT_BENCHMARK_VERSION,
    user_tasks: Sequence[str] | None = None,
    injection_tasks: Sequence[str] | None = None,
    logdir: str | Path | None = None,
    include_tool_filter: bool = False,
    screen: "PalisadeScreen | None" = None,
) -> E1Result:
    """Run the AgentDojo benchmark for `undefended` and `palisade` -> the E1 rows.

    Needs a live LLM (`model_id` resolves an OpenAI/Anthropic/... backend, so set
    the matching API key). With ``include_tool_filter`` it also runs AgentDojo's
    runnable least-privilege ``tool_filter`` baseline on the same base model (the
    R1-M5 runnable-IFC-baseline slot). Returns measured rows + the cited baselines.
    """
    _require_agentdojo()
    _register_custom_model(model_id)
    # A *stable* default logdir, keyed by model. AgentDojo skips any task it
    # already has a log for when ``force_rerun=False``, so a stable directory
    # makes a crashed multi-hour run resumable: re-invoking replays the cached
    # tasks in seconds and continues from the failure. The previous default
    # minted a fresh temp directory per invocation, which silently discarded
    # every completed task when a run died partway through. Pass ``--logdir``
    # to override, or delete the directory to force a clean run.
    if logdir is not None:
        logdir = Path(logdir)
    else:
        safe_model = _BEDROCK_TOOL_NAME_RE.sub("_", model_id)
        logdir = Path(tempfile.gettempdir()) / f"agentdojo_e1_{safe_model}"
        logdir.mkdir(parents=True, exist_ok=True)
        print(f"[agentdojo-e1] resumable log dir: {logdir}")
    pipelines = build_pipelines(
        model_id, include_tool_filter=include_tool_filter, screen=screen
    )
    _PIPELINE_LABEL["palisade"] = _palisade_label(screen)

    per_defense_benign: dict[str, dict] = {}
    per_defense_sec: dict[str, dict] = {}
    per_defense_util: dict[str, dict] = {}
    n_benign = n_pairs = 0
    # AgentDojo's benchmark helpers build a `TraceLogger` whose directory comes
    # from the active logging context; with no context `Logger.get()` returns a
    # bare `NullLogger` that never set `.logdir` (it is set only in `__enter__`).
    # Run inside an `OutputLogger` so the trace logger resolves; traces land
    # under `logdir` (a fresh temp dir by default, so runs are clean).
    with OutputLogger(str(logdir), live=None):
        for dname, pipe in pipelines.items():
            benign: dict = {}
            sec: dict = {}
            util: dict = {}
            for sname in suites:
                suite = get_suite(benchmark_version, sname)
                attack_obj = load_attack(attack, suite, pipe)
                clean = benchmark_suite_without_injections(
                    pipe, suite, logdir=logdir, force_rerun=False,
                    user_tasks=user_tasks, benchmark_version=benchmark_version,
                )
                atk = benchmark_suite_with_injections(
                    pipe, suite, attack_obj, logdir=logdir, force_rerun=False,
                    user_tasks=user_tasks, injection_tasks=injection_tasks,
                    verbose=False, benchmark_version=benchmark_version,
                )
                benign.update({(sname, *k) if isinstance(k, tuple) else (sname, k): v
                               for k, v in clean["utility_results"].items()})
                sec.update({(sname, *k): v for k, v in atk["security_results"].items()})
                util.update({(sname, *k): v for k, v in atk["utility_results"].items()})
            per_defense_benign[dname] = benign
            per_defense_sec[dname] = sec
            per_defense_util[dname] = util
            n_benign = len(benign)
            n_pairs = len(sec)

    undefended_benign = per_defense_benign.get("undefended")
    measured = tuple(
        _summarize(
            _PIPELINE_LABEL.get(d, d),
            per_defense_benign[d], per_defense_sec[d], per_defense_util[d],
            undefended_benign=(None if d == "undefended" else undefended_benign),
        )
        for d in pipelines
    )
    return E1Result(
        measured=measured,
        model_id=model_id,
        suites=tuple(suites),
        attack=attack,
        benchmark_version=benchmark_version,
        n_benign_tasks=n_benign,
        n_injection_pairs=n_pairs,
    )


def write_results_doc(result: E1Result, path: str | None = None) -> str:
    if path is None:
        root = REPO_ROOT
        path = str(root / "docs" / "palisade" / "agentdojo_e1.md")
    Path(path).write_text(result.to_markdown(), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI (live)
    parser = argparse.ArgumentParser(
        description="AgentDojo E1: PALISADE vs undefended on the real benchmark."
    )
    parser.add_argument("--model", default="gpt-4o-2024-05-13", help="LLM id (needs the matching API key).")
    parser.add_argument("--suites", nargs="+", default=["banking"], choices=AGENTDOJO_SUITES)
    parser.add_argument("--attack", default="important_instructions")
    parser.add_argument("--benchmark-version", default=DEFAULT_BENCHMARK_VERSION)
    parser.add_argument("--user-tasks", nargs="*", default=None, help="Subset of user-task ids (cost guard).")
    parser.add_argument(
        "--injection-tasks", nargs="*", default=None, metavar="ID",
        help="Subset of injection-task ids. The E1 table (tab:e1) uses `injection_task_1` "
             "-- the one injection task common to all four suites -- for a single injection "
             "per user task (97 pairs). Omit to run the full user x injection cross-product "
             "(all injection tasks; ~10x more attack pairs).",
    )
    parser.add_argument("--logdir", default=None)
    parser.add_argument(
        "--tool-filter-baseline", action="store_true",
        help="Also run AgentDojo's runnable least-privilege tool_filter baseline "
             "on the same base model (the R1-M5 runnable-IFC comparator; OpenAI models).",
    )
    parser.add_argument("--report-out", default=None, metavar="PATH")
    parser.add_argument(
        "--slow", action="store_true",
        help="Escalate every fast-tier allow to the deployed quarantined model, "
             "the same rule the sidecar uses (a fast-tier deny short-circuits). "
             "Without this the run exercises the deterministic tier only.",
    )
    parser.add_argument(
        "--slow-model", default="openai:gpt-oss-120b",
        help="Model spec for the quarantined slow tier (default: openai:gpt-oss-120b).",
    )
    parser.add_argument(
        "--judge-model", default="gpt-oss-120b",
        help="Model id for the slow tier's judge stage (default: gpt-oss-120b).",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Ablate the judge from --slow, leaving the quarantined model alone. "
             "The judge is part of the deployed slow tier, so this is an ablation.",
    )
    args = parser.parse_args(argv)
    _require_agentdojo()
    screen = None
    if args.slow:
        from palisade.quarantine import build_quarantine_agent

        agent = build_quarantine_agent(args.slow_model)
        agent._max_result_retries = 3
        judge = None
        if not args.no_judge:
            from siege.redteam.baselines.detectors import LlmJudgeDetector

            judge = LlmJudgeDetector(model=args.judge_model)
            if not judge.available():
                print(
                    "[agentdojo-e1] FAILED: the slow-tier judge needs "
                    "OPENAI_BASE_URL and OPENAI_API_KEY. Pass --no-judge to "
                    "measure the quarantined model alone."
                )
                return 1
        screen = PalisadeScreen(quarantine_agent=agent, judge=judge)
        print(
            f"[agentdojo-e1] slow tier: Q-LLM {args.slow_model}, judge "
            + (args.judge_model if judge is not None else "ABLATED")
            + "; every fast-tier allow escalates. Expect a materially longer run."
        )
    result = run_agentdojo_e1(
        args.model, args.suites, attack=args.attack,
        benchmark_version=args.benchmark_version, user_tasks=args.user_tasks,
        injection_tasks=args.injection_tasks,
        logdir=args.logdir, include_tool_filter=args.tool_filter_baseline,
        screen=screen,
    )
    print(f"wrote {write_results_doc(result, args.report_out)}")
    print(result.to_markdown())
    if screen is not None:
        st = screen.stats
        errors = st.get("errors", 0)
        print(
            f"\n[agentdojo-e1] tier health: {st['fast_flags']} fast flag(s), "
            f"{st['slow_flags']} slow flag(s) on {st['escalations']} escalation(s); "
            f"{st['model_calls']} model call(s), {st['cache_hits']} cache hit(s), "
            f"{st.get('salvaged', 0)} salvaged, {errors} error(s)."
            + (
                "  UNUSABLE: a failed escalation is scored as *not* flagged here, "
                "so a non-zero error count understates the slow tier."
                if errors else "  clean."
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
