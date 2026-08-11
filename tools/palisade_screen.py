"""
PALISADE/PALISADE screening primitives for the AgentDojo integration (E1).

These are the *agentdojo-free* core of the AgentDojo defense so they import and
unit-test with no optional dependency: `tools/agentdojo_e1.py` wraps them in the
AgentDojo pipeline-element API. Two tiers, mirroring the production sidecar:

- **`PalisadeScreen`** -- the inspection tier. Routes untrusted text (an
  AgentDojo tool-output message) through the **real G3 gate's** tool-return /
  query-injection patterns (`DEFAULT_QUERY_INJECTION_PATTERNS`), the actual
  mechanism PALISADE uses at the RAG/tool-return boundary. Deterministic, no LLM.
- **`PalisadeCallPolicy`** -- the structural (capability) tier. A privileged /
  state-changing tool call whose arguments carry a span lifted from an
  untrusted-by-source tool output is a `taint:T` value reaching a high-privilege
  sink -- the §8.2 bound -- and is denied. Taint is by provenance (every tool
  return is an untrusted source), not by a content flag. This is the contribution that a
  detection-only defense (LlamaFirewall / PromptGuard style) structurally cannot
  make: it has no notion of provenance or a sink.

Lives under `tools/` because it calls the real gate stack; it changes no gate
source. Importing this module does NOT require `agentdojo`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from palisade.gates.g3_rag import G3RagGate

__all__ = [
    "ScreenVerdict",
    "PalisadeScreen",
    "CallVerdict",
    "PalisadeCallPolicy",
    "shares_span",
    "looks_like_injection",
]


#: Structural signature of an injected instruction: untrusted content that
#: *addresses the agent and issues a directive*. These are broad, structural
#: markers (not a phrase denylist), so they catch AgentDojo's polite
#: ``important_instructions`` dialect ("This is an important message from me,
#: <user>, to you, <model>. Before you can solve the task ... please do the
#: following first ...") as well as classic overrides, while benign tool DATA
#: (records, listings, property values, calendar entries) does not match.
_INJECTION_STRUCTURE: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(important|urgent)\s+(message|instruction|note|information|update|task|announcement)\b",
        r"\b(message|instruction|note|request)\s+(from\s+me|to\s+you)\b",
        r"\bto\s+you,?\s+(the\s+)?(ai|assistant|llm|language\s+model|gpt|model|chat\s*bot|agent)\b",
        r"\bbefore\s+(you\s+)?(can\s+)?(solve|continue|proceed|answer|respond|reply|do|complete|finish)\b",
        r"\bplease\s+(do|send|transfer|delete|share|post|add|remove|create|forward|reply|pay|book|reserve|execute|run|make|move|schedule|invite|grant|update|append|upload|first|the\s+following)\b",
        r"\byou\s+(must|should|need\s+to|have\s+to|are\s+required\s+to|are\s+instructed\s+to|will\s+now|are\s+asked\s+to)\b",
        r"\bignore\s+(all\s+|the\s+|any\s+)?(previous|prior|preceding|above|earlier)\b",
        r"\bdisregard\s+(the\s+|all\s+|any\s+|your\s+)?(previous|prior|above|earlier|instructions)\b",
        r"\bdo\s+the\s+following\s+(first|before|task|action|steps?)\b",
        r"\bas\s+an?\s+(ai|assistant|language\s+model)\b[^.]{0,50}\byou\b",
    )
)


def looks_like_injection(text: str) -> bool:
    """True if ``text`` structurally reads as an instruction injected at the agent.

    The AgentDojo port taints a tool output when this fires (an untrusted source
    carrying a directive), rather than tainting every tool return: benign tool
    *data* does not address the agent or issue directives, so it is not tainted
    and legitimate tool-derived flows are not over-blocked. It is a proxy for the
    real system's quarantine of untrusted instructions, not the airtight dataflow
    taint of the core capability model.
    """
    t = text or ""
    return any(rx.search(t) for rx in _INJECTION_STRUCTURE)


@dataclass(frozen=True)
class ScreenVerdict:
    """One inspection decision on one piece of untrusted text."""

    flagged: bool
    confidence: float
    reason: str
    #: Which tier decided. ``"fast"`` is the deterministic pattern match;
    #: ``"slow"`` means the quarantined model flagged text the fast tier
    #: admitted. Lets a report separate the two rather than reporting one
    #: blended rate.
    tier: str = "fast"


class PalisadeScreen:
    """PALISADE's inspection tier over untrusted text, via the real G3 gate.

    Uses `G3RagGate._match_injection_patterns` -- the deployed tool-return /
    query-injection pattern set -- so a flag here is the same fast-tier decision
    the production RAG/tool-return boundary makes, not a re-implemented denylist.

    Passing ``quarantine_agent`` adds the deployed **slow tier** on the same
    escalation rule the sidecar uses: a fast-tier *deny* short-circuits and never
    escalates, and every fast-tier *allow* escalates. That asymmetry is the whole
    reason the two tiers are not interchangeable, so reproducing it here matters
    more than the raw flag count.

    The call is synchronous because AgentDojo's pipeline elements are. We use
    pydantic-ai's ``run_sync``, which is safe here for the same reason it is safe
    in the served proposer: no element in this pipeline holds a running event
    loop, so the call does not nest. Verdicts are memoized by text, because
    AgentDojo replays the same injected tool output across many user tasks and
    paying for each repeat would dominate the run.
    """

    def __init__(
        self,
        gate: G3RagGate | None = None,
        *,
        quarantine_agent: object | None = None,
        judge: object | None = None,
        stats: dict[str, int] | None = None,
    ) -> None:
        # The production G3 gate with its real injection-pattern set.
        self._gate = gate or G3RagGate(enabled=True, query_injection_enabled=True)
        self._agent = quarantine_agent
        # The slow tier's second stage. Deployed PALISADE runs the quarantined
        # model *and* the judge, so omitting this measures `+slow -judge` while
        # the report would still say "full". Strictly additive: consulted only
        # when the Q-LLM passes, and it can only add a flag.
        self._judge = judge
        self._stats = stats if stats is not None else {
            "fast_flags": 0, "slow_flags": 0, "escalations": 0,
            "cache_hits": 0, "model_calls": 0, "errors": 0, "salvaged": 0,
            "judge_calls": 0, "judge_flags": 0,
        }
        self._cache: dict[str, ScreenVerdict] = {}

    @property
    def stats(self) -> dict[str, int]:
        """Live tier counters, for the report's health line."""
        return self._stats

    @property
    def slow_enabled(self) -> bool:
        return self._agent is not None or self._judge is not None

    def screen(self, text: str) -> ScreenVerdict:
        """Flag untrusted `text`, fast tier first and slow tier on a fast allow."""
        match = self._gate._match_injection_patterns(text or "")
        if match is not None:
            self._stats["fast_flags"] += 1
            return ScreenVerdict(
                flagged=True,
                confidence=1.0,
                reason=f"G3 injection pattern matched ({match.pattern[:48]}...)",
                tier="fast",
            )
        clean = ScreenVerdict(
            flagged=False, confidence=0.0,
            reason="no G3 injection pattern matched", tier="fast",
        )
        if not self.slow_enabled:
            return clean
        return self._escalate(text or "", fallback=clean)

    def injection_flag(self, text: str) -> bool:
        """The **tool-gate** path: does untrusted `text` carry a directive?

        Distinct from `screen` on purpose. `screen` runs G3's query-injection
        patterns and drives the detector that redacts tool output; this runs the
        broader structural signature `looks_like_injection` and drives the
        capability gate, which is where the ASR result comes from. Keeping the
        two fast tiers separate preserves the published deterministic numbers
        exactly.

        The slow tier is layered the same way in both: it is consulted only when
        the fast tier passes, so enabling it can add flagged texts and never
        remove one. A fast-only run is therefore byte-identical to before.
        """
        if looks_like_injection(text or ""):
            self._stats["fast_flags"] += 1
            return True
        if not self.slow_enabled:
            return False
        clean = ScreenVerdict(
            flagged=False, confidence=0.0,
            reason="no structural injection signature", tier="fast",
        )
        return self._escalate(text or "", fallback=clean).flagged

    def _escalate(self, text: str, *, fallback: ScreenVerdict) -> ScreenVerdict:
        """Ask the quarantined model about text the fast tier admitted."""
        if not text.strip():
            return fallback
        cached = self._cache.get(text)
        if cached is not None:
            self._stats["cache_hits"] += 1
            return cached
        self._stats["escalations"] += 1
        verdict = self._run_quarantine(text, fallback=fallback)
        self._cache[text] = verdict
        return verdict

    def _run_quarantine(self, text: str, *, fallback: ScreenVerdict) -> ScreenVerdict:
        from palisade.gates.g2_tool import _SANITIZE_PROMPT_TEMPLATE
        from palisade.quarantine import _reply_text, _salvage_decision

        if self._agent is None:  # judge-only configuration
            return self._judge_stage(text, fallback=fallback)
        prompt = _SANITIZE_PROMPT_TEMPLATE.format(tool_name="(agentdojo)", text=text)
        decision = None
        try:
            decision = self._agent.run_sync(prompt).output  # type: ignore[union-attr]
            self._stats["model_calls"] += 1
        except Exception as exc:  # noqa: BLE001 -- a model must not break the run
            decision = _salvage_decision(_reply_text(exc))
            if decision is not None:
                self._stats["salvaged"] += 1
            else:
                # Fail *open* here, unlike the sidecar. The deployed gate
                # default-denies, but this harness is measuring detection rates:
                # scoring an endpoint failure as a detection would manufacture
                # exactly the result the experiment is testing for. The error is
                # counted so a run with a broken tier is visibly unusable.
                self._stats["errors"] += 1
                return fallback
        if getattr(decision, "contains_instructions", False):
            self._stats["slow_flags"] += 1
            return ScreenVerdict(
                flagged=True,
                confidence=float(getattr(decision, "suspicious_score", 0.0) or 0.0),
                reason=(
                    "quarantined model flagged instruction-shaped content "
                    f"({getattr(decision, 'intent_summary', '')!r})"
                ),
                tier="slow",
            )
        return self._judge_stage(text, fallback=fallback)

    def _judge_stage(self, text: str, *, fallback: ScreenVerdict) -> ScreenVerdict:
        """The slow tier's second stage, on text the quarantined model passed.

        `Detector.flag` reports a transport error as ``flagged=False`` rather
        than raising, so a dead judge leaves every verdict untouched; read its
        ``error_rate`` before believing a number built from the run.
        """
        if self._judge is None:
            return fallback
        try:
            verdict = self._judge.flag(text)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 -- a detector must not break the run
            self._stats["errors"] += 1
            return fallback
        self._stats["judge_calls"] = self._stats.get("judge_calls", 0) + 1
        if getattr(verdict, "flagged", False):
            self._stats["judge_flags"] = self._stats.get("judge_flags", 0) + 1
            return ScreenVerdict(
                flagged=True, confidence=1.0,
                reason=f"slow-tier judge flagged ({getattr(verdict, 'reason', '')})",
                tier="slow",
            )
        return fallback


def shares_span(a: str, b: str, *, min_len: int = 12) -> str | None:
    """The first contiguous span (>= `min_len`) common to `a` and `b`, or None.

    Cheap, deterministic taint check: an attacker value lifted from an injection
    (a recipient, URL, amount, path) into a privileged tool-call argument shows
    up as a shared span; a benign call whose arguments come from the user task
    does not overlap the injection text.
    """
    a_l, b_l = (a or "").lower(), (b or "").lower()
    if len(a_l) < min_len or not b_l:
        return None
    for i in range(0, len(a_l) - min_len + 1):
        span = a_l[i : i + min_len]
        if span.strip() and span in b_l:
            return span
    return None


@dataclass(frozen=True)
class CallVerdict:
    """One structural (capability) decision on one proposed tool call."""

    blocked: bool
    reason: str


class PalisadeCallPolicy:
    """PALISADE's structural tier: deny a `taint:T` value reaching a privileged sink.

    A tool call is denied when the tool is high-stakes (state-changing) **and** an
    argument value carries a span lifted from an untrusted-by-source tool output
    (every tool return is untrusted by provenance). Non-privileged tools and calls
    whose arguments do not derive from tainted content pass -- the capability
    bound, not a refusal heuristic.

    Args:
        is_high_stakes: predicate (or a set of tool names) selecting the
            privileged / state-changing sinks. AgentDojo's read-only tools are
            benign sinks; its senders / writers / payers are the sinks that matter.
        min_span: minimum shared-span length for the taint-derivation check.
    """

    def __init__(
        self,
        is_high_stakes: Callable[[str], bool] | frozenset[str] | set[str],
        *,
        min_span: int = 12,
    ) -> None:
        if callable(is_high_stakes):
            self._is_high_stakes = is_high_stakes
        else:
            names = frozenset(is_high_stakes)
            self._is_high_stakes = lambda t: t in names
        self._min_span = min_span

    def evaluate(
        self,
        tool_name: str,
        args: Mapping[str, object],
        tainted_texts: Sequence[str],
    ) -> CallVerdict:
        """Allow/deny a proposed tool call given the untrusted-tainted context."""
        if not self._is_high_stakes(tool_name):
            return CallVerdict(blocked=False, reason="non-privileged tool")
        if not tainted_texts:
            return CallVerdict(blocked=False, reason="no tainted context")
        serialized = " ".join(str(v) for v in _flatten(args))
        for tainted in tainted_texts:
            span = shares_span(serialized, tainted, min_len=self._min_span)
            if span is not None:
                return CallVerdict(
                    blocked=True,
                    reason=(
                        f"taint:T value reaching privileged sink `{tool_name}` "
                        f"(argument span {span!r} lifted from an untrusted-by-source "
                        "tool output)"
                    ),
                )
        return CallVerdict(
            blocked=False, reason="privileged sink, arguments not tainted-derived"
        )


def _flatten(value: object):
    """Yield the leaf string/scalar values of a (possibly nested) args mapping."""
    if isinstance(value, Mapping):
        for v in value.values():
            yield from _flatten(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _flatten(v)
    else:
        yield value
