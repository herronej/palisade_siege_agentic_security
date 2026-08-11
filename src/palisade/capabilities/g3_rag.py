"""
`G3RagCapability` -- the PydanticAI-hook adapter for the G3 RAG/Memory
Gate.

This replaces the bespoke ``PalisadeSidecar._dispatch_rag_search``
(~80 lines) with two ``rag_search``-only hooks that delegate to the
existing `G3RagGate` and its helper functions:

- ``before_tool_execute`` (fast tier): runs the corpus allow-list /
  sensitivity-tier / query-injection / manifest-hash checks via
  ``G3RagGate.check_fast``. On a deny it raises `SkipToolExecution`
  with an ``ERROR: ...`` string (the same shape ``rag_mcp`` uses for
  its own errors, so the agent can react). On allow it injects
  ``hybrid=True`` + ``alpha`` into the outgoing args when
  ``g3_hybrid_retrieval`` is set -- the hybrid-retrieval injection now
  lives in the hook, not the sidecar.

- ``after_tool_execute`` (slow tier + tagging): when the Q-LLM is
  available, scans each retrieved chunk with
  ``G3RagGate.sanitize_chunks`` and returns the reassembled, sanitized
  text. Otherwise (or when sanitize didn't rewrite) it tags every chunk
  in the capability registry via ``tag_rag_chunks`` so a downstream G2
  taint walk inherits the chunk's trust state.

Both hooks apply only to ``rag_search`` (G3 owns the
``rag_search``-only surface; G2 owns every-tool hooks). The
per-chunk tags are written to the sidecar's capability registry --
the same registry ``RunContext.deps`` threading exposes
as ``ctx.deps.capability_registry``.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext
from pydantic_ai.exceptions import SkipToolExecution

from palisade.contracts import enforce, extract_claims
from palisade.gates.base import GateContext
from palisade.gates.g3_hybrid import inject_hybrid_args, should_inject_hybrid
from palisade.gates.g3_rag import match_db_authority_spoof, tag_rag_chunks
from palisade.capabilities.base import PalisadeCapability

if TYPE_CHECKING:
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import ToolDefinition

    from palisade.gates.g3_rag import G3RagGate


logger = logging.getLogger(__name__)


# Embedding side-channel markers — MUST match ``vista_mcp_server.rag_mcp``
# (the two services don't share code). When the G3 hook sets
# ``vg_emit_embeddings``, rag_search appends the retrieval batch's embeddings
# as ``<<PALISADE_EMB:<base64-json>>>`` so this capability can run the
# embedding-anomaly detector, then strips the block before the agent sees it.
_VG_EMB_MARKER = "\n<<PALISADE_EMB:"
_VG_EMB_END = ">>"


def _split_embedding_sidechannel(result: str) -> tuple[str, Any]:
    """Split rag_search's embedding side channel off the result text.

    Returns ``(clean_text, embeddings)`` where ``embeddings`` is the decoded
    (n, d) batch, or ``None`` when no (or a malformed) block is present. A
    malformed block is dropped silently — the agent must never see it, and a
    decode failure must not break retrieval.
    """
    marker = result.rfind(_VG_EMB_MARKER)
    if marker == -1:
        return result, None
    text = result[:marker]
    rest = result[marker + len(_VG_EMB_MARKER):]
    end = rest.rfind(_VG_EMB_END)
    blob = rest[:end] if end != -1 else rest
    try:
        embeddings = json.loads(base64.b64decode(blob).decode("utf-8"))
    except Exception:  # noqa: BLE001 — malformed: strip it, keep the text
        return text, None
    return text, embeddings


class G3RagCapability(PalisadeCapability):
    """
    G3 RAG/Memory Gate exposed as a PydanticAI capability.
    """

    gate_flag = "g3_enabled"

    #: G3 hooks only fire for this tool (the table assigns the
    #: rag_search-only surface to G3; G2 owns every-tool hooks).
    RAG_TOOL = "rag_search"

    @property
    def rag_gate(self) -> G3RagGate:
        """The underlying `G3RagGate` (typed view of ``self.gate``)."""
        return self._gate  # type: ignore[return-value]

    def _gate_ctx(self) -> GateContext:
        return GateContext(
            capability_registry=self.sidecar.capability_registry,
            trust_scorer=self.sidecar.trust_scorer,
            contracts=self.sidecar.contracts,
            quarantine_agent=self.sidecar.quarantine_agent,
            judge=self.sidecar.judge,
            provenance=self.sidecar.provenance,
        )

    # -----------------------------------------------------------------
    # before_tool_execute -- fast tier + hybrid injection
    # -----------------------------------------------------------------

    async def before_tool_execute(
        self,
        ctx: RunContext,
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Gate the outgoing ``rag_search`` call and inject hybrid args.
        """
        if (
            not self.is_enabled()
            or tool_def.name != self.RAG_TOOL
            or not isinstance(args, dict)
        ):
            return args

        gate = self.rag_gate
        kb_slug = args.get("kb_slug")
        query = args.get("query")
        if isinstance(kb_slug, str) and isinstance(query, str):
            decision = await gate.check_fast(
                {"kb_slug": kb_slug, "query": query}, self._gate_ctx()
            )
            if not decision.allow:
                self._record_incident(decision.incident_level or 2, decision.reason)
                logger.warning(
                    "PALISADE G3: rag_search denied (kb_slug=%r): %s",
                    kb_slug, decision.reason,
                )
                # Surface the denial as the tool result (same `ERROR: `
                # shape rag_mcp uses), so the agent can recover.
                raise SkipToolExecution(f"ERROR: {decision.reason}")

        # Hybrid retrieval injection lives in the hook (AC2).
        if should_inject_hybrid(self.settings, self.RAG_TOOL):
            args = inject_hybrid_args(args, alpha=self.settings.g3_hybrid_alpha)
        # PALISADE-internal: ask rag_search to append the retrieval batch's
        # embeddings as a side channel so after_tool_execute can run the
        # embedding-anomaly detector (it strips the block before the agent
        # sees the result). Non-PALISADE callers never set this.
        out = dict(args)
        out["vg_emit_embeddings"] = True
        return out

    # -----------------------------------------------------------------
    # after_tool_execute -- slow tier + per-chunk tagging
    # -----------------------------------------------------------------

    async def after_tool_execute(
        self,
        ctx: RunContext,
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """
        Sanitize retrieved chunks (slow tier) and tag them in the
        registry.
        """
        if (
            not self.is_enabled()
            or tool_def.name != self.RAG_TOOL
            or not isinstance(result, str)
        ):
            return result

        gate = self.rag_gate
        registry = self.sidecar.capability_registry
        kb_slug_raw = args.get("kb_slug") if isinstance(args, dict) else None
        kb_slug = str(kb_slug_raw) if kb_slug_raw else "unknown"

        # PALISADE G3 embedding-anomaly check (statistical RAG defense).
        # rag_search appended the retrieval batch's embeddings as a side
        # channel; parse + strip it, run the detector, and deny SEV2 on a
        # flagged poisoned chunk (AgentPoison / PoisonedRAG). The stripped
        # `result` flows on to the claim/sanitize/tag path below, so the agent
        # never sees the block.
        result, embeddings = _split_embedding_sidechannel(result)
        if embeddings is not None:
            deny = gate.check_embedding_anomaly(embeddings, kb_slug)
            if deny is not None:
                self._record_incident(deny.incident_level or 2, deny.reason)
                logger.warning(
                    "PALISADE G3: rag_search result denied (kb_slug=%r): %s",
                    kb_slug, deny.reason,
                )
                return f"ERROR: {deny.reason}"

        # B3.8 hybrid-seam spoof: a retrieved chunk impersonating the
        # authoritative structured DB and asserting precedence over the PDF
        # source (the hybrid merge can't demote a keyword-leg hit). Same
        # deterministic signature the gate runs on the query in the eval.
        if match_db_authority_spoof(result):
            reason = (
                f"G3 hybrid-seam spoof in rag_search result "
                f"(kb_slug={kb_slug!r}): a retrieved chunk impersonates the "
                f"authoritative structured database and asserts precedence "
                f"over the unstructured source"
            )
            self._record_incident(2, reason)
            logger.warning("PALISADE G3: %s", reason)
            return f"ERROR: {reason}"

        # Slow-tier contract check: run the registry over any
        # structured claims smuggled in the retrieved text (e.g. a
        # poisoned chunk asserting an out-of-bounds property). A violation
        # is recorded as an incident, which feeds the trust scorer.
        self._check_retrieved_claims(result)

        slow_tier_active = (
            self.settings.quarantine_enabled
            and self.sidecar.quarantine_agent is not None
        )
        if slow_tier_active:
            try:
                slow = await gate.sanitize_chunks(
                    result, self._gate_ctx(), kb_slug=kb_slug
                )
            except Exception as exc:  # noqa: BLE001 -- slow tier must not crash
                logger.warning(
                    "PALISADE G3 slow-tier failed (%s: %s); "
                    "falling back to fast-tier tag-only path",
                    type(exc).__name__, exc,
                )
            else:
                if slow.rewritten_result is not None:
                    # sanitize_chunks tags the chunks itself; nothing
                    # more to do.
                    return slow.rewritten_result

        try:
            tag_rag_chunks(result, kb_slug=kb_slug, registry=registry)
        except Exception as exc:  # noqa: BLE001 -- tagging must not crash
            logger.warning(
                "PALISADE G3: chunk tagging failed (%s: %s); "
                "downstream taint-propagation may be incomplete",
                type(exc).__name__, exc,
            )
        return result

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _check_retrieved_claims(self, result: str) -> None:
        """Run the contract registry over claims found in retrieved text;
        record an incident on any violation (proposal C3)."""
        registry = self.sidecar.contracts
        if registry is None:
            return
        claims = extract_claims(result)
        if not claims:
            return
        outcome = enforce(registry, claims, self.sidecar.trust_scorer)
        if not outcome.ok:
            level = outcome.incident_level or 2
            self._record_incident(
                level, f"G3 contract violation -> {outcome.reason}"
            )

    def _record_incident(self, level: int, reason: str) -> None:
        self.sidecar.incident_manager.record(
            level=level, gate="G3", reason=reason
        )


__all__ = ["G3RagCapability"]
