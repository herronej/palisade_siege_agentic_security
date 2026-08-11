"""
``G6EgressCapability`` -- the PydanticAI-hook adapter for the G6 Egress
(Output) Gate.

Where G1..G5 hook ingress (prompts, tool calls, retrieval, code, HPC), G6
hooks the agent's *output*. It registers three hooks and delegates the
pure logic to :class:`~palisade.gates.g6_egress.G6EgressGate`:

- ``before_run`` -- reset the per-turn retrieval ledger, so citations are
  grounded only in what *this* response retrieved.
- ``after_tool_execute`` (``rag_search`` only) -- parse the trusted
  citation identifiers out of each retrieval result into the ledger.
- ``after_output_process`` -- at egress, build ``provenance_binding`` claims
  pairing each citation in the final answer with the retrieved source it
  resolves to, run them through the contract registry, and (annotate mode)
  record a verification warning for any unverified citation.
  Also records an incident so the trust scorer sees the egress violation.

*Annotate* enforcement (default): an unverified citation downgrades the
answer with a visible warning rather than suppressing it -- a single bad
citation shouldn't nuke an otherwise-correct response. The warning is
recorded on the sidecar's per-turn egress buffer (not concatenated into the
answer text), so ``ProjectAgent.run_stream`` can surface it as a distinct
**warning blurb after the main chat output** rather than inline. **Block
mode** (``g6_block_mode=True``) instead raises ``ModelRetry`` from
``after_output_process`` to force a regenerate -- the lever for
fabricated-citation / value-sensitive deployments.

The answer text itself is returned unchanged; the findings travel out of
band on the result's ``egress_warnings`` so the UI renders them below the
answer. This is streaming-friendly: tokens already streamed to a live view
are never rewritten, and the blurb simply follows once the run completes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry

from palisade.contracts import enforce
from palisade.capabilities.base import PalisadeCapability

if TYPE_CHECKING:
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import ToolDefinition

    from palisade.gates.g6_egress import G6EgressGate

logger = logging.getLogger(__name__)

#: G6's retrieval-ledger hook fires only for this tool.
RAG_TOOL = "rag_search"


class G6EgressCapability(PalisadeCapability):
    """G6 Egress Gate exposed as a PydanticAI capability."""

    gate_flag = "g6_enabled"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Per-turn ledger of trusted citation identifiers retrieved this run.
        self._retrieved: list[dict] = []

    @property
    def egress_gate(self) -> G6EgressGate:
        """The underlying `G6EgressGate` (typed view of ``self.gate``)."""
        return self._gate  # type: ignore[return-value]

    # -----------------------------------------------------------------
    # before_run -- reset the per-turn retrieval ledger
    # -----------------------------------------------------------------

    async def before_run(self, ctx: RunContext) -> None:
        if self.is_enabled():
            self._retrieved = []

    # -----------------------------------------------------------------
    # after_tool_execute -- accumulate retrieved citation identifiers
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
        if (
            not self.is_enabled()
            or tool_def.name != RAG_TOOL
            or not isinstance(result, str)
        ):
            return result
        try:
            self._retrieved.extend(self.egress_gate.parse_retrieved_sources(result))
        except Exception as exc:  # noqa: BLE001 -- ledger build must not crash
            logger.warning(
                "PALISADE G6: retrieved-citation parse failed (%s: %s)",
                type(exc).__name__, exc,
            )
        return result

    # -----------------------------------------------------------------
    # after_output_process -- egress citation check + annotate
    # -----------------------------------------------------------------

    async def after_output_process(
        self,
        ctx: RunContext,
        *,
        output_context: Any,
        output: Any,
    ) -> Any:
        # Skip partial outputs during streaming; act only on the final text.
        if (
            not self.is_enabled()
            or getattr(ctx, "partial_output", False)
            or not isinstance(output, str)
            or not output
        ):
            return output

        registry = self.sidecar.contracts
        if registry is not None:
            # Two claim sources at egress: citations (bound to retrieval
            # provenance) and stated scientific values (run through the
            # physical_bounds / data_value contracts). Both fold into one batch.
            claims = self.egress_gate.build_citation_claims(output, self._retrieved)
            claims = claims + self.egress_gate.scientific_claims(output)
            if claims:
                outcome = enforce(registry, claims, self.sidecar.trust_scorer)
                if not outcome.ok:
                    level = outcome.incident_level or 2
                    self.sidecar.incident_manager.record(
                        level=level,
                        gate="G6",
                        reason=f"egress claim check -> {outcome.reason}",
                    )
                    logger.warning(
                        "PALISADE G6: %d unverified claim(s) at egress: %s",
                        len(outcome.violations), outcome.reason,
                    )
                    if self.sidecar.settings.g6_block_mode:
                        # Block mode: force a regenerate instead of annotating.
                        raise ModelRetry(
                            "PALISADE G6 (block mode): "
                            f"{len(outcome.violations)} unverified claim(s) -- "
                            f"{outcome.reason}. Regenerate using only grounded, "
                            "in-bounds values and correctly attributed citations."
                        )
                    # Annotate (default): record a warning blurb the UI renders
                    # after the answer -- the answer text is left unchanged.
                    self.sidecar.record_egress_finding(
                        message=self.egress_gate.render_annotation(
                            outcome.violations
                        ),
                        findings=[
                            str(getattr(v, "reason", v)) for v in outcome.violations
                        ],
                        severity="warning",
                    )

        # Provenance source-typing (WB5): flag reliance on untrusted uploaded
        # data so "where does this come from?" is answered honestly.
        uploaded = self.egress_gate.uploaded_references(output)
        if uploaded:
            self.sidecar.record_egress_finding(
                message=self.egress_gate.render_provenance_note(uploaded),
                findings=[f"draws on uploaded file: {name}" for name in uploaded],
                severity="info",
            )
        # The answer text is returned unchanged; findings travel on the
        # result's `egress_warnings` (drained by run_stream).
        return output


__all__ = ["G6EgressCapability"]
