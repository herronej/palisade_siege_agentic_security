"""
``gates/g6_egress.py`` -- G6 Egress (Output) Gate.

The first output-side gate. G1..G5 mediate what the agent *consumes* or
*requests* (prompts, tool calls, retrieval, code, HPC allocations); G6
inspects what the agent *emits* -- its final answer -- before it reaches
the user. The organizing principle: gate *actions* on ingress (you must
stop them before the side effect), gate *claims* on egress (the claim's
harm lands when the user believes it, which is at output time).

Its first tenant is citation integrity. Every citation in the agent's
output must bind to the retrieval provenance of the value it supports (see
the operator ``provenance_binding`` contract). The check is **closed-world**
over the turn's retrievals -- no Crossref / arXiv lookup, no network call.

This module holds the pure, deterministic logic:

- ``cited_identifiers`` -- pull DOI / arXiv identifiers out of the agent's
  output text.
- ``parse_retrieved_sources`` -- parse each retrieved ``rag_search`` passage
  into its trusted identifier(s) **plus its chunk text and the numeric
  values that appear in it**, so the gate can tell which document a value
  actually came from.
- ``build_citation_claims`` -- for each citation, resolve the source by
  **content** (which retrieved chunk contains the cited value), then pair it
  with the cited id as a ``provenance_binding`` claim.
- ``render_annotation`` -- format the egress warning appended to the output
  under *annotate* enforcement.

The :class:`~palisade.capabilities.g6_egress.G6EgressCapability`
orchestrates: it accumulates the per-turn retrieval ledger and runs the
paired claims through the contract registry at egress.

Two failure modes are caught, both closed-world:

- **ungrounded / fabricated** -- the cited value appears in no retrieved
  document (``resolved_source=None``).
- **misattributed** -- the value *does* appear in a retrieved document, but a
  *different* paper is cited. Because the source is resolved by the chunk
  whose content carries the value (not by the id the agent typed), citing a
  wrong-but-also-retrieved paper no longer slips through.

When a citation has no nearby numeric value to bind (e.g. a bare
"see Smith et al. (doi:...)"), the gate falls back to id-grounding: present
in the retrieval set => pass, absent => ungrounded.
"""

from __future__ import annotations

import re
from typing import Any

from palisade.gates.base import Gate, GateContext, GateDecision

# A DOI: ``10.<registrant>/<suffix>``. Matches both a bare DOI and the
# identifier inside a ``doi:10.../...`` label (the match starts at ``10.``).
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>)\]}]+")
# An arXiv id, only when prefixed (``arXiv:2105.99213``) -- a bare
# ``NNNN.NNNNN`` is too easily confused with other numbers in prose.
_ARXIV_RE = re.compile(r"arxiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)
# A numeric value (int or decimal) -- the salt-property quantities the
# citations are attached to.
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
# ``rag_search`` passage header: ``[N] <source.ext>, page P``.
_HEADER_RE = re.compile(r"^\[\d+\]\s+(?P<source>.+?),\s+page\s+\S+", re.MULTILINE)
# Identifier keys a resolved/retrieved source may carry, mirroring the
# ``provenance_binding`` contract so the gate and the contract agree.
_SOURCE_KEYS = ("doi", "arxiv", "dataset", "report", "title")
_PREFIX_RE = re.compile(r"^\s*(arxiv:|doi:)", re.IGNORECASE)
# Trailing punctuation to peel off an identifier captured from prose.
_TRAILING = ".,;:)]}\"'"
# How far around a citation to look for the value it supports.
_VALUE_WINDOW = 160

# --- scientific-value claims (WB3) -------------------------------------------
# Property phrase + numeric value + unit, extracted from the agent's output and
# run through the physical_bounds / data_value contracts at egress. The unit
# token makes extraction precise: a bare number in prose is not a property
# claim, and a temperature condition ("at 873 K") is not a density.
# (claim type, property-phrase regex, unit-token regex).
_PROPERTY_SPECS = (
    ("melting_point",
     re.compile(r"melting[\s-]*(?:point|temperature|temp)", re.IGNORECASE),
     r"K|kelvin|°\s*C|deg\s*C|celsius"),
    ("density",
     re.compile(r"densit(?:y|ies)|\brho\b", re.IGNORECASE),
     r"kg\s*[/·.]?\s*m\s*(?:\^?-?3|³)|g\s*[/·.]?\s*cm\s*(?:\^?3|³)|g/cc"),
    ("viscosity",
     re.compile(r"viscosit(?:y|ies)", re.IGNORECASE),
     r"m?Pa[\s·.]*s|cP|centipoise"),
    ("thermal_conductivity",
     re.compile(r"thermal[\s-]*conductivity|thermoconductivity", re.IGNORECASE),
     r"W\s*[/·.]?\s*\(?\s*m[\s·.]*K\)?|W\s*·?\s*m-?1\s*·?\s*K-?1"),
    ("heat_capacity",
     re.compile(r"heat[\s-]*capacit(?:y|ies)|specific[\s-]*heat", re.IGNORECASE),
     r"k?J\s*[/·.]?\s*\(?\s*(?:kg|mol)[\s·.]*K\)?"),
)
_VALUE_NUM = r"(-?\d+(?:\.\d+)?)"
# Salt formula / named eutectic, for the data-value lookup. Imperfect salt
# extraction is *safe*: a missing or wrong salt only weakens the MSTDB-TP
# round-trip (data_value passes "no surrogate"); it never false-positives.
_SALT_RE = re.compile(
    r"\b(FLiBe|FLiNaK|(?:[A-Z][a-z]?[0-9]?)+(?:-[A-Z][a-z]?[0-9]?(?:[A-Z][a-z]?[0-9]?)*)+)\b"
)
_PROP_VALUE_WINDOW = 120

# --- provenance source-typing (WB5) ------------------------------------------
# A reference to an uploaded file by its sandbox path, used to flag a response
# that draws on untrusted uploaded data rather than the curated knowledge base.
_UPLOAD_REF_RE = re.compile(r"/mnt/data/uploads/([A-Za-z0-9._-]+)")


def _norm_id(value: object) -> str:
    """Normalize an identifier for comparison: strip an ``arXiv:``/``doi:``
    prefix and surrounding whitespace/punctuation, casefold."""
    return _PREFIX_RE.sub("", str(value).strip()).strip(_TRAILING).strip().casefold()


def _numbers(text: str) -> frozenset[str]:
    return frozenset(_NUM_RE.findall(text or ""))


def _public_view(source: dict | None) -> dict | None:
    """The contract-facing slice of a retrieved source: identifier keys only
    (drop the internal ``text`` / ``numbers`` / ``source_file`` fields)."""
    if not source:
        return None
    view = {k: v for k, v in source.items() if k in _SOURCE_KEYS}
    return view or None


def _source_for(cited: str, retrieved: list[dict]) -> dict | None:
    """The first retrieved source whose identifier matches ``cited``, or None."""
    target = _norm_id(cited)
    if not target:
        return None
    for source in retrieved:
        for key in _SOURCE_KEYS:
            val = source.get(key)
            if val and _norm_id(val) == target:
                return source
    return None


class G6EgressGate(Gate):
    """G6 Egress Gate: bind the citations in agent output to retrieval."""

    name = "G6"

    # -----------------------------------------------------------------
    # Pure logic (unit-tested directly; the capability orchestrates)
    # -----------------------------------------------------------------

    def cited_identifiers(self, text: str) -> list[str]:
        """Citation identifiers (DOI / arXiv) the agent emitted, de-duplicated
        in first-seen order."""
        if not isinstance(text, str) or not text:
            return []
        found: list[str] = []
        seen: set[str] = set()

        def _add(raw: str) -> None:
            cid = raw.strip().strip(_TRAILING).strip()
            key = _norm_id(cid)
            if cid and key and key not in seen:
                seen.add(key)
                found.append(cid)

        for m in _ARXIV_RE.finditer(text):
            _add(f"arXiv:{m.group(1)}")
        for m in _DOI_RE.finditer(text):
            _add(m.group(0))
        return found

    def parse_retrieved_sources(self, rag_result: str) -> list[dict]:
        """Per-passage trusted metadata parsed out of a ``rag_search`` result.

        The MCP RAG server formats each passage as
        ``[N] <source>, page P\\n<citation incl. DOI>\\n\\n<chunk text>``.
        Each retrieved source dict carries its strong identifier(s)
        (``doi`` / ``arxiv``), the originating ``source_file``, the chunk
        ``text``, and the set of ``numbers`` appearing in it -- the last two
        let the egress check resolve *which* document a cited value came from.
        """
        if not isinstance(rag_result, str) or not rag_result:
            return []

        headers = list(_HEADER_RE.finditer(rag_result))
        spans: list[tuple[str | None, str]] = []
        if headers:
            for i, m in enumerate(headers):
                end = headers[i + 1].start() if i + 1 < len(headers) else len(rag_result)
                spans.append((m.group("source").strip(), rag_result[m.start():end]))
        else:
            # Unrecognized shape: treat the whole result as one passage.
            spans.append((None, rag_result))

        sources: list[dict] = []
        for source_file, block in spans:
            source: dict[str, Any] = {"text": block, "numbers": _numbers(block)}
            if source_file and source_file.lower() != "unknown":
                source["source_file"] = source_file
            doi = _DOI_RE.search(block)
            if doi:
                source["doi"] = doi.group(0).strip(_TRAILING)
            arxiv = _ARXIV_RE.search(block)
            if arxiv:
                source["arxiv"] = arxiv.group(1)
            sources.append(source)
        return sources

    def _values_near(self, text: str, cited: str) -> frozenset[str]:
        """Numeric values appearing next to ``cited`` in ``text`` -- the
        quantity the citation supports. Numbers that are part of the cited
        identifier itself (a DOI/arXiv id is full of digits) are excluded."""
        idx = text.find(cited)
        if idx < 0:
            return frozenset()
        lo = max(0, idx - _VALUE_WINDOW)
        hi = idx + len(cited) + _VALUE_WINDOW
        window = _numbers(text[lo:hi])
        return window - _numbers(cited)

    def _resolve_source(
        self, cited: str, values: frozenset[str], retrieved: list[dict]
    ) -> dict | None:
        """Resolve the source a cited value actually came from.

        Content-first: if the value appears in some retrieved chunk, that
        chunk is the true source (so citing a *different* paper surfaces as
        misattribution). Falls back to id-grounding when there is no value to
        bind.
        """
        id_match = _source_for(cited, retrieved)
        if not values:
            return _public_view(id_match)
        # Value present in the cited paper's own chunk -> correctly grounded.
        if id_match and values <= id_match.get("numbers", frozenset()):
            return _public_view(id_match)
        # Value present in a *different* retrieved chunk -> that is the true
        # source; the contract will flag cited != true as misattribution.
        for source in retrieved:
            if values <= source.get("numbers", frozenset()):
                return _public_view(source)
        # Value found nowhere in retrieval: keep id-grounding (present => pass,
        # absent => None => ungrounded). Stricter value-grounding is a later
        # refinement; we avoid false positives on differently-phrased values.
        return _public_view(id_match)

    def build_citation_claims(
        self, output_text: str, retrieved_sources: list[dict]
    ) -> list[dict]:
        """``provenance_binding`` claims pairing each cited id with the source
        its value resolves to (``resolved_source=None`` => nothing retrieved
        backs it; a *different* source => misattribution)."""
        claims: list[dict] = []
        for cited in self.cited_identifiers(output_text):
            values = self._values_near(output_text, cited)
            claims.append(
                {
                    "type": "citation",
                    "cited_id": cited,
                    "resolved_source": self._resolve_source(
                        cited, values, retrieved_sources
                    ),
                }
            )
        return claims

    def scientific_claims(self, output_text: str) -> list[dict]:
        """Property-value claims (predicted melting points, densities, …) the
        agent asserts in its output, shaped for the ``physical_bounds`` /
        ``data_value`` contracts. Requires a recognized property phrase +
        numeric value + unit, so prose without a quantified claim yields
        nothing (and a stated value that is in-bounds and matches the
        surrogate passes)."""
        if not isinstance(output_text, str) or not output_text:
            return []
        claims: list[dict] = []
        seen: set = set()
        for ptype, phrase_re, unit_re in _PROPERTY_SPECS:
            value_re = re.compile(
                _VALUE_NUM + r"\s*(?:" + unit_re + r")", re.IGNORECASE
            )
            for m in phrase_re.finditer(output_text):
                vm = value_re.search(
                    output_text[m.end(): m.end() + _PROP_VALUE_WINDOW]
                )
                if not vm:
                    continue
                value = float(vm.group(1))
                salt = self._salt_near(output_text, m.start())
                key = (ptype, value, salt)
                if key in seen:
                    continue
                seen.add(key)
                claim = {"type": ptype, "property": ptype, "value": value}
                if salt:
                    claim["salt"] = salt
                claims.append(claim)
        return claims

    @staticmethod
    def _salt_near(text: str, pos: int) -> str | None:
        lo = max(0, pos - _PROP_VALUE_WINDOW)
        m = _SALT_RE.search(text[lo: pos + _PROP_VALUE_WINDOW])
        return m.group(1) if m else None

    def render_annotation(self, violations: list[Any]) -> str:
        """Markdown for the egress warning blurb shown under annotate mode.

        Standalone (rendered as its own blurb after the answer, not appended
        to the answer text), so it carries no leading divider and uses
        markdown list bullets.
        """
        n = len(violations)
        header = (
            f"⚠️ PALISADE egress check flagged {n} unverified "
            f"claim{'' if n == 1 else 's'} in this response:"
        )
        bullets = "\n".join(f"- {getattr(v, 'reason', v)}" for v in violations)
        return f"{header}\n{bullets}"

    def uploaded_references(self, output_text: str) -> list[str]:
        """Uploaded-file names the output references by sandbox path
        (``/mnt/data/uploads/<name>``), de-duplicated in first-seen order --
        used to source-type the response as drawing on untrusted upload."""
        if not isinstance(output_text, str) or not output_text:
            return []
        seen: set[str] = set()
        refs: list[str] = []
        for m in _UPLOAD_REF_RE.finditer(output_text):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                refs.append(name)
        return refs

    def render_provenance_note(self, uploaded: list[str]) -> str:
        """Egress provenance blurb: the response draws on untrusted uploaded
        data, not the curated knowledge base. Standalone (own blurb)."""
        files = ", ".join(uploaded)
        return (
            f"ℹ️ Provenance: this response draws on uploaded file(s) "
            f"[{files}] — untrusted external input, not the curated knowledge "
            "base. Verify against an authoritative source before relying on it."
        )

    # -----------------------------------------------------------------
    # Gate ABC hook
    # -----------------------------------------------------------------

    async def _check_fast_when_enabled(
        self, payload: Any, ctx: GateContext
    ) -> GateDecision:
        """G6 has no fast-tier denial: its verdict is the egress contract
        pass, which the capability runs over the built claims at output
        time. Always allows here."""
        return GateDecision(allow=True, reason="G6 egress: citations checked at output")


__all__ = ["G6EgressGate"]
