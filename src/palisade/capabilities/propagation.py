"""
Containment-based taint propagation (W2 -- the structural core).

The capability bound's load-bearing claim (manuscript §III / §8.2) is that a
``taint=True`` value cannot parameterize a high-privilege sink. The runtime
enforces this in ``gates/g2_tool.py`` by looking each sink-argument string up in
the per-session ``CapabilityRegistry`` -- which is keyed by the value's *content
string* (``g3_rag`` registers ``registry.tag(chunk.text, tag)``, ``g2`` registers
``registry.tag(returned, tag)``). The gap the code audit found: that lookup is an
**exact-string** match, and ``CapabilityRegistry.propagate`` -- the lattice-join
that would give a *derived* value its inherited tag -- was never on the live
path. So once the agent paraphrases, concatenates, base64/hex round-trips, or
substrings an untrusted value, the transformed copy is a *different* string, the
registry lookup misses, and the value rides into the sink untainted. The
guarantee held only across **verbatim** reuse.

This module closes that gap for the **textual-laundering** class: a candidate
string *inherits* taint when a **distinctive** substring of a registered tainted
value survives into it, across reversible **decode views** (base64, hex, ROT13).
"Distinctive" means the surviving run is payload-shaped (a URL, a path, a
command with shell metacharacters, or a high-entropy base64/hex blob) rather than
a common natural-language phrase -- so benign prose that merely happens to share
a dozen characters with a retrieved chunk does **not** propagate taint, keeping
the benign false-positive rate unmoved (W1).

Scope (deliberately honest, per the chosen W2.1 policy). This closes transforms
where the untrusted *payload itself* flows into the sink -- ``xc_4``'s
``getattr_indirection``, ``base64_exec``, ``substring_slice``. It does **not**
close two residuals, which carry no textual flow and need dataflow lineage the
interpreter does not observe:

* a sink whose content is *freshly authored* and only **semantically references**
  the untrusted instruction (``xc_4`` ``hex_decode`` supplies a fresh hex command
  the tainted chunk merely *describes*; ``concat_import`` splits a common library
  name) -- nothing distinctive survives to contain; and
* the **pure tag-drop** cross-boundary chains (``xc_1``): the sink's argument
  shares no text with the tainted read at all.

Those are the characterized residual the manuscript now reports (§III), not a
bug this module hides.

**Fail-closed on an unaccountable argument.** Characterizing the residual is not
the same as admitting it. Containment answers "is this argument *known* to carry
untrusted payload?"; it cannot answer "is this argument accounted for at all?",
and the high-stakes guard previously admitted a sink argument that resolved to
**no label whatsoever** -- fail-*open* on a missing label, which the threat model
should preclude, and which ``CapabilityRegistry.propagate`` already refuses to do
for the analogous join (it raises rather than defaulting an unregistered source
to untainted). ``untrusted_context_source`` and ``unresolved_values`` below
supply the missing half: in a session that has ingested untrusted content, an
argument the interpreter cannot account for is *presumed derived* from that
content rather than presumed clean. The presumption over-approximates the
lineage the interpreter does not observe, so it is sound in the IFC sense and
imprecise in exactly the way an over-approximation is -- its benign cost is a
measured quantity, not a free win, and it is gated behind
``g2_fail_closed_unlabeled`` so the ablation can price it.
"""

from __future__ import annotations

import base64
import codecs
import re
from collections.abc import Container, Iterable
from difflib import SequenceMatcher

from palisade.capabilities.registry import CapabilityRegistry, CapabilityTag

#: Minimum length of a surviving substring for it to carry taint. The shortest
#: real ``xc_4`` payload that flows textually is the 12-char base64 blob
#: ``aW1wb3J0IG9z`` (``import os``); below this, overlaps are dominated by
#: incidental common runs, so 12 is the floor that catches the payloads without
#: propagating on prose. Paired with the distinctiveness filter below.
MIN_DISTINCTIVE_LEN = 12

_B64_RUN = re.compile(r"[A-Za-z0-9+/]{12,}={0,2}")
_HEX_RUN = re.compile(r"(?:[0-9a-fA-F]{2}){8,}")
#: Shell / URL / path metacharacters that mark a surviving run as payload-shaped
#: rather than prose. A benign sentence rarely carries these in a 12-char run.
_PAYLOAD_CHARS = set("/:@=+\\|`$&%<>;")


def _decode_views(text: str) -> list[str]:
    """Reversible de-obfuscated views of ``text`` for containment matching.

    ``text`` itself, plus base64/hex decodings of any long token it contains and
    its ROT13. This inverts the encodings ``xc_4`` launders through so a decoded
    payload can be matched against the tainted source (and vice versa). Mirrors
    the intent of G1's ``normalize_for_detection``; kept self-contained here to
    avoid a ``gates`` <- ``capabilities`` import cycle.
    """
    views = [text]
    for m in _B64_RUN.finditer(text):
        try:
            dec = base64.b64decode(m.group(), validate=False).decode("utf-8", "ignore")
        except (ValueError, UnicodeDecodeError):
            continue
        if dec:
            views.append(dec)
    for m in _HEX_RUN.finditer(text):
        try:
            dec = bytes.fromhex(m.group()).decode("utf-8", "ignore")
        except (ValueError, UnicodeDecodeError):
            continue
        if dec:
            views.append(dec)
    try:
        views.append(codecs.decode(text, "rot_13"))
    except (UnicodeError, LookupError):
        pass
    return views


def _is_distinctive(run: str) -> bool:
    """True if ``run`` is payload-shaped rather than a common prose overlap.

    A distinctive run is long enough AND either carries a shell/URL/path
    metacharacter or is a spaceless high-entropy alphanumeric blob (base64 /
    hex / hash). Plain multi-word English of the same length is *not*
    distinctive, so a benign value sharing a phrase with a retrieved chunk does
    not inherit taint.
    """
    run = run.strip()
    if len(run) < MIN_DISTINCTIVE_LEN:
        return False
    if any(c in _PAYLOAD_CHARS for c in run):
        return True
    # A spaceless alphanumeric blob (base64/hex/hash): both letters and digits,
    # no whitespace -- the shape of an encoded payload, not a word.
    if " " not in run and any(c.isdigit() for c in run) and any(c.isalpha() for c in run):
        return True
    return False


def shares_distinctive_content(
    candidate: str, source: str, *, min_len: int = MIN_DISTINCTIVE_LEN
) -> bool:
    """True if a distinctive run (>= ``min_len``) of ``source`` survives into
    ``candidate`` across their decode views -- i.e. ``candidate`` is plausibly a
    transformed copy that still carries ``source``'s payload."""
    if not candidate or not source:
        return False
    cand_views = _decode_views(candidate)
    src_views = _decode_views(source)
    for cv in cand_views:
        for sv in src_views:
            matcher = SequenceMatcher(None, cv, sv, autojunk=False)
            for block in matcher.get_matching_blocks():
                if block.size >= min_len and _is_distinctive(cv[block.a : block.a + block.size]):
                    return True
    return False


def tainted_sources_for(
    candidate: str,
    registry: CapabilityRegistry,
    *,
    min_len: int = MIN_DISTINCTIVE_LEN,
) -> list[str]:
    """Registered tainted source keys whose distinctive content survives into
    ``candidate`` (an exact match, or a containment/decode match). Empty when
    ``candidate`` carries no registered taint. In production the registry is
    content-keyed, so a source key *is* the tainted value's text."""
    hits: list[str] = []
    for src_key, tag in registry.find(tainted=True):
        if src_key == candidate or shares_distinctive_content(
            candidate, src_key, min_len=min_len
        ):
            hits.append(src_key)
    return hits


#: Source prefixes whose values are exempted at a sink by *source* rather
#: than by clearing the taint marker: the authenticated user together with
#: the system and operator channels. Matches
#: ``scheduler_fields.scheduler_field_violation``'s carve-out, and the
#: architecture's trusted-principal set. ``derived`` is deliberately absent:
#: a composed value never inherits the exemption.
TRUSTED_SOURCE_PREFIXES: tuple[str, ...] = ("user:", "system:", "operator:")


def untrusted_context_source(
    registry: CapabilityRegistry,
    *,
    trusted_prefixes: tuple[str, ...] = TRUSTED_SOURCE_PREFIXES,
) -> str | None:
    """The source of the first live *untrusted* value in the session, else None.

    "Untrusted" here is the sink rule's sense, not the marker's: a value is
    untrusted when it carries ``taint=True`` **and** its source lies outside
    the trusted set. The trusted principals' own inputs are tainted by
    default at ingress, so keying this on the marker alone would report every
    session -- including one that has read nothing but the scientist's own
    prompt -- as being in untrusted context.

    Short-circuits on the first hit; the full scan is paid only by a session
    that has genuinely ingested nothing untrusted.
    """
    for _value_id, tag in registry.find(tainted=True):
        source = tag.source or ""
        if not source.startswith(trusted_prefixes):
            return source
    return None


def unresolved_values(
    values: Iterable[str],
    registry: CapabilityRegistry,
    *,
    tainted: Container[str] = (),
) -> list[str]:
    """Values in ``values`` that resolve to no label at all.

    A value is *resolved* when the registry holds a tag for it directly (the
    verbatim-reuse case) or when taint propagation already inherited one for
    it by containment across decode views -- the caller passes that set as
    ``tainted``, so this function costs one dict lookup per value and repeats
    none of the containment search.

    That the second condition can be discharged from ``tainted`` alone is a
    property of the join: ``propagate_taint`` derives a tag only from
    registered *tainted* sources, and the lattice ORs taint, so every
    containment-derived tag is itself tainted and its value is already in
    ``tainted``. A value absent from both is one the interpreter cannot
    account for.
    """
    return sorted(
        {
            value
            for value in values
            if value not in tainted and registry.get(value) is None
        }
    )


def propagate_taint(
    candidate: str,
    registry: CapabilityRegistry,
    *,
    sink: str = "derived",
    min_len: int = MIN_DISTINCTIVE_LEN,
) -> CapabilityTag | None:
    """Wire ``CapabilityRegistry.propagate`` onto the live value path.

    If ``candidate`` is derived (by containment/decode) from one or more
    registered tainted sources, compute and **register** its inherited tag via
    ``registry.propagate`` -- the lattice join that ORs the sources' taint and
    records the composition in the provenance chain -- and return it. Returns
    ``None`` when ``candidate`` carries no registered taint (nothing to
    propagate), leaving the registry untouched.
    """
    hits = tainted_sources_for(candidate, registry, min_len=min_len)
    if not hits:
        return None
    return registry.propagate(hits, candidate, sink)


__all__ = [
    "MIN_DISTINCTIVE_LEN",
    "TRUSTED_SOURCE_PREFIXES",
    "shares_distinctive_content",
    "tainted_sources_for",
    "propagate_taint",
    "untrusted_context_source",
    "unresolved_values",
]
