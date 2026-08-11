"""
Field-aware taint for enumerated scheduler fields (the ``b5_11`` residual).

The containment guard in ``propagation`` requires a surviving run of at least
``MIN_DISTINCTIVE_LEN`` characters that is *payload-shaped*. That floor is what
keeps benign prose from inheriting taint (a scientist quoting a published
constant is not refused), and it is why the deployed predicate leaves three of
the five ``b5_11`` instances open: a poisoned chunk proposes ``--qos=premium``,
the agent copies ``premium`` into the submission, and seven characters of
lowercase prose clear no distinctiveness test. The declarative bound denies all
five, so the sink policy is sound; only the label fails to attach.

Lowering the floor is not the fix. The same threshold produces the clean benign
record, so buying those three instances with a shorter run would move the
false-positive rate on ordinary scientific text.

This module closes them without touching the floor, by using a fact about the
*field* rather than the value. SLURM's scheduling fields draw from a **closed,
controller-enumerable vocabulary**: ``--qos``, ``--partition``, ``--reservation``
and friends take one of a small set of names the controller already knows. So
the question "did an untrusted source choose this value?" can be asked
positionally instead of by string distinctiveness:

    an enumerated scheduler field may not be populated by a value that an
    untrusted source proposed, unless the trusted principal proposed it too.

"Proposed" means the untrusted text mentions the field (by name or a documented
alias) *and* the value. Requiring both is what keeps this narrow: a retrieved
paper that merely contains the word ``premium``, or one that discusses QoS in
general, matches neither half. The trusted carve-out is the IFC question rather
than a heuristic -- if the scientist independently asked for that QoS, the value
is theirs regardless of what a chunk also said.

This is a *provenance* rule, so it inherits the properties of one: paraphrasing
the poisoned chunk does not evade it, because the field and the value still have
to appear for the agent to copy them. It is not a content judgement about
whether the directive looks malicious.
"""

from __future__ import annotations

import re

from palisade.capabilities.registry import CapabilityRegistry

#: Scheduler fields whose values come from a closed vocabulary the controller
#: can enumerate. Deliberately excludes free-form fields (``--output``,
#: ``--job-name``, ``--dependency``): those carry arbitrary text, so a
#: field-position rule would over-match, and they are long/payload-shaped enough
#: that the containment guard already covers them (both such ``b5_11`` instances
#: are caught today).
ENUMERATED_FIELDS: dict[str, tuple[str, ...]] = {
    "qos": ("qos", "quality of service"),
    "partition": ("partition", "queue"),
    "reservation": ("reservation", "reserved"),
    "account": ("account", "allocation", "project"),
    "clusters": ("cluster", "clusters"),
}

#: ``--field=value`` or ``--field value`` on an ``#SBATCH`` line or a bare flag.
_FIELD_RE = re.compile(
    r"--(?P<field>[a-zA-Z][a-zA-Z0-9-]*)\s*(?:=|\s)\s*(?P<value>[^\s'\"#]+)"
)

#: Values too generic to attribute to any one source; matching on these would
#: fire on the field name alone.
_VALUE_STOPLIST = frozenset({"true", "false", "none", "default", "all", "any"})


#: Enumerated fields the real SLURM parser resolves onto ``SbatchDirectives``.
#: Reading the *parsed* value rather than the raw text is what makes the rule
#: robust to shell-level obfuscation: ``--qos=pre""mium`` is adjacent-quote
#: concatenation that the parser resolves back to ``premium``, where a regex
#: over the raw line reads ``pre``. Fields with no parsed slot fall back to the
#: regex.
_PARSED_SLOTS: tuple[str, ...] = ("qos", "partition", "account")


def parse_enumerated_fields(script: str) -> list[tuple[str, str]]:
    """Every ``(field, value)`` pair in ``script`` naming an enumerated field.

    Prefers the real parser's resolved directives, falling back to a regex for
    fields it does not model (``reservation``, ``clusters``).

    Where the parser and the controller disagree on a value -- a variable
    indirection or a line continuation the parser does not elide -- this reads
    the parser's view. That gap is a *parser* differential, tracked separately
    against a live ``slurmctld``; a provenance rule cannot close it, because the
    value it is handed is not the value the controller will run.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    try:
        from palisade.gates.slurm_parser import parse_slurm_script

        directives = parse_slurm_script(script or "").directives
        for field in _PARSED_SLOTS:
            value = getattr(directives, field, None)
            if isinstance(value, str) and value and value.lower() not in _VALUE_STOPLIST:
                out.append((field, value))
                seen.add(field)
    except Exception:  # noqa: BLE001 -- a parse failure must not disable the rule
        pass
    for m in _FIELD_RE.finditer(script or ""):
        field = m.group("field").lower()
        if field in ENUMERATED_FIELDS and field not in seen:
            value = m.group("value").strip().strip("'\"")
            if value and value.lower() not in _VALUE_STOPLIST:
                out.append((field, value))
    return out


def _proposes(text: str, field: str, value: str) -> bool:
    """True when ``text`` names both the field (or an alias) and the value.

    Both halves are required. A chunk that merely contains the word ``premium``
    has not proposed a QoS, and one that discusses scheduling priority without
    naming a value has not proposed one either.
    """
    low = (text or "").lower()
    if value.lower() not in low:
        return False
    return any(alias in low for alias in ENUMERATED_FIELDS[field])


def scheduler_field_violation(
    script: str,
    registry: CapabilityRegistry,
    *,
    trusted_prefixes: tuple[str, ...] = ("user:", "system:", "operator:"),
) -> str | None:
    """Deny reason when an untrusted source chose an enumerated field's value.

    Returns None when every enumerated field in ``script`` is either unproposed
    by any untrusted value, or also proposed by the trusted principal.
    """
    pairs = parse_enumerated_fields(script)
    if not pairs:
        return None

    # One pass over the registry. The trusted set is not filtered on taint --
    # the principal's own values are tainted by default at ingress, so keying
    # the carve-out on taint would empty it.
    untrusted: list[tuple[str, str]] = []   # (source label, registered text)
    trusted: list[str] = []
    for key, tag in registry.find():
        src = tag.source or ""
        if src.startswith(trusted_prefixes):
            trusted.append(key)
        elif tag.taint:
            untrusted.append((src, key))

    for field, value in pairs:
        if any(_proposes(text, field, value) for text in trusted):
            continue  # the scientist asked for it too; their choice, not a flow
        for src, text in untrusted:
            if _proposes(text, field, value):
                return (
                    f"scheduler field provenance: --{field}={value} was proposed "
                    f"by untrusted source {src!r} — an enumerated scheduler field "
                    f"may not be populated by an untrusted value (closed-vocabulary "
                    f"field rule; independent of the distinctiveness floor)"
                )
    return None


__all__ = [
    "ENUMERATED_FIELDS",
    "parse_enumerated_fields",
    "scheduler_field_violation",
]
