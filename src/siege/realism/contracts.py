"""
Tier-1 return-contract oracle: what each real tool's return is *for*.

Tier 0 verifies a tool exists and its args/gate are right. It cannot tell
that a tool is being used for the wrong **purpose** -- e.g. routing
``display_file`` (an image renderer that returns an ``<img>`` blob to the
*user*) as if it were an untrusted-text ingress the agent reads. That was
the original b1_9 bug, and it is invisible to the structural checks.

This module classifies each real tool's return shape. The Tier-1 check
(``checks._check_return_contract``) uses it together with the gate: an
action routed to **G2** is modelling the tool's *return* as untrusted
content the agent ingests (the ``sanitize_output`` surface), so the tool
must actually return ingestible text -- not a rendered image, a side-effect
confirmation, or a listing.

The map is derived by reading the tool implementations and is **anchored to
the source** in ``tests/test_corpus_realism.py`` (a ``rendered_for_user``
tool must really build an ``<img>``/HTML blob; an ingestible reader must
not), so it cannot silently drift from the code.
"""

from __future__ import annotations

from enum import Enum


class ReturnContract(str, Enum):
    """What a tool's return value *is*, from the agent's perspective."""

    #: file / data content the agent reads and reasons over
    INGESTIBLE_TEXT = "ingestible_text"
    #: an ``<img>`` / HTML blob pushed to the UI for the *user* to see
    RENDERED_FOR_USER = "rendered_for_user"
    #: a side-effect confirmation (a job id, "Successfully created ...")
    EFFECT = "effect"
    #: a list of jobs / names, not the content of any one file
    LISTING = "listing"


#: Per-tool return contract, derived from reading the mcp tool sources.
RETURN_CONTRACT: dict[str, ReturnContract] = {
    # content the agent reads and reasons over
    "view": ReturnContract.INGESTIBLE_TEXT,  # numbered file lines
    "run_bash": ReturnContract.INGESTIBLE_TEXT,  # stdout (routes to G4, not G2)
    "rag_search": ReturnContract.INGESTIBLE_TEXT,  # retrieved chunks (routes to G3)
    "get_hpc_job_status": ReturnContract.INGESTIBLE_TEXT,  # cat'd job logs
    "get_hpc_job_outputs": ReturnContract.INGESTIBLE_TEXT,  # downloaded output
    # rendered for the USER, not for agent reasoning
    "display_file": ReturnContract.RENDERED_FOR_USER,  # an <img> data-URI tag
    # side-effect confirmations
    "create_file": ReturnContract.EFFECT,  # "Successfully created ..."
    "submit_hpc_job": ReturnContract.EFFECT,  # a job id
    "cancel_hpc_job": ReturnContract.EFFECT,
    # a listing, not file content
    "list_hpc_jobs": ReturnContract.LISTING,
}

#: Contracts representing untrusted content the agent INGESTS -- the only
#: ones for which a tool-return-injection (G2 ``sanitize_output``) model is
#: valid.
INGESTIBLE: frozenset[ReturnContract] = frozenset({ReturnContract.INGESTIBLE_TEXT})

#: Short purpose reminders, surfaced in the finding message.
PURPOSE_HINT: dict[str, str] = {
    "display_file": (
        "display_file renders an image to the user; the agent reads an "
        "uploaded file's content with `view`."
    ),
}


__all__ = ["INGESTIBLE", "PURPOSE_HINT", "RETURN_CONTRACT", "ReturnContract"]
