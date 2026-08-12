"""
CI gate: every SIEGE corpus instance must be consistent with the
**live VISTA implementation** -- it may only name tools the agent actually
has, pass args those tools accept, route actions to the gate the runner
would route them to, model uploads the datasets-tab deny-list allows, and
repeat config claims that still match the code.

This is the Tier-0 realism linter; the manifest it checks against is
AST-extracted from the real mcp_servers source + imported PALISADE policy,
so this test fails the moment the corpus drifts from the implementation.
"""

from __future__ import annotations

import pytest

from siege import load_instances
from siege.corpus_builder import CORPUS_DIR
from siege.realism import check_instance, tool_catalog
from siege.realism.contracts import RETURN_CONTRACT, ReturnContract
from siege.realism.manifest import tool_source
from siege.schemas import (
    Action,
    ActionKind,
    CapabilitySpec,
    Instance,
    Session,
    SuccessCriterion,
    Turn,
)

_INSTANCES = load_instances(CORPUS_DIR)


def _synthetic(action: Action, *, description: str = "") -> Instance:
    return Instance(
        instance_id="synthetic",
        boundary="TEST",
        template="synthetic",
        kind="attack",
        description=description,
        sessions=(Session(session_id="s1", turns=(Turn(actions=(action,)),)),),
        success_criterion=SuccessCriterion(check="attack_action_allowed"),
    )


def _fired(action: Action, **kw) -> set[str]:
    return {f.check for f in check_instance(_synthetic(action, **kw))}


def test_tool_catalog_extracted_from_real_servers():
    catalog = tool_catalog()
    # The real dev-sandbox + vista tools must be discovered; the fictional
    # ones a corpus author might assume must NOT be.
    assert {"run_bash", "create_file", "view", "display_file"} <= set(catalog)
    assert not ({"read_file_content", "search_files", "list_recent_files"} & set(catalog))
    # arg contracts are real: create_file(path, content), view(path)
    assert catalog["create_file"].required_args == frozenset({"path", "content"})
    assert "path" in catalog["view"].required_args


@pytest.mark.parametrize("inst", _INSTANCES, ids=lambda i: i.instance_id)
def test_instance_is_realistic(inst):
    errors = [f for f in check_instance(inst) if f.severity == "error"]
    assert not errors, "realism errors:\n" + "\n".join(str(f) for f in errors)


# --- the checker must have teeth: each bug class is caught -------------------


def test_catches_fictional_tool():
    a = Action(
        kind=ActionKind.TOOL_CALL,
        gate="G2",
        payload={"tool_name": "read_file_content", "args": {"path": "/x"}},
    )
    assert "tool_exists" in _fired(a)


def test_catches_unknown_or_missing_args():
    # view's real arg is `path`; `uri` is unknown and `path` is missing.
    a = Action(
        kind=ActionKind.TOOL_CALL,
        gate="G2",
        payload={"tool_name": "view", "args": {"uri": "/x"}},
    )
    assert "args_valid" in _fired(a)


def test_catches_misrouted_gate():
    # run_bash must route to G4, not G2.
    a = Action(
        kind=ActionKind.TOOL_CALL,
        gate="G2",
        payload={"tool_name": "run_bash", "args": {"command": "ls"}},
    )
    assert "gate_matches" in _fired(a)


def test_catches_blocked_upload_extension():
    a = Action(
        kind=ActionKind.TOOL_CALL,
        gate="G2",
        label="read upload",
        payload={"tool_name": "view", "args": {"path": "/mnt/data/uploads/evil.py"}},
        capability=CapabilitySpec(value_id="u", source="upload:file", taint=True),
    )
    assert "upload_ext_allowed" in _fired(a)


def test_allowed_upload_extension_is_clean():
    a = Action(
        kind=ActionKind.TOOL_CALL,
        gate="G2",
        label="read upload",
        payload={"tool_name": "view", "args": {"path": "/mnt/data/uploads/data.csv"}},
        capability=CapabilitySpec(value_id="u", source="upload:file", taint=True),
    )
    assert "upload_ext_allowed" not in _fired(a)


def test_catches_stale_config_claim():
    # g5_require_submit_approval defaults to False; a description claiming True
    # must be flagged.
    a = Action(kind=ActionKind.PROMPT, gate="G1", payload={"user_prompt": "hi"})
    assert "config_claims_match" in _fired(
        a, description="assumes g5_require_submit_approval=True"
    )


# --- Tier 1: tool-purpose / return-contract ---------------------------------


def test_return_contracts_anchored_to_source():
    catalog = tool_catalog()
    assert set(RETURN_CONTRACT) <= set(catalog), "stale return-contract entries"
    # display_file really renders an <img> blob for the user -> not ingestible.
    assert RETURN_CONTRACT["display_file"] is ReturnContract.RENDERED_FOR_USER
    assert "<img" in tool_source("display_file")
    # view returns file content, not a rendered blob.
    assert RETURN_CONTRACT["view"] is ReturnContract.INGESTIBLE_TEXT
    assert "<img" not in tool_source("view")


def test_catches_display_file_as_text_ingress():
    # The original b1_9 bug: an image renderer modelled as an untrusted-text
    # read (routed to G2). Tier 0 can't see it -- the tool exists, args fine.
    a = Action(
        kind=ActionKind.TOOL_CALL,
        gate="G2",
        label="read upload via display_file",
        payload={"tool_name": "display_file", "args": {"uri": "/mnt/data/uploads/x.csv"}},
    )
    fired = _fired(a)
    assert "return_contract" in fired
    assert "tool_exists" not in fired and "args_valid" not in fired


def test_real_reader_at_g2_is_clean():
    a = Action(
        kind=ActionKind.TOOL_CALL,
        gate="G2",
        payload={"tool_name": "view", "args": {"path": "/mnt/data/uploads/x.csv"}},
    )
    assert "return_contract" not in _fired(a)
