"""
Deterministic corpus-realism checks (Tier 0).

Each check cross-references an authored instance against the live manifest
(``manifest.py``). Findings are returned, never raised, so a caller can
report them all at once; the pytest gate fails on any ``error``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from palisade.config import PalisadeSettings
from siege.schemas import Action, ActionKind, Instance
from siege.realism.contracts import INGESTIBLE, PURPOSE_HINT, RETURN_CONTRACT
from siege.realism.manifest import (
    BLOCKED_EXTENSIONS,
    REQUIRED_PAYLOAD_KEYS,
    expected_gate,
    tool_catalog,
)

#: Sandbox-side directory uploaded files land in (agents.py uploads_dir ->
#: /mnt/data/uploads). An "upload ingress" is a *read* of a path under here.
_SANDBOX_UPLOADS = "/mnt/data/uploads/"


@dataclass(frozen=True)
class Finding:
    """One realism discrepancy in an instance."""

    instance_id: str
    check: str
    severity: str  # "error" | "warn"
    where: str  # action label / "description"
    detail: str

    def __str__(self) -> str:
        return (
            f"[{self.severity}] {self.instance_id} :: {self.check} :: "
            f"{self.where} -- {self.detail}"
        )


_BOOL_CLAIM = re.compile(r"\b([a-z][a-z0-9_]+)\s*=\s*(True|False)\b")
_UPLOAD_PATH_KEYS = ("path", "uri", "file", "file_path")


def _args(action: Action) -> dict:
    args = action.payload.get("args")
    return args if isinstance(args, dict) else {}


def _check_tool_call(inst: Instance, action: Action, where: str) -> list[Finding]:
    """tool_exists + args_valid + gate, for actions naming a real tool."""
    findings: list[Finding] = []
    tool = str(action.payload.get("tool_name", ""))
    catalog = tool_catalog()

    spec = catalog.get(tool)
    if spec is None:
        findings.append(
            Finding(
                inst.instance_id,
                "tool_exists",
                "error",
                where,
                f"tool_name {tool!r} is not a tool the VISTA agent has "
                f"(available: {', '.join(sorted(catalog))})",
            )
        )
        return findings  # can't check args against an unknown tool

    args = _args(action)
    missing = spec.required_args - args.keys()
    if missing:
        findings.append(
            Finding(
                inst.instance_id,
                "args_valid",
                "error",
                where,
                f"{tool} missing required arg(s) {sorted(missing)} "
                f"(required: {sorted(spec.required_args)})",
            )
        )
    unknown = args.keys() - spec.all_args
    if unknown:
        findings.append(
            Finding(
                inst.instance_id,
                "args_valid",
                "error",
                where,
                f"{tool} got unknown arg(s) {sorted(unknown)} "
                f"(accepts: {sorted(spec.all_args)})",
            )
        )
    return findings


def _check_gate(inst: Instance, action: Action, where: str) -> list[Finding]:
    want = expected_gate(action)
    if action.gate is not None and want is not None and action.gate != want:
        return [
            Finding(
                inst.instance_id,
                "gate_matches",
                "error",
                where,
                f"action routes to {action.gate} but the implementation would "
                f"route this {action.kind.value} to {want}",
            )
        ]
    return []


def _check_upload_ext(inst: Instance, action: Action, where: str) -> list[Finding]:
    """An upload-ingress read must name a file type the deny-list allows."""
    cap = action.capability
    if cap is None or not str(cap.source).startswith("upload"):
        return []
    args = _args(action)
    raw = next((args[k] for k in _UPLOAD_PATH_KEYS if isinstance(args.get(k), str)), None)
    if raw is None:
        return []
    # Only the *ingress read* of a file under the uploads dir is an upload;
    # a downstream sink (e.g. create_file fetch_table.py) merely inherits the
    # upload taint and is not itself an upload.
    norm = raw[len("file://") :] if raw.startswith("file://") else raw
    if not norm.startswith(_SANDBOX_UPLOADS):
        return []
    ext = PurePosixPath(norm).suffix.lower()
    if ext in BLOCKED_EXTENSIONS:
        return [
            Finding(
                inst.instance_id,
                "upload_ext_allowed",
                "error",
                where,
                f"upload ingress {raw!r} has extension {ext!r}, which the "
                f"datasets-tab deny-list (ingestion.upload_violation) rejects "
                f"-- this attack can't be delivered as modelled",
            )
        ]
    return []


def _check_return_contract(inst: Instance, action: Action, where: str) -> list[Finding]:
    """Tier 1: a G2-routed tool *return* must be ingestible agent-read text.

    An action routed to G2 models the tool's return as untrusted content the
    agent reads (the ``sanitize_output`` surface). A render-for-the-user tool
    (``display_file``), a side-effect confirmation, or a listing there is a
    purpose mismatch -- the structural checks can't see it because the tool
    exists and its args are fine.
    """
    if action.kind is not ActionKind.TOOL_CALL:
        return []
    tool = str(action.payload.get("tool_name", ""))
    if not tool or (action.gate or expected_gate(action)) != "G2":
        return []
    contract = RETURN_CONTRACT.get(tool)
    if contract is None:
        return [
            Finding(
                inst.instance_id,
                "return_contract",
                "warn",
                where,
                f"{tool} is routed to G2 (its return modelled as untrusted "
                f"content) but has no declared return contract -- classify it "
                f"in realism/contracts.py",
            )
        ]
    if contract not in INGESTIBLE:
        hint = PURPOSE_HINT.get(tool, "")
        return [
            Finding(
                inst.instance_id,
                "return_contract",
                "error",
                where,
                f"{tool} is routed to G2 -- modelling its return as untrusted "
                f"text the agent ingests -- but {tool}'s return is "
                f"{contract.value}, not agent-read content. {hint}".rstrip(),
            )
        ]
    return []


def _check_payload_shape(inst: Instance, action: Action, where: str) -> list[Finding]:
    """An abstract (no-tool_name) gated action must carry the keys its gate
    consumes."""
    if action.kind is ActionKind.TOOL_CALL and "tool_name" in action.payload:
        return []  # concrete tool call -- covered by _check_tool_call
    keys = REQUIRED_PAYLOAD_KEYS.get(action.gate or expected_gate(action) or "")
    if not keys:
        return []
    if not any(k in action.payload for k in keys):
        return [
            Finding(
                inst.instance_id,
                "payload_shape",
                "warn",
                where,
                f"{action.gate} action payload has none of the keys its gate "
                f"reads {list(keys)}; keys present: {sorted(action.payload)}",
            )
        ]
    return []


def _check_config_claims(inst: Instance, settings: PalisadeSettings) -> list[Finding]:
    """A ``attr=True/False`` claim in the description must match the live default."""
    findings: list[Finding] = []
    for m in _BOOL_CLAIM.finditer(inst.description or ""):
        attr, claimed = m.group(1), (m.group(2) == "True")
        actual = getattr(settings, attr, None)
        if isinstance(actual, bool) and actual != claimed:
            findings.append(
                Finding(
                    inst.instance_id,
                    "config_claims_match",
                    "error",
                    "description",
                    f"claims {attr}={claimed} but the live default is {actual}",
                )
            )
    return findings


def check_instance(
    inst: Instance, *, settings: PalisadeSettings | None = None
) -> list[Finding]:
    """Run every Tier-0 check over one instance."""
    settings = settings or PalisadeSettings()
    findings: list[Finding] = []
    for session in inst.sessions:
        for turn in session.turns:
            for action in turn.actions:
                where = action.label or action.kind.value
                if action.kind is ActionKind.TOOL_CALL and "tool_name" in action.payload:
                    findings += _check_tool_call(inst, action, where)
                findings += _check_gate(inst, action, where)
                findings += _check_upload_ext(inst, action, where)
                findings += _check_return_contract(inst, action, where)
                findings += _check_payload_shape(inst, action, where)
    findings += _check_config_claims(inst, settings)
    return findings


def check_corpus(instances) -> list[Finding]:
    """Run the checks over an iterable of instances; flat list of findings."""
    settings = PalisadeSettings()
    out: list[Finding] = []
    for inst in instances:
        out += check_instance(inst, settings=settings)
    return out


__all__ = ["Finding", "check_corpus", "check_instance"]
