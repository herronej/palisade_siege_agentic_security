"""
Ground-truth manifest of the live VISTA implementation.

Everything here is **derived from the implementation at call time** -- the
real MCP tool signatures (AST-parsed from the ``mcp_servers`` source, with
no server startup or import side effects), the gate routing, the upload
deny-list, the sandbox write-roots, and the live ``PalisadeSettings``
defaults -- so the corpus checks fail the moment the corpus drifts from the
code.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

# Source-of-truth re-exports (imported, never copied).
from palisade.gates.g4_code import DEFAULT_WRITE_ROOTS  # noqa: F401  (re-exported)
from palisade.ingestion import BLOCKED_EXTENSIONS  # noqa: F401  (re-exported)
from siege.schemas import Action, ActionKind

# The two MCP servers the ProjectAgent actually mounts (agents.py:
# get_vista_mcp_server + get_dev_mcp_server). Tool sources live under these
# package roots; we AST-parse them rather than import (vista's config
# instantiates AppSettings at import, which needs env, and importing would
# also start nothing useful for a static check).
_TOOL_SERVER_SUBPATHS = (
    "dev_mcp_server/src/dev_mcp_server",
    "vista_mcp_server/src/vista_mcp_server",
)

#: FastMCP injects these; they are not user-facing tool arguments.
_NON_ARG_PARAMS = frozenset({"ctx", "context", "self"})


@dataclass(frozen=True)
class ToolSpec:
    """A real MCP tool's user-facing argument contract."""

    name: str
    required_args: frozenset[str]
    optional_args: frozenset[str]
    server: str = ""  # "dev_mcp_server" | "vista_mcp_server"

    @property
    def all_args(self) -> frozenset[str]:
        return self.required_args | self.optional_args


#: Snapshot of the reference deployment's tool contracts, shipped so the
#: realism checks run from the standalone artifact. Regenerate against a
#: VISTA checkout with ``python -m tools.freeze_tool_manifest``.
FROZEN_MANIFEST = Path(__file__).resolve().parent / "mcp_tool_manifest.json"


def _find_mcp_servers_dir() -> Path | None:
    """Locate the live ``mcp_servers/`` tree by walking up from this file.

    Mirrors how the backend finds it (``settings.mcp_servers_path`` defaults
    to ``../mcp_servers``) but without importing backend config. Returns
    ``None`` from the standalone artifact, where there is no such tree and
    the frozen manifest is used instead.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "mcp_servers"
        if (candidate / "dev_mcp_server").is_dir():
            return candidate
    return None


@lru_cache(maxsize=1)
def _frozen() -> dict[str, dict]:
    """The shipped snapshot, or a hard error naming how to regenerate it."""
    if not FROZEN_MANIFEST.is_file():
        raise FileNotFoundError(
            f"no live mcp_servers/ tree above {Path(__file__).resolve()} and no "
            f"frozen manifest at {FROZEN_MANIFEST}. Regenerate it with "
            f"`python -m tools.freeze_tool_manifest /path/to/vista`."
        )
    return json.loads(FROZEN_MANIFEST.read_text())


def _is_mcp_tool_decorator(dec: ast.expr) -> bool:
    """True for ``@mcp.tool`` or ``@mcp.tool(...)``."""
    node = dec.func if isinstance(dec, ast.Call) else dec
    return isinstance(node, ast.Attribute) and node.attr == "tool"


def _tool_from_func(node: ast.AsyncFunctionDef | ast.FunctionDef) -> ToolSpec:
    a = node.args
    positional = list(a.posonlyargs) + list(a.args)
    n_defaults = len(a.defaults)  # defaults align to the tail of `positional`
    required: set[str] = set()
    optional: set[str] = set()
    for i, arg in enumerate(positional):
        if arg.arg in _NON_ARG_PARAMS:
            continue
        has_default = i >= len(positional) - n_defaults
        (optional if has_default else required).add(arg.arg)
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        if arg.arg in _NON_ARG_PARAMS:
            continue
        (optional if default is not None else required).add(arg.arg)
    return ToolSpec(node.name, frozenset(required), frozenset(optional))


def _extract_tools(path: Path) -> list[ToolSpec]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[ToolSpec] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and any(
            _is_mcp_tool_decorator(d) for d in node.decorator_list
        ):
            out.append(_tool_from_func(node))
    return out


@lru_cache(maxsize=None)
def tool_source(tool_name: str) -> str:
    """The source text of a tool's implementation function.

    Used by the Tier-1 return-contract anchor to verify a tool's declared
    return shape against what it actually builds (e.g. a ``<img>`` blob).
    """
    base = _find_mcp_servers_dir()
    if base is None:
        entry = _frozen().get(tool_name)
        if entry is None:
            raise KeyError(f"no @mcp.tool named {tool_name!r} in the frozen manifest")
        return entry["source"]
    for sub in _TOOL_SERVER_SUBPATHS:
        server_dir = base / sub
        if not server_dir.is_dir():
            continue
        for py in sorted(server_dir.rglob("*.py")):
            text = py.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(py))
            for node in ast.walk(tree):
                if (
                    isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
                    and node.name == tool_name
                    and any(_is_mcp_tool_decorator(d) for d in node.decorator_list)
                ):
                    seg = ast.get_source_segment(text, node)
                    if seg:
                        return seg
    raise KeyError(f"no @mcp.tool named {tool_name!r} in the mounted servers")


@lru_cache(maxsize=1)
def tool_catalog() -> dict[str, ToolSpec]:
    """The agent's real tool set, ``{tool_name: ToolSpec}``.

    AST-parsed from the mounted MCP servers' source. Deterministic and
    side-effect-free.
    """
    base = _find_mcp_servers_dir()
    if base is None:
        return {
            name: ToolSpec(
                name,
                frozenset(e["required_args"]),
                frozenset(e["optional_args"]),
                e["server"],
            )
            for name, e in _frozen().items()
        }
    catalog: dict[str, ToolSpec] = {}
    for sub in _TOOL_SERVER_SUBPATHS:
        server_dir = base / sub
        server_name = sub.split("/", 1)[0]
        if not server_dir.is_dir():
            continue
        for py in sorted(server_dir.rglob("*.py")):
            for spec in _extract_tools(py):
                catalog[spec.name] = replace(spec, server=server_name)
    if not catalog:  # pragma: no cover - guards a broken layout
        raise RuntimeError(f"no MCP tools found under {base}")
    return catalog


@lru_cache(maxsize=None)
def tool_docstring(tool_name: str) -> str:
    """The docstring of a tool's implementation (ground truth for the Tier-2
    judge's brief)."""
    base = _find_mcp_servers_dir()
    if base is None:
        entry = _frozen().get(tool_name)
        if entry is None:
            raise KeyError(f"no @mcp.tool named {tool_name!r} in the frozen manifest")
        return entry["docstring"]
    for sub in _TOOL_SERVER_SUBPATHS:
        server_dir = base / sub
        if not server_dir.is_dir():
            continue
        for py in sorted(server_dir.rglob("*.py")):
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            for node in ast.walk(tree):
                if (
                    isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
                    and node.name == tool_name
                    and any(_is_mcp_tool_decorator(d) for d in node.decorator_list)
                ):
                    return (ast.get_docstring(node) or "").strip()
    raise KeyError(f"no @mcp.tool named {tool_name!r} in the mounted servers")


#: Short, curated summary of what each gate defends -- ground truth for the
#: Tier-2 judge (derived from the gate module docstrings).
GATE_GLOSSARY: dict[str, str] = {
    "G1": (
        "prompt gate -- screens the user PROMPT (jailbreak / dual-use / PII). "
        "Its attached-file MIME policy is inert in this product: uploads arrive "
        "via the Datasets UI and bypass G1 entirely."
    ),
    "G2": (
        "tool-boundary gate -- sanitize_output runs on every string TOOL RETURN "
        "(after_tool_execute), neutralising injections in untrusted tool output. "
        "This is the live defender of a tool-return-injection."
    ),
    "G3": (
        "RAG / retrieval gate -- per-chunk sanitization of retrieved corpus and "
        "cross-session memory content."
    ),
    "G4": (
        "sandbox / code gate -- Tier-0 path & exec confinement (always on), an "
        "optional Semgrep tier, and the Q-LLM code-intent tier. Defends run_bash "
        "and create_file."
    ),
    "G5": "HPC-job gate -- screens submit_hpc_job (Slurm script) submissions.",
    "G6": (
        "egress / correctness contracts -- bounds outbound data and checks "
        "scientific-claim correctness (data-value poisoning, citation forgery)."
    ),
}


# -----------------------------------------------------------------
# Gate routing oracle
# -----------------------------------------------------------------
#
# Mirrors the runner's real routing: ``schemas.DEFAULT_GATE_FOR_KIND`` for
# non-tool actions, and ``Action.resolved_gate``'s tool-name routing for
# tool calls (submit_hpc_job -> G5, run_bash/create_file -> G4, else G2),
# extended with rag_search -> G3 (the RAG boundary). Used to flag an action
# whose explicit ``gate`` disagrees with where the implementation routes it.

_TOOL_TO_GATE: dict[str, str] = {
    "submit_hpc_job": "G5",
    "run_bash": "G4",
    "create_file": "G4",
    "rag_search": "G3",
}


def expected_gate(action: Action) -> str | None:
    """The gate the implementation would route ``action`` to, or None when
    the action kind is not gate-checked (MEMORY_WRITE / RESPONSE)."""
    kind = action.kind
    if kind is ActionKind.PROMPT:
        return "G1"
    if kind in (ActionKind.MEMORY_READ, ActionKind.RAG_RETRIEVE):
        return "G3"
    if kind is ActionKind.TOOL_CALL:
        tool = str(action.payload.get("tool_name", ""))
        if tool in _TOOL_TO_GATE:
            return _TOOL_TO_GATE[tool]
        # The abstract HPC submission is modelled by a slurm_script payload
        # with no tool_name.
        if "slurm_script" in action.payload:
            return "G5"
        # Any other tool *return* is defended at the G2 sanitize_output
        # surface (the resolved_gate "everything else -> G2" default).
        return "G2"
    return None


# -----------------------------------------------------------------
# Payload-shape oracle (abstract, non-tool-name actions)
# -----------------------------------------------------------------
#
# What each gate's check actually reads from an action payload, so a
# malformed abstract action (a G1 prompt with no user_prompt, a G5
# submission with no slurm_script) is caught. Lenient: at least one of the
# listed keys must be present.

REQUIRED_PAYLOAD_KEYS: dict[str, tuple[str, ...]] = {
    "G1": ("user_prompt",),
    "G5": ("slurm_script",),
    "G3": ("query", "kb_slug", "key", "gate_payload"),
}


__all__ = [
    "BLOCKED_EXTENSIONS",
    "DEFAULT_WRITE_ROOTS",
    "GATE_GLOSSARY",
    "REQUIRED_PAYLOAD_KEYS",
    "ToolSpec",
    "expected_gate",
    "tool_catalog",
    "tool_docstring",
    "tool_source",
]
