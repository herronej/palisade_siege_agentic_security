"""
SIEGE **corpus realism** checks (Tier 0).

A deterministic linter that cross-checks every authored attack/benign
instance against a ground-truth manifest **extracted from the live VISTA
implementation** -- the real MCP tool catalog (AST-parsed from the
``mcp_servers`` source), the gate routing, the upload deny-list, the
sandbox write-roots, and the live ``PalisadeSettings`` defaults.

The point is anti-drift: the corpus fails the check the moment it names a
tool the agent doesn't have, passes args the tool doesn't accept, routes
an action to the wrong gate, models an impossible upload, or repeats a
config claim that no longer matches the code.

Run it::

    uv run python -m siege.realism

and it is enforced in CI by ``tests/test_corpus_realism.py``.
"""

from __future__ import annotations

from siege.realism.checks import Finding, check_corpus, check_instance
from siege.realism.contracts import RETURN_CONTRACT, ReturnContract
from siege.realism.manifest import ToolSpec, expected_gate, tool_catalog, tool_source

__all__ = [
    "Finding",
    "RETURN_CONTRACT",
    "ReturnContract",
    "ToolSpec",
    "check_corpus",
    "check_instance",
    "expected_gate",
    "tool_catalog",
    "tool_source",
]
