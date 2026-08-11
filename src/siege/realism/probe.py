"""
Tier-1 LIVE probe: validate the static manifest against a *running*
``vista_mcp_server``, and confirm read-only return shapes empirically.

Tier 0 AST-parses the tool catalog from source; this probe cross-checks that
catalog against what the server actually exposes over MCP
(``list_tools()``) -- catching a tool/arg the running server has that the
static parse missed (or vice versa) -- and provides a return-shape check
(``check_return_shape``) so a live ``direct_call_tool`` result can be
confirmed against its declared return contract.

**Opt-in.** It needs ``vista_mcp_server`` reachable (``./launch.sh``); the
pytest gate skips when it isn't. The pure diff / shape logic is unit-tested
without a server.
"""

from __future__ import annotations

import socket
from urllib.parse import urlparse

from siege.realism.checks import Finding
from siege.realism.contracts import RETURN_CONTRACT, ReturnContract
from siege.realism.manifest import ToolSpec, tool_catalog

#: Default endpoint the vista_mcp_server serves on (AGENTS.md: :8000/mcp).
DEFAULT_VISTA_MCP_URL = "http://localhost:8000/mcp"


def server_reachable(url: str, timeout: float = 1.5) -> bool:
    """Cheap TCP reachability check, so the probe test can skip cleanly."""
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def diff_catalog(
    live: dict[str, set[str]],
    static: dict[str, ToolSpec],
    *,
    server: str,
) -> list[Finding]:
    """Cross-check the live ``list_tools`` against the static AST manifest.

    A tool/arg the running server exposes but the manifest lacks is an
    **error** (the manifest -- and therefore the corpus checks -- is stale).
    A manifest tool the server doesn't expose is a **warn** (it may be a
    disabled sub-server, e.g. agenthpc, off by default).
    """
    findings: list[Finding] = []
    for name, live_args in live.items():
        spec = static.get(name)
        if spec is None:
            findings.append(
                Finding(
                    "(live)",
                    "live_catalog",
                    "error",
                    name,
                    f"running {server} exposes tool {name!r} the static manifest "
                    f"(AST parse) does not know -- the manifest is stale",
                )
            )
            continue
        unknown = live_args - spec.all_args
        if unknown:
            findings.append(
                Finding(
                    "(live)",
                    "live_catalog",
                    "error",
                    name,
                    f"{name}: server arg(s) {sorted(unknown)} are absent from the "
                    f"static manifest args {sorted(spec.all_args)}",
                )
            )
    for name, spec in static.items():
        if spec.server == server and name not in live:
            findings.append(
                Finding(
                    "(live)",
                    "live_catalog",
                    "warn",
                    name,
                    f"manifest has {server} tool {name!r} the running server does "
                    f"not expose (a disabled sub-server, or a stale entry)",
                )
            )
    return findings


def check_return_shape(tool: str, result_text: str) -> list[Finding]:
    """Confirm a live tool result matches its declared return contract.

    A ``rendered_for_user`` tool must return an ``<img>`` / data-URI blob; an
    ``ingestible_text`` tool must NOT (that would mean the corpus's
    text-ingress model is wrong, the display_file class of bug).
    """
    contract = RETURN_CONTRACT.get(tool)
    if contract is None:
        return []
    rendered = ("<img" in result_text) or result_text.lstrip().startswith("data:")
    if contract is ReturnContract.RENDERED_FOR_USER and not rendered:
        return [
            Finding(
                "(live)",
                "return_shape",
                "warn",
                tool,
                f"{tool} is declared rendered_for_user but its live return is not "
                f"an <img>/HTML blob",
            )
        ]
    if contract is ReturnContract.INGESTIBLE_TEXT and rendered:
        return [
            Finding(
                "(live)",
                "return_shape",
                "error",
                tool,
                f"{tool} is declared ingestible_text but its live return is a "
                f"rendered <img> blob -- not agent-read content",
            )
        ]
    return []


async def list_live_tools(url: str) -> dict[str, set[str]]:
    """``{tool_name: {arg_names}}`` from the running server over MCP."""
    from pydantic_ai.mcp import MCPServerStreamableHTTP

    server = MCPServerStreamableHTTP(url)
    out: dict[str, set[str]] = {}
    async with server:
        for td in await server.list_tools():
            schema = getattr(td, "parameters_json_schema", None) or {}
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            out[str(td.name)] = set(props)
    return out


async def probe_live(url: str | None = None) -> list[Finding]:
    """Connect to a running vista_mcp_server and diff its catalog against the
    static manifest. Raises if the server is unreachable -- gate on
    ``server_reachable`` first."""
    url = url or DEFAULT_VISTA_MCP_URL
    live = await list_live_tools(url)
    return diff_catalog(live, tool_catalog(), server="vista_mcp_server")


__all__ = [
    "DEFAULT_VISTA_MCP_URL",
    "check_return_shape",
    "diff_catalog",
    "list_live_tools",
    "probe_live",
    "server_reachable",
]
