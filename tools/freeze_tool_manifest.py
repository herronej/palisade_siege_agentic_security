"""Freeze the reference deployment's MCP tool contracts into the artifact.

``siege.realism.manifest`` normally AST-parses the live ``mcp_servers/`` tree,
so the corpus checks fail the moment the corpus drifts from the code. The
standalone artifact has no such tree, so we ship a snapshot taken at the
release tag and fall back to it.

Regenerate against a VISTA checkout::

    uv run python -m tools.freeze_tool_manifest /path/to/vista

The snapshot is data, not a second source of truth: when the live tree is
present it always wins, and this file is only consulted in its absence.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

from siege.realism.manifest import (
    _TOOL_SERVER_SUBPATHS,
    FROZEN_MANIFEST,
    _extract_tools,
    _is_mcp_tool_decorator,
)


def build(mcp_servers: Path) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for sub in _TOOL_SERVER_SUBPATHS:
        server_dir = mcp_servers / sub
        server_name = sub.split("/", 1)[0]
        if not server_dir.is_dir():
            print(f"  warning: {server_dir} missing", file=sys.stderr)
            continue
        for py in sorted(server_dir.rglob("*.py")):
            text = py.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(py))
            for spec in _extract_tools(py):
                entry: dict[str, object] = {
                    "required_args": sorted(spec.required_args),
                    "optional_args": sorted(spec.optional_args),
                    "server": server_name,
                    "source": "",
                    "docstring": "",
                }
                for node in ast.walk(tree):
                    if (
                        isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
                        and node.name == spec.name
                        and any(_is_mcp_tool_decorator(d) for d in node.decorator_list)
                    ):
                        entry["source"] = ast.get_source_segment(text, node) or ""
                        entry["docstring"] = ast.get_docstring(node) or ""
                out[spec.name] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    root = Path(argv[0]).resolve()
    mcp = root / "mcp_servers" if (root / "mcp_servers").is_dir() else root
    catalog = build(mcp)
    if not catalog:
        print(f"no MCP tools found under {mcp}", file=sys.stderr)
        return 1
    FROZEN_MANIFEST.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    print(f"froze {len(catalog)} tool contracts -> {FROZEN_MANIFEST}")
    for name in sorted(catalog):
        print(f"  {name} ({catalog[name]['server']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
