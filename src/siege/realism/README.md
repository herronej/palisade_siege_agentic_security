# Corpus-realism checks

These checks validate SIEGE instances against the *real* implementation rather
than against a description of it: every action must name a tool that exists,
with arguments that exist, routed to the gate that would actually mediate it.
The corpus fails the moment it drifts from the code.

## `mcp_tool_manifest.json`

Normally the checks AST-parse the live `mcp_servers/` tree of the reference
deployment. The standalone artifact has no such tree, so this file is a frozen
snapshot of the 21 tool contracts — name, required and optional arguments,
implementation source and docstring.

**It is a verbatim snapshot and is deliberately exempt from the PALISADE/SIEGE
rename.** The strings inside it are VISTA's own source text; rewriting
`VISTAGuard` to `PALISADE` there would make the snapshot disagree with the
system it is a snapshot of, which is exactly the drift these checks exist to
catch.

The live tree always wins when present. Regenerate the snapshot against a
VISTA checkout with:

```bash
uv run python -m tools.freeze_tool_manifest /path/to/vista
```
