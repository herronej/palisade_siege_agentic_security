"""
Unit tests for the G2 high-stakes-tool annotation reader.

Scope is the four acceptance criteria of the work item
`Read MCP ToolAnnotations to build the high-stakes tool list`:

1. `discover_high_stakes_tools(mcp_url) -> set[str]` returns tool
   names where any annotation in `{destructiveHint, openWorldHint}`
   is True.
2. Discovery happens at sidecar construction; cached for the
   session.
3. The right set is computed against a mocked MCP server.
4. Discovery failure does not crash the sidecar -- the high-stakes
   set falls back to the hardcoded conservative set.

The tests use a hand-rolled fake MCP server class rather than
`unittest.mock` so the contract with PydanticAI's
`MCPServerStreamableHTTP` is visible: we depend on `url=` /
`timeout=` constructor kwargs and on a single `async def
list_tools()` method that returns `list[mcp.types.Tool]`. If
PydanticAI ever changes that surface, this fake will need updating
in the same PR, which is the right signal.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from mcp.types import Tool, ToolAnnotations

# anyio's pytest plugin requires async tests to be marked. Apply at
# module level so individual tests don't have to repeat the marker;
# sync tests inherit the marker harmlessly.
pytestmark = pytest.mark.anyio

from palisade.host import HostProjectModel
from palisade.config import PalisadeSettings
from palisade.gates import g2_tool
from palisade.gates.g2_tool import (
    HIGH_STAKES_FALLBACK,
    _is_high_stakes,
    discover_high_stakes_tools,
)
from palisade.sidecar import PalisadeSidecar


# -----------------------------------------------------------------
# Test doubles
# -----------------------------------------------------------------


class _FakeMcpServer:
    """
    Stand-in for `pydantic_ai.mcp.MCPServerStreamableHTTP`.

    Records its construction kwargs (so tests can assert on the URL
    and timeout that `discover_high_stakes_tools` passed in) and
    exposes a single `list_tools()` coroutine that returns the
    configured tool list.

    `_FakeMcpServerFactory` below wraps an instance of this class
    in a callable that mimics `MCPServerStreamableHTTP(url=..., ...)`
    construction, which is what `monkeypatch.setattr` substitutes
    into the module under test.
    """

    def __init__(self, *, url: str, timeout: float) -> None:
        self.url = url
        self.timeout = timeout
        # Set by the factory before list_tools is called.
        self._tools: list[Tool] = []
        self.list_tools_calls = 0

    async def list_tools(self) -> list[Tool]:
        self.list_tools_calls += 1
        return list(self._tools)


class _FakeMcpServerFactory:
    """
    Callable that mimics `MCPServerStreamableHTTP(...)` and records
    each constructed fake server so tests can introspect them.

    Stored constructor kwargs let tests verify
    `discover_high_stakes_tools` forwarded `mcp_url` and `timeout`
    correctly.
    """

    def __init__(self, tools: list[Tool] | Exception) -> None:
        self._tools_or_exc = tools
        self.constructed: list[_FakeMcpServer] = []

    def __call__(self, *, url: str, timeout: float) -> _FakeMcpServer:
        server = _FakeMcpServer(url=url, timeout=timeout)
        # The Exception branch is how we simulate "list_tools
        # raises" without coupling the fake to the specific
        # exception class.
        if isinstance(self._tools_or_exc, Exception):
            exc = self._tools_or_exc

            async def _raises() -> list[Tool]:  # noqa: D401
                raise exc

            server.list_tools = _raises  # type: ignore[assignment]
        else:
            server._tools = self._tools_or_exc
        self.constructed.append(server)
        return server


def _tool(
    name: str,
    *,
    destructive: bool | None = None,
    open_world: bool | None = None,
    read_only: bool | None = None,
    annotated: bool = True,
) -> Tool:
    """
    Build a `mcp.types.Tool` with the requested annotations.

    `annotated=False` skips the annotations field entirely (models
    `@mcp.tool()` decorators with no annotation argument). The
    individual hint kwargs default to `None` so callers can express
    "annotation present but hint unset" by simply not passing the
    kwarg.
    """
    if not annotated:
        return Tool(name=name, inputSchema={})
    return Tool(
        name=name,
        inputSchema={},
        annotations=ToolAnnotations(
            destructiveHint=destructive,
            openWorldHint=open_world,
            readOnlyHint=read_only,
        ),
    )


def _make_project() -> HostProject:
    """Minimal valid `HostProject` for sidecar tests."""
    return HostProjectModel(
        id=uuid.uuid4(),
        name="test-project",
        description=None,
        system_prompt=None,
        skills=[],
        knowledge_bases=[],
        tools=[],
        usage_limits={},
    )


# -----------------------------------------------------------------
# _is_high_stakes predicate
# -----------------------------------------------------------------


def test_is_high_stakes_true_when_destructive():
    assert _is_high_stakes(_tool("run_bash", destructive=True)) is True


def test_is_high_stakes_true_when_open_world():
    assert _is_high_stakes(_tool("submit_hpc_job", open_world=True)) is True


def test_is_high_stakes_true_when_both_hints():
    assert _is_high_stakes(
        _tool("dangerous", destructive=True, open_world=True)
    ) is True


def test_is_high_stakes_false_when_only_read_only():
    # The acceptance criterion explicitly does NOT consult
    # readOnlyHint; a read-only tool with the other two hints
    # unset is not high-stakes.
    assert _is_high_stakes(_tool("display_file", read_only=True)) is False


def test_is_high_stakes_false_when_destructive_hint_false():
    # destructiveHint=False is the explicit "not destructive" case
    # from VISTA's annotated tools (see rag_mcp.py). Must not be
    # treated as high-stakes.
    assert _is_high_stakes(
        _tool("rag_search", destructive=False, open_world=False, read_only=True)
    ) is False


def test_is_high_stakes_false_when_no_annotations():
    # A tool with no annotations field entirely defaults to
    # non-high-stakes. This models third-party MCP tools the
    # author didn't annotate.
    assert _is_high_stakes(_tool("plain", annotated=False)) is False


def test_is_high_stakes_false_when_hints_all_unset():
    # `annotations=ToolAnnotations()` with every hint None is the
    # "annotation present but no claims" case. Treated the same as
    # no-annotation.
    assert _is_high_stakes(_tool("noclaim")) is False


# -----------------------------------------------------------------
# discover_high_stakes_tools -- happy path
# -----------------------------------------------------------------


async def test_discover_returns_destructive_and_open_world_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    AC1: returns tool names where any annotation in
    `{destructiveHint, openWorldHint}` is True.

    Mirrors VISTA's actual MCP server layout: four high-stakes
    tools (two destructive, two open-world-only -- the latter
    modeled on `get_available_jobs` / `list_hpc_jobs`), several
    benign tools, one unannotated tool.
    """
    tools = [
        _tool("run_bash", destructive=True, read_only=False),
        _tool("create_file", destructive=True, read_only=False),
        _tool("submit_hpc_job", destructive=True, open_world=True),
        # open-world-only is sufficient; this exercises the OR
        # branch and prevents a regression where the implementation
        # would AND the two hints by accident.
        _tool("list_hpc_jobs", destructive=False, open_world=True, read_only=True),
        # Benign tools: explicitly NOT in the result.
        _tool("rag_search", destructive=False, open_world=False, read_only=True),
        _tool("display_file", destructive=False, open_world=False, read_only=True),
        _tool("plain_tool", annotated=False),
    ]
    factory = _FakeMcpServerFactory(tools)
    monkeypatch.setattr(g2_tool, "MCPServerStreamableHTTP", factory)

    result = await discover_high_stakes_tools("http://mock/mcp")

    assert isinstance(result, frozenset)
    assert result == {
        "run_bash",
        "create_file",
        "submit_hpc_job",
        "list_hpc_jobs",
    }


async def test_discover_returns_empty_when_server_has_no_high_stakes_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A server that exposes only read-only / unannotated tools
    yields an empty set -- *not* the fallback. The fallback is
    reserved for the discovery-error path.
    """
    tools = [
        _tool("rag_search", destructive=False, open_world=False, read_only=True),
        _tool("display_file", read_only=True),
    ]
    monkeypatch.setattr(
        g2_tool, "MCPServerStreamableHTTP", _FakeMcpServerFactory(tools)
    )

    result = await discover_high_stakes_tools("http://mock/mcp")

    assert result == frozenset()


async def test_discover_forwards_url_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The MCP-server construction picks up the URL and timeout we
    pass in. Documents the contract with PydanticAI's
    `MCPServerStreamableHTTP` signature -- if those kwargs ever
    change names, this test catches it before the runtime would.
    """
    factory = _FakeMcpServerFactory([_tool("x", destructive=True)])
    monkeypatch.setattr(g2_tool, "MCPServerStreamableHTTP", factory)

    await discover_high_stakes_tools("http://specific-host:8000/mcp", timeout=3.5)

    assert len(factory.constructed) == 1
    server = factory.constructed[0]
    assert server.url == "http://specific-host:8000/mcp"
    assert server.timeout == 3.5


# -----------------------------------------------------------------
# discover_high_stakes_tools -- failure modes
# -----------------------------------------------------------------


async def test_discover_falls_back_when_list_tools_raises(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    AC4: MCP server down -> hardcoded fallback, no exception.
    """
    factory = _FakeMcpServerFactory(ConnectionRefusedError("nope"))
    monkeypatch.setattr(g2_tool, "MCPServerStreamableHTTP", factory)

    with caplog.at_level("WARNING", logger=g2_tool.logger.name):
        result = await discover_high_stakes_tools("http://down/mcp")

    assert result == HIGH_STAKES_FALLBACK
    # A noisy log line is the operator's signal that discovery
    # fell back. Assert on a substring rather than the exact
    # message so we can refine wording without breaking tests.
    assert any(
        "fallback" in record.message.lower() for record in caplog.records
    ), f"expected warning about fallback in {[r.message for r in caplog.records]}"


async def test_discover_falls_back_when_server_construction_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Even a synchronous failure constructing the MCP server (e.g.,
    a malformed URL caught early by PydanticAI's validators)
    collapses to the fallback. Models the "MCP server down /
    misconfigured" deployment scenario more aggressively than the
    happy-path failure mode.
    """

    def _boom(*, url: str, timeout: float) -> Any:
        raise RuntimeError("validation error")

    monkeypatch.setattr(g2_tool, "MCPServerStreamableHTTP", _boom)

    result = await discover_high_stakes_tools("not-a-url")

    assert result == HIGH_STAKES_FALLBACK


def test_fallback_matches_currently_annotated_destructive_tools():
    """
    Pins the fallback contents to the four destructive tools
    listed in the work item's acceptance criterion. If a future PR
    edits the fallback without also updating the acceptance
    criterion, this test fires.
    """
    assert HIGH_STAKES_FALLBACK == frozenset(
        {"run_bash", "create_file", "submit_hpc_job", "cancel_hpc_job"}
    )


# -----------------------------------------------------------------
# Sidecar integration -- "discovery happens at sidecar
# construction; cached for the session"
# -----------------------------------------------------------------


def test_sidecar_starts_with_fallback_high_stakes_tools():
    """
    AC2 part 1: the sidecar has a usable high-stakes set the
    instant `__init__` returns -- without anyone having awaited
    discovery yet. This is the strictest reading of "discovery
    happens at sidecar construction": the *result* (or its
    fallback) is in place at construction time.
    """
    settings = PalisadeSettings(enabled=True, g2_enabled=True)
    sidecar = PalisadeSidecar(settings, _make_project())

    assert sidecar.high_stakes_tools == HIGH_STAKES_FALLBACK
    assert sidecar.high_stakes_tools_populated is False


async def test_sidecar_populate_refreshes_high_stakes_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    AC2 part 2 + AC3: a live populate against a mocked MCP server
    refreshes the cache to the discovered set, and the cache
    persists for subsequent reads (i.e., is "cached for the
    session").
    """
    discovered_tools = [
        _tool("run_bash", destructive=True),
        _tool("submit_hpc_job", destructive=True, open_world=True),
        # A new destructive tool the fallback wouldn't know about.
        # Models the "MCP server adds a destructive tool, PALISADE
        # picks it up automatically" scenario from the module
        # docstring.
        _tool("drop_database", destructive=True),
        _tool("rag_search", destructive=False, read_only=True),
    ]
    monkeypatch.setattr(
        g2_tool,
        "MCPServerStreamableHTTP",
        _FakeMcpServerFactory(discovered_tools),
    )

    settings = PalisadeSettings(enabled=True, g2_enabled=True)
    sidecar = PalisadeSidecar(settings, _make_project())

    returned = await sidecar.populate_high_stakes_tools("http://mock/mcp")

    expected = frozenset({"run_bash", "submit_hpc_job", "drop_database"})
    assert returned == expected
    assert sidecar.high_stakes_tools == expected
    assert sidecar.high_stakes_tools_populated is True

    # Cached-for-the-session contract: re-reading the property
    # returns the same set without re-querying the MCP server.
    # We assert by reading twice and checking equality / identity
    # of frozensets (frozensets compare by value but the same
    # underlying instance is the cheap proof).
    assert sidecar.high_stakes_tools is sidecar.high_stakes_tools


async def test_sidecar_populate_leaves_fallback_on_discovery_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    AC4 at the sidecar level: a discovery failure during populate
    leaves the fallback in place and the sidecar usable.
    `high_stakes_tools_populated` stays False so ops can see we
    fell back.
    """
    monkeypatch.setattr(
        g2_tool,
        "MCPServerStreamableHTTP",
        _FakeMcpServerFactory(TimeoutError("MCP unreachable")),
    )

    settings = PalisadeSettings(enabled=True, g2_enabled=True)
    sidecar = PalisadeSidecar(settings, _make_project())

    # Crucially: this does not raise. The acceptance criterion is
    # "discovery failure does not crash sidecar."
    returned = await sidecar.populate_high_stakes_tools("http://down/mcp")

    assert returned == HIGH_STAKES_FALLBACK
    assert sidecar.high_stakes_tools == HIGH_STAKES_FALLBACK
    assert sidecar.high_stakes_tools_populated is False


async def test_sidecar_populate_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A second populate call against the same MCP server simply
    refreshes the cache. Documents the "successive calls
    overwrite" contract from the sidecar method's docstring; the
    wiring caller is expected to call this once per
    session but a defensive double-call should be safe.
    """
    factory = _FakeMcpServerFactory([
        _tool("run_bash", destructive=True),
        _tool("custom_destructive", destructive=True),
    ])
    monkeypatch.setattr(g2_tool, "MCPServerStreamableHTTP", factory)

    settings = PalisadeSettings(enabled=True, g2_enabled=True)
    sidecar = PalisadeSidecar(settings, _make_project())

    first = await sidecar.populate_high_stakes_tools("http://mock/mcp")
    second = await sidecar.populate_high_stakes_tools("http://mock/mcp")

    assert first == second
    assert len(factory.constructed) == 2  # one MCP server constructed per call


# -----------------------------------------------------------------
# Anyio backend selection (matches the pattern in test_sidecar.py)
# -----------------------------------------------------------------


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
