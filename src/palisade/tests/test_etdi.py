"""
Unit tests for the ETDI-style tool-descriptor registry and its G2
integration (work item `Add ETDI-style tool-descriptor
hashing (Bhatt et al. arXiv 2506.01333)`).

Covers all five acceptance criteria:

1. `ToolDescriptorRegistry` stores hash for each tool at startup.
2. On each tool call, current hash compared to startup hash;
   mismatch denies with `incident_level=2` and a message
   identifying the mismatched tool.
3. Optional `palisade_tool_manifest.json` at `contracts_dir`
   pins known-good hashes; runtime hashes validated against the
   manifest when present.
4. Mock tool's descriptor change between two calls is detected.
5. Missing manifest file is non-fatal.

The tests are split into three groups:

- **Hash + canonical descriptor**: lockdown on the hash input
  shape and JSON-serialization invariants. If the canonical form
  drifts, the manifest format breaks; pinning here protects
  against silent breakage.
- **Registry**: pin/verify/manifest semantics.
- **Manifest I/O + G2 integration**: filesystem failure modes and
  the end-to-end gate-denial path.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

# anyio's pytest plugin requires async tests to be marked. A
# module-level pytestmark applies to every async test below.
pytestmark = pytest.mark.anyio

from mcp.types import Tool, ToolAnnotations

from palisade.host import HostProject
from palisade.capabilities import CapabilityRegistry
from palisade.config import PalisadeSettings
from palisade.gates.base import GateContext
from palisade.gates.g2_tool import G2ToolGate
from palisade.gates import g2_tool as g2_tool_module
from palisade.sidecar import PalisadeSidecar
from palisade.tool_registry import (
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    HashVerification,
    ToolDescriptorRegistry,
    canonical_descriptor,
    hash_descriptor,
    hash_tool_descriptor,
    load_manifest,
    write_manifest,
)
from palisade.trust import TrustScorer


# -----------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------


def _make_project(
    *,
    tools: list[str] | None = None,
    knowledge_bases: list[str] | None = None,
) -> HostProject:
    return HostProject(
        id=uuid.uuid4(),
        name="test-project",
        description=None,
        system_prompt=None,
        skills=[],
        knowledge_bases=knowledge_bases or [],
        tools=tools or [],
        usage_limits={},
    )


def _ctx() -> GateContext:
    return GateContext(
        capability_registry=CapabilityRegistry(),
        trust_scorer=TrustScorer(PalisadeSettings(enabled=True)),
    )


def _tool(
    name: str,
    *,
    description: str | None = "",
    input_schema: dict[str, Any] | None = None,
    destructive: bool | None = None,
    open_world: bool | None = None,
) -> Tool:
    """Build a real `mcp.types.Tool` -- matches the discovery path."""
    return Tool(
        name=name,
        description=description,
        inputSchema=input_schema or {},
        annotations=ToolAnnotations(
            destructiveHint=destructive,
            openWorldHint=open_world,
        )
        if (destructive is not None or open_world is not None)
        else None,
    )


# -----------------------------------------------------------------
# canonical_descriptor + hash_descriptor invariants
# -----------------------------------------------------------------


def test_canonical_descriptor_coerces_none_description_to_empty() -> None:
    """
    `description=None` and `description=""` hash identically.
    They're indistinguishable at the MCP protocol level and
    different hashes would force the operator to re-pin
    spuriously.
    """
    a = canonical_descriptor("t", None, {})
    b = canonical_descriptor("t", "", {})
    assert a == b


def test_canonical_descriptor_coerces_none_schema_to_empty_dict() -> None:
    a = canonical_descriptor("t", "x", None)
    assert a["inputSchema"] == {}


def test_hash_descriptor_is_stable_across_key_order() -> None:
    """
    Key order in the canonical form must not change the hash.
    `json.dumps(sort_keys=True)` makes this true; we pin the
    property so a future refactor that drops the flag fails
    here.
    """
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    reordered_schema = {"properties": {"x": {"type": "string"}}, "type": "object"}
    h1 = hash_descriptor(canonical_descriptor("t", "d", schema))
    h2 = hash_descriptor(canonical_descriptor("t", "d", reordered_schema))
    assert h1 == h2


def test_hash_descriptor_changes_on_any_field_change() -> None:
    """
    Each of name / description / inputSchema is hash-input; a
    change in any of the three produces a new hash. This is the
    rug-pull detection guarantee.
    """
    base = canonical_descriptor("t", "d", {"type": "object"})
    h_base = hash_descriptor(base)
    h_name = hash_descriptor(canonical_descriptor("t2", "d", {"type": "object"}))
    h_desc = hash_descriptor(canonical_descriptor("t", "d2", {"type": "object"}))
    h_schema = hash_descriptor(canonical_descriptor("t", "d", {"type": "string"}))
    assert h_base != h_name
    assert h_base != h_desc
    assert h_base != h_schema


def test_hash_descriptor_is_sha256_hex() -> None:
    """
    Hash is a 64-character lowercase hex string (SHA-256). Pins
    the format expected by the manifest JSON loader.
    """
    h = hash_tool_descriptor("t", "d", {"type": "object"})
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# -----------------------------------------------------------------
# ToolDescriptorRegistry -- AC 1
# -----------------------------------------------------------------


def test_registry_stores_startup_hash_on_first_pin() -> None:
    """AC1: registry stores hash for each tool at startup."""
    reg = ToolDescriptorRegistry()
    h = reg.pin_startup("run_bash", description="run a shell command", input_schema={"type": "object"})
    assert reg.has_tool("run_bash")
    assert reg.startup_hashes["run_bash"] == h


def test_registry_pin_startup_is_idempotent_on_startup_hash() -> None:
    """
    Second `pin_startup` with the same name updates the current
    cache but leaves the startup hash alone. Models the "sidecar
    re-populates within a session" path where re-pinning must
    not erase the rug-pull baseline.
    """
    reg = ToolDescriptorRegistry()
    first = reg.pin_startup("t", description="a", input_schema={})
    second = reg.pin_startup("t", description="b", input_schema={"diff": 1})
    assert reg.startup_hashes["t"] == first
    # The current cache moved on (the hash is the second one).
    assert second != first


def test_registry_verify_passes_on_unchanged_descriptor() -> None:
    """Verify is ok when the cache hasn't drifted from the pin."""
    reg = ToolDescriptorRegistry()
    reg.pin_startup("t", description="hello", input_schema={"type": "object"})
    result = reg.verify("t")
    assert result.ok is True
    assert result.mismatch_kind is None


def test_registry_verify_passes_with_benign_reason_for_unknown_tool() -> None:
    """
    Unknown tools verify ok with a "not registered" reason. G2's
    allow-list is the place to deny unknowns; the ETDI registry
    only enforces what it's been told.
    """
    reg = ToolDescriptorRegistry()
    result = reg.verify("never_pinned")
    assert result.ok is True
    assert "not registered" in result.reason


# -----------------------------------------------------------------
# Rug-pull detection -- AC 2 + AC 4
# -----------------------------------------------------------------


def test_registry_verify_detects_descriptor_change_between_calls() -> None:
    """
    AC4: mock tool's descriptor change between two `verify` calls
    is detected. The test simulates the rug-pull by calling
    `update_current` with a changed descriptor between two
    verify calls.
    """
    reg = ToolDescriptorRegistry()
    reg.pin_startup("t", description="benign", input_schema={"type": "object"})

    first = reg.verify("t")
    assert first.ok is True

    # Rug-pull: schema replaced under us.
    reg.update_current(
        "t",
        description="benign",
        input_schema={"type": "object", "required": ["secret_arg"]},
    )

    second = reg.verify("t")
    assert second.ok is False
    assert second.mismatch_kind == "startup"
    # AC2: message identifies the mismatched tool.
    assert "'t'" in second.reason or "\"t\"" in second.reason
    # AC2: it's a rug-pull, not a manifest violation.
    assert "rug-pull" in second.reason.lower()


def test_registry_verify_detects_description_change() -> None:
    """Description changes are part of the hash (per work-item spec)."""
    reg = ToolDescriptorRegistry()
    reg.pin_startup("t", description="d1", input_schema={})
    reg.update_current("t", description="d2", input_schema={})
    assert reg.verify("t").ok is False


# -----------------------------------------------------------------
# Manifest pinning -- AC 3
# -----------------------------------------------------------------


def test_registry_with_matching_manifest_verifies_ok() -> None:
    """
    AC3: manifest hash matches the discovered tool's hash -> ok.
    """
    desc = "hello"
    schema: dict[str, Any] = {"type": "object"}
    pinned_hash = hash_tool_descriptor("t", desc, schema)

    reg = ToolDescriptorRegistry(manifest={"t": pinned_hash})
    reg.pin_startup("t", description=desc, input_schema=schema)

    result = reg.verify("t")
    assert result.ok is True
    assert reg.manifest_violations == {}


def test_registry_with_conflicting_manifest_denies_with_manifest_kind() -> None:
    """
    AC3: manifest pin disagrees with the discovered tool's hash
    -> `verify` returns `ok=False`, `mismatch_kind="manifest"`.
    Models the "operator pinned a known-good hash but the live
    MCP server now ships a different tool" deployment-time
    failure.
    """
    reg = ToolDescriptorRegistry(
        manifest={"t": "0" * 64}  # 64-char hex placeholder that won't match
    )
    reg.pin_startup("t", description="hi", input_schema={"type": "object"})

    result = reg.verify("t")
    assert result.ok is False
    assert result.mismatch_kind == "manifest"
    assert "manifest" in result.reason.lower()
    assert "'t'" in result.reason


def test_registry_manifest_violation_persists_across_verify_calls() -> None:
    """
    The manifest-violation flag is set once at `pin_startup` and
    reused on every `verify`. A future "update_current" that
    happens to make the live descriptor match the manifest does
    NOT clear the violation -- the startup pin was already
    wrong.
    """
    reg = ToolDescriptorRegistry(manifest={"t": "0" * 64})
    reg.pin_startup("t", description="hi", input_schema={})
    assert reg.verify("t").mismatch_kind == "manifest"
    assert reg.verify("t").mismatch_kind == "manifest"


def test_registry_manifest_hashes_are_normalized_to_lowercase() -> None:
    """Pinned hex in mixed case still matches lowercase-emitted hashes."""
    desc = "hi"
    schema: dict[str, Any] = {"type": "object"}
    real_hash = hash_tool_descriptor("t", desc, schema)

    reg = ToolDescriptorRegistry(manifest={"t": real_hash.upper()})
    reg.pin_startup("t", description=desc, input_schema=schema)
    assert reg.verify("t").ok is True


# -----------------------------------------------------------------
# load_manifest -- AC 5 (non-fatal) plus other failure modes
# -----------------------------------------------------------------


def test_load_manifest_returns_empty_when_file_missing(
    tmp_path: Path,
) -> None:
    """AC5: missing manifest file is non-fatal -> {}."""
    result = load_manifest(tmp_path / "does-not-exist.json")
    assert result == {}


def test_load_manifest_returns_empty_for_invalid_json(
    tmp_path: Path,
) -> None:
    """Unparseable JSON -> {} (non-fatal)."""
    path = tmp_path / MANIFEST_FILENAME
    path.write_text("{ this is not json")
    assert load_manifest(path) == {}


def test_load_manifest_returns_empty_for_wrong_version(
    tmp_path: Path,
) -> None:
    """Version mismatch -> {} with a warning log."""
    path = tmp_path / MANIFEST_FILENAME
    path.write_text(json.dumps({"version": 999, "tools": {"t": "abc"}}))
    assert load_manifest(path) == {}


def test_load_manifest_returns_empty_for_non_object_root(
    tmp_path: Path,
) -> None:
    """Top-level array (or non-object) -> {}."""
    path = tmp_path / MANIFEST_FILENAME
    path.write_text(json.dumps(["t", "abc"]))
    assert load_manifest(path) == {}


def test_load_manifest_returns_empty_when_tools_missing(
    tmp_path: Path,
) -> None:
    path = tmp_path / MANIFEST_FILENAME
    path.write_text(json.dumps({"version": MANIFEST_VERSION}))
    assert load_manifest(path) == {}


def test_load_manifest_skips_non_string_entries(tmp_path: Path) -> None:
    """Malformed entries inside `tools` are skipped, not fatal."""
    path = tmp_path / MANIFEST_FILENAME
    path.write_text(
        json.dumps(
            {
                "version": MANIFEST_VERSION,
                "tools": {
                    "good": "abc123",
                    "bad_value": 42,  # non-string hash -- skipped
                },
            }
        )
    )
    result = load_manifest(path)
    assert result == {"good": "abc123"}


def test_load_manifest_round_trips_with_write_manifest(tmp_path: Path) -> None:
    """
    `write_manifest` produces a file `load_manifest` can read
    back. Pins the on-disk format so a future schema bump
    surfaces here.
    """
    path = tmp_path / MANIFEST_FILENAME
    write_manifest(
        path,
        {"run_bash": "deadbeef" * 8, "rag_search": "feedface" * 8},
        generated_at="2026-05-22T00:00:00Z",
    )
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["version"] == MANIFEST_VERSION
    assert payload["tools"] == {
        "rag_search": "feedface" * 8,
        "run_bash": "deadbeef" * 8,
    }
    assert payload["generated_at"] == "2026-05-22T00:00:00Z"
    assert load_manifest(path) == {
        "run_bash": "deadbeef" * 8,
        "rag_search": "feedface" * 8,
    }


def test_write_manifest_creates_parent_directories(tmp_path: Path) -> None:
    """The manifest writer should not fail on a fresh contracts dir."""
    nested = tmp_path / "palisade_contracts" / "subdir"
    path = nested / MANIFEST_FILENAME
    write_manifest(path, {"t": "abc"})
    assert path.exists()


# -----------------------------------------------------------------
# G2 integration -- AC 2 end-to-end
# -----------------------------------------------------------------


async def test_g2_denies_sev2_on_descriptor_rug_pull() -> None:
    """
    AC2 end-to-end: with the registry wired into G2, a
    descriptor change between two calls denies the second call
    with `incident_level=2` and a message identifying the
    tool.

    The test path mirrors how the sidecar uses the registry:
    populate -> verify clean -> rug-pull -> verify deny.
    """
    reg = ToolDescriptorRegistry()
    reg.pin_startup(
        "run_bash",
        description="run a shell command",
        input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
    )

    gate = G2ToolGate(
        enabled=True,
        allow_patterns=["*"],
        high_stakes=frozenset({"run_bash"}),
        schemas={"run_bash": {"type": "object", "properties": {"command": {"type": "string"}}}},
        tool_registry=reg,
    )

    # First call: clean.
    first = await gate.check_fast(
        {"tool_name": "run_bash", "args": {"command": "ls"}}, _ctx()
    )
    assert first.allow is True

    # Rug-pull between calls.
    reg.update_current(
        "run_bash",
        description="now does something different",
        input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
    )

    second = await gate.check_fast(
        {"tool_name": "run_bash", "args": {"command": "ls"}}, _ctx()
    )
    assert second.allow is False
    assert second.incident_level == 2
    assert "ETDI" in second.reason
    assert "run_bash" in second.reason


async def test_g2_denies_sev2_on_manifest_violation() -> None:
    """
    A startup pin that disagrees with the operator manifest
    denies on every call -- the registry returns
    `mismatch_kind="manifest"` and the gate routes it to SEV2.
    """
    reg = ToolDescriptorRegistry(manifest={"run_bash": "0" * 64})
    reg.pin_startup(
        "run_bash",
        description="d",
        input_schema={"type": "object"},
    )

    gate = G2ToolGate(
        enabled=True,
        allow_patterns=["*"],
        tool_registry=reg,
    )

    decision = await gate.check_fast(
        {"tool_name": "run_bash", "args": {}}, _ctx()
    )
    assert decision.allow is False
    assert decision.incident_level == 2
    assert "manifest" in decision.reason.lower()


async def test_g2_etdi_check_runs_before_schema() -> None:
    """
    A rug-pulled tool with malformed args denies on ETDI (SEV2
    with "rug-pull" message), not on schema (also SEV2 but a
    different message). The ordering matters for audit trails:
    a rug-pull is the more diagnostically informative finding.
    """
    reg = ToolDescriptorRegistry()
    reg.pin_startup(
        "run_bash",
        description="d",
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    )
    reg.update_current(
        "run_bash",
        description="d-changed",  # rug-pull
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    )

    gate = G2ToolGate(
        enabled=True,
        allow_patterns=["*"],
        schemas={
            "run_bash": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            }
        },
        tool_registry=reg,
    )

    # Args are garbage (missing required `command`) AND the
    # descriptor has been rug-pulled. The ETDI check fires first.
    decision = await gate.check_fast(
        {"tool_name": "run_bash", "args": {}}, _ctx()
    )
    assert decision.allow is False
    assert decision.incident_level == 2
    assert "rug-pull" in decision.reason.lower()


async def test_g2_etdi_skips_when_registry_is_none() -> None:
    """
    With no registry attached (unit-test fixture), the ETDI step
    is a no-op. Lets the other G2 tests in `test_g2_fast.py`
    keep working without constructing a registry.
    """
    gate = G2ToolGate(
        enabled=True,
        allow_patterns=["*"],
        tool_registry=None,
    )
    decision = await gate.check_fast(
        {"tool_name": "any", "args": {}}, _ctx()
    )
    assert decision.allow is True


async def test_g2_etdi_skips_for_unregistered_tools() -> None:
    """
    A tool that is not in the registry (no startup pin) passes
    the ETDI step. G2's allow-list is the appropriate place to
    catch unknown tools.
    """
    reg = ToolDescriptorRegistry()
    reg.pin_startup("known", description="d", input_schema={})

    gate = G2ToolGate(
        enabled=True,
        allow_patterns=["*"],
        tool_registry=reg,
    )
    decision = await gate.check_fast(
        {"tool_name": "unregistered", "args": {}}, _ctx()
    )
    assert decision.allow is True


# -----------------------------------------------------------------
# Sidecar integration -- end-to-end with populate
# -----------------------------------------------------------------


class _FakeMcpServerFactory:
    """
    Minimal MCP-server stub matching the pattern in
    `test_g2_annotations.py`. Hand-rolled so a single change in
    PydanticAI's `MCPServerStreamableHTTP` surface trips the
    other annotation tests too -- we want one clear failure
    point, not silent skew.
    """

    def __init__(self, tools: list[Tool]) -> None:
        self._tools = tools

    def __call__(self, *, url: str, timeout: float) -> Any:
        outer = self

        class _Server:
            async def list_tools(self) -> list[Tool]:
                return list(outer._tools)

        return _Server()


async def test_sidecar_pins_descriptors_during_populate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    `populate_tool_metadata` pins descriptors via the registry.
    After populate, verifying every discovered tool succeeds and
    the startup hashes match the canonical-descriptor hashes
    computed from the same source data.
    """
    tools = [
        _tool(
            "run_bash",
            description="run a shell command",
            input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
            destructive=True,
        ),
        _tool(
            "rag_search",
            description="search a corpus",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        ),
    ]
    monkeypatch.setattr(
        g2_tool_module, "MCPServerStreamableHTTP", _FakeMcpServerFactory(tools)
    )

    settings = PalisadeSettings(enabled=True, g2_enabled=True)
    sidecar = PalisadeSidecar(settings, _make_project())
    await sidecar.populate_tool_metadata("http://mock/mcp")

    registry = sidecar.tool_registry
    assert registry.has_tool("run_bash")
    assert registry.has_tool("rag_search")

    expected_run_bash_hash = hash_tool_descriptor(
        "run_bash",
        "run a shell command",
        {"type": "object", "properties": {"command": {"type": "string"}}},
    )
    assert registry.startup_hashes["run_bash"] == expected_run_bash_hash

    # Both verify clean immediately after populate.
    assert registry.verify("run_bash").ok is True
    assert registry.verify("rag_search").ok is True


async def test_sidecar_loads_manifest_at_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    AC3: the sidecar reads `<contracts_dir>/<MANIFEST_FILENAME>`
    at construction and seeds the registry with pinned hashes.
    """
    manifest_path = tmp_path / MANIFEST_FILENAME
    write_manifest(
        manifest_path,
        {"run_bash": "deadbeef" * 8},  # fake hash; won't match real one
    )

    settings = PalisadeSettings(
        enabled=True,
        g2_enabled=True,
        contracts_dir=str(tmp_path),
    )
    sidecar = PalisadeSidecar(settings, _make_project())

    assert sidecar.tool_registry.manifest_pinned == {"run_bash": "deadbeef" * 8}


async def test_sidecar_construction_is_non_fatal_when_manifest_missing(
    tmp_path: Path,
) -> None:
    """
    AC5: a sidecar constructed against a `contracts_dir` with no
    manifest file does NOT raise; the registry just has an
    empty pinned set.
    """
    settings = PalisadeSettings(
        enabled=True,
        g2_enabled=True,
        contracts_dir=str(tmp_path / "does-not-exist"),
    )
    sidecar = PalisadeSidecar(settings, _make_project())
    assert sidecar.tool_registry.manifest_pinned == {}


async def test_sidecar_construction_is_non_fatal_when_manifest_malformed(
    tmp_path: Path,
) -> None:
    """Malformed manifest -> empty pinned set, no crash."""
    (tmp_path / MANIFEST_FILENAME).write_text("{ not valid json")
    settings = PalisadeSettings(
        enabled=True,
        g2_enabled=True,
        contracts_dir=str(tmp_path),
    )
    sidecar = PalisadeSidecar(settings, _make_project())
    assert sidecar.tool_registry.manifest_pinned == {}


async def test_sidecar_g2_denies_on_manifest_mismatch_after_populate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    End-to-end: a manifest pin that disagrees with the live
    MCP-server descriptor causes G2 to deny the tool with SEV2.
    """
    # Manifest pins a bogus hash.
    write_manifest(
        tmp_path / MANIFEST_FILENAME,
        {"run_bash": "0" * 64},
    )

    tools = [
        _tool(
            "run_bash",
            description="d",
            input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
            destructive=True,
        ),
    ]
    monkeypatch.setattr(
        g2_tool_module, "MCPServerStreamableHTTP", _FakeMcpServerFactory(tools)
    )

    settings = PalisadeSettings(
        enabled=True,
        g2_enabled=True,
        contracts_dir=str(tmp_path),
    )
    sidecar = PalisadeSidecar(settings, _make_project(tools=["*"]))
    await sidecar.populate_tool_metadata("http://mock/mcp")

    g2 = sidecar.gates["G2"]
    assert isinstance(g2, G2ToolGate)

    decision = await g2.check_fast(
        {"tool_name": "run_bash", "args": {"command": "ls"}}, _ctx()
    )
    assert decision.allow is False
    assert decision.incident_level == 2
    assert "manifest" in decision.reason.lower()


# -----------------------------------------------------------------
# Snapshot helper -- operator-tooling-facing API
# -----------------------------------------------------------------


def test_snapshot_manifest_returns_startup_hashes() -> None:
    """
    `snapshot_manifest` returns `{tool_name: startup_hash}` for
    use by ops tooling that wants to capture the live deployment
    state into a manifest file.
    """
    reg = ToolDescriptorRegistry()
    reg.pin_startup("a", description="x", input_schema={})
    reg.pin_startup("b", description="y", input_schema={"type": "object"})

    snap = reg.snapshot_manifest()
    assert set(snap.keys()) == {"a", "b"}
    assert snap["a"] == hash_tool_descriptor("a", "x", {})
    assert snap["b"] == hash_tool_descriptor("b", "y", {"type": "object"})


# -----------------------------------------------------------------
# HashVerification dataclass surface
# -----------------------------------------------------------------


def test_hash_verification_is_frozen() -> None:
    """The result dataclass is frozen so audit-trail consumers can't mutate it."""
    v = HashVerification(ok=True)
    with pytest.raises(Exception):
        v.ok = False  # type: ignore[misc]


# -----------------------------------------------------------------
# Anyio backend selection (matches the other palisade tests)
# -----------------------------------------------------------------


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
