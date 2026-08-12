"""
Unit tests for the G5 settings + operator policy-file loading
(``Add G5 settings + operator policy files``).

Covers:

- The new ``PalisadeSettings`` fields and their defaults.
- ``load_allocation_policy``: missing / malformed / wrong-version file ->
  bundled defaults + WARNING (matching the ``g3_kb_policy.json`` pattern);
  a valid file parses.
- Bundled mining denylist (>= 10 IOCs) and the OLCF / NERSC / OSTI host
  allow-list.
- Sidecar startup wiring: the operator override at
  ``<contracts_dir>/g5_allocation_policy.json`` is read; the explicit
  ``g5_allocation_policy_path`` wins; ``g5_resource_ceilings`` fills in
  when the file supplies no allocations; ``g5_binary_denylist`` augments;
  ``g5_chained_job_dag_enabled`` reaches the gate.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import pytest

from palisade.host import HostProjectModel
from palisade.config import PalisadeSettings
from palisade.gates.g5_hpc import (
    ALLOCATION_POLICY_FILENAME,
    ALLOCATION_POLICY_VERSION,
    DEFAULT_HOST_ALLOW_LIST,
    DEFAULT_MINING_BINARY_DENYLIST,
    AllocationPolicy,
    G5HpcJobGate,
    load_allocation_policy,
)
from palisade.sidecar import PalisadeSidecar


# -----------------------------------------------------------------
# Settings fields + defaults
# -----------------------------------------------------------------


def test_g5_settings_defaults() -> None:
    s = PalisadeSettings()
    assert s.g5_enabled is False
    assert s.g5_allocation_policy_path is None
    assert s.g5_resource_ceilings == {}
    assert s.g5_binary_denylist == frozenset()
    assert s.g5_chained_job_dag_enabled is True


def test_g5_settings_overridable() -> None:
    s = PalisadeSettings(
        g5_enabled=True,
        g5_allocation_policy_path="/etc/vista/g5.json",
        g5_resource_ceilings={"a": {"max_nodes": 4}},
        g5_binary_denylist={"customminer"},
        g5_chained_job_dag_enabled=False,
    )
    assert s.g5_allocation_policy_path == "/etc/vista/g5.json"
    assert s.g5_resource_ceilings == {"a": {"max_nodes": 4}}
    assert s.g5_binary_denylist == frozenset({"customminer"})
    assert s.g5_chained_job_dag_enabled is False


# -----------------------------------------------------------------
# Bundled defaults (acceptance: >= 10 IOCs, OLCF/NERSC/OSTI hosts)
# -----------------------------------------------------------------


def test_bundled_denylist_has_at_least_10_entries() -> None:
    assert len(DEFAULT_MINING_BINARY_DENYLIST) >= 10
    for known in ("xmrig", "ethminer", "phoenixminer", "t-rex"):
        assert known in DEFAULT_MINING_BINARY_DENYLIST


def test_bundled_host_allow_list_includes_doe_hosts() -> None:
    assert "*.olcf.ornl.gov" in DEFAULT_HOST_ALLOW_LIST
    assert "*.nersc.gov" in DEFAULT_HOST_ALLOW_LIST
    assert "*.osti.gov" in DEFAULT_HOST_ALLOW_LIST


# -----------------------------------------------------------------
# load_allocation_policy: fallback paths
# -----------------------------------------------------------------


_VALID_POLICY = {
    "version": 1,
    "allocations": {
        "approved-research": {
            "max_nodes": 64,
            "max_time_seconds": 14400,
            "max_gpus": 8,
            "permitted_partitions": ["batch", "gpu"],
        },
        "constrained-resource": {
            "max_nodes": 4,
            "max_time_seconds": 1800,
            "max_gpus": 0,
            "permitted_partitions": ["debug"],
        },
    },
    "binary_denylist": ["xmrig", "ethminer", "phoenixminer", "t-rex"],
    "host_allow_list": ["*.olcf.ornl.gov", "*.nersc.gov", "*.osti.gov"],
}


def test_missing_file_uses_defaults_no_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="palisade.gates.g5_hpc"):
        policy = load_allocation_policy(tmp_path / "absent.json")
    assert policy.allocations_enforced is False
    # Missing file is an INFO, not a WARNING (matches load_kb_policy).
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_malformed_json_uses_defaults_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    f = tmp_path / ALLOCATION_POLICY_FILENAME
    f.write_text("{ not: valid json ", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        policy = load_allocation_policy(f)
    assert policy.allocations_enforced is False
    assert any("not valid JSON" in r.message for r in caplog.records)


def test_wrong_version_uses_defaults_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    f = tmp_path / ALLOCATION_POLICY_FILENAME
    f.write_text(json.dumps({"version": 99, "allocations": {}}), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        policy = load_allocation_policy(f)
    assert policy.allocations_enforced is False
    assert any("rejected" in r.message for r in caplog.records)


def test_valid_file_parses(tmp_path: Path) -> None:
    f = tmp_path / ALLOCATION_POLICY_FILENAME
    f.write_text(json.dumps(_VALID_POLICY), encoding="utf-8")
    policy = load_allocation_policy(f)
    assert policy.allocations_enforced is True
    assert policy.loaded_from == f
    assert policy.allocations["approved-research"].max_nodes == 64
    assert policy.allocations["constrained-resource"].permitted_partitions == frozenset(
        {"debug"}
    )
    # File denylist augments the bundled IOCs (never replaces).
    assert "xmrig" in policy.binary_denylist
    assert len(policy.binary_denylist) >= len(DEFAULT_MINING_BINARY_DENYLIST)
    assert "*.nersc.gov" in policy.host_allow_list


def test_unreadable_path_uses_defaults_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A directory at the policy path is not readable as a file.
    d = tmp_path / ALLOCATION_POLICY_FILENAME
    d.mkdir()
    with caplog.at_level(logging.WARNING):
        policy = load_allocation_policy(d)
    assert policy.allocations_enforced is False


# -----------------------------------------------------------------
# Sidecar startup wiring
# -----------------------------------------------------------------


def _project() -> HostProject:
    return HostProjectModel(
        id=uuid.uuid4(), name="g5-policy", description=None, system_prompt=None,
        skills=[], knowledge_bases=[], tools=[], usage_limits={},
    )


def _g5(sidecar: PalisadeSidecar) -> G5HpcJobGate:
    gate = sidecar.gates["G5"]
    assert isinstance(gate, G5HpcJobGate)
    return gate


def test_operator_override_in_contracts_dir_is_read(tmp_path: Path) -> None:
    (tmp_path / ALLOCATION_POLICY_FILENAME).write_text(
        json.dumps(_VALID_POLICY), encoding="utf-8"
    )
    settings = PalisadeSettings(
        enabled=True, g5_enabled=True, contracts_dir=str(tmp_path)
    )
    sidecar = PalisadeSidecar(settings, _project())
    policy = _g5(sidecar).policy
    assert policy.allocations_enforced is True
    assert policy.allocations["approved-research"].max_nodes == 64


def test_explicit_policy_path_overrides_contracts_dir(tmp_path: Path) -> None:
    # contracts_dir holds nothing; the explicit path points elsewhere.
    explicit = tmp_path / "elsewhere" / "policy.json"
    explicit.parent.mkdir()
    explicit.write_text(json.dumps(_VALID_POLICY), encoding="utf-8")
    settings = PalisadeSettings(
        enabled=True,
        g5_enabled=True,
        contracts_dir=str(tmp_path / "empty"),
        g5_allocation_policy_path=str(explicit),
    )
    sidecar = PalisadeSidecar(settings, _project())
    assert _g5(sidecar).policy.allocations_enforced is True


def test_missing_file_falls_back_to_settings_ceilings(tmp_path: Path) -> None:
    settings = PalisadeSettings(
        enabled=True,
        g5_enabled=True,
        contracts_dir=str(tmp_path),  # no policy file here
        g5_resource_ceilings={
            "approved-research": {
                "max_nodes": 16,
                "max_time_seconds": 3600,
                "max_gpus": 4,
                "permitted_partitions": ["batch"],
            }
        },
    )
    sidecar = PalisadeSidecar(settings, _project())
    policy = _g5(sidecar).policy
    assert policy.allocations_enforced is True
    assert policy.allocations["approved-research"].max_nodes == 16


def test_file_allocations_win_over_settings_ceilings(tmp_path: Path) -> None:
    (tmp_path / ALLOCATION_POLICY_FILENAME).write_text(
        json.dumps(_VALID_POLICY), encoding="utf-8"
    )
    settings = PalisadeSettings(
        enabled=True,
        g5_enabled=True,
        contracts_dir=str(tmp_path),
        g5_resource_ceilings={"approved-research": {"max_nodes": 1}},
    )
    sidecar = PalisadeSidecar(settings, _project())
    # The file's 64-node cap wins over the settings' 1-node ceiling.
    assert _g5(sidecar).policy.allocations["approved-research"].max_nodes == 64


def test_settings_binary_denylist_augments(tmp_path: Path) -> None:
    settings = PalisadeSettings(
        enabled=True,
        g5_enabled=True,
        contracts_dir=str(tmp_path),
        g5_binary_denylist={"MyCustomMiner"},
    )
    sidecar = PalisadeSidecar(settings, _project())
    denylist = _g5(sidecar).policy.binary_denylist
    assert "mycustomminer" in denylist  # lower-cased
    assert "xmrig" in denylist  # bundled list still present


def test_chained_job_dag_toggle_reaches_gate(tmp_path: Path) -> None:
    on = PalisadeSidecar(
        PalisadeSettings(enabled=True, g5_enabled=True, contracts_dir=str(tmp_path)),
        _project(),
    )
    assert _g5(on).chained_job_dag_enabled is True

    off = PalisadeSidecar(
        PalisadeSettings(
            enabled=True, g5_enabled=True, contracts_dir=str(tmp_path),
            g5_chained_job_dag_enabled=False,
        ),
        _project(),
    )
    assert _g5(off).chained_job_dag_enabled is False


def test_no_policy_no_ceilings_is_unenforced(tmp_path: Path) -> None:
    settings = PalisadeSettings(
        enabled=True, g5_enabled=True, contracts_dir=str(tmp_path)
    )
    sidecar = PalisadeSidecar(settings, _project())
    policy = _g5(sidecar).policy
    assert policy.allocations_enforced is False
    # Bundled denylist + host allow-list are still in effect.
    assert len(policy.binary_denylist) >= 10
    assert "*.nersc.gov" in policy.host_allow_list
