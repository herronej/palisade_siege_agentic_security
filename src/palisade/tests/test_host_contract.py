"""The host contract is structural, and must stay that way.

PALISADE ships as a dependency of a host application. If the sidecar ever
starts requiring its own concrete model — by validating, by ``isinstance``
gating, or by touching an attribute the contract does not declare — every host
needs an adapter, and the reference deployment would have to convert its
``ProjectPublic`` on every session construction.

These tests fail if that happens.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import pytest

from palisade.config import PalisadeSettings
from palisade.host import HostProject, HostProjectModel, HostUser, HostUserModel
from palisade.sidecar import PalisadeSidecar


@dataclass
class ForeignProject:
    """A host's own project object. Deliberately not a pydantic model, not a
    subclass of anything of ours, and carrying extra fields we never declared —
    which is what a real host looks like."""

    id: uuid.UUID
    name: str
    knowledge_bases: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    owner_email: str = "someone@example.org"
    created_at: str = "2026-01-01"


@dataclass
class ForeignUser:
    id: uuid.UUID
    email: str
    is_admin: bool = False


def _foreign() -> ForeignProject:
    return ForeignProject(
        id=uuid.uuid4(), name="msr-thermo", knowledge_bases=["salts"], tools=["rag_search"]
    )


def test_foreign_project_satisfies_the_contract() -> None:
    assert isinstance(_foreign(), HostProject)
    assert isinstance(ForeignUser(id=uuid.uuid4(), email="a@b.example"), HostUser)


def test_our_own_models_satisfy_it_too() -> None:
    assert isinstance(HostProjectModel(id=uuid.uuid4(), name="p"), HostProject)
    assert isinstance(HostUserModel(id=uuid.uuid4(), name="u", email="a@b.example"), HostUser)


def test_sidecar_accepts_a_foreign_project_unconverted() -> None:
    """The whole point: a host passes its own object, and gets it back."""
    project = _foreign()
    sidecar = PalisadeSidecar(PalisadeSettings(), project)
    assert sidecar.project is project, "the sidecar copied or coerced the host's object"


def test_sidecar_reads_only_the_declared_attributes() -> None:
    """A project carrying *only* what the contract declares must work.

    If this fails, the sidecar has grown a dependency on a field the contract
    does not advertise, and hosts will break on an upgrade without warning.
    """

    @dataclass
    class Minimal:
        id: uuid.UUID
        name: str
        knowledge_bases: list[str]
        tools: list[str]

    sidecar = PalisadeSidecar(
        PalisadeSettings(enabled=True, g2_enabled=True),
        Minimal(id=uuid.uuid4(), name="m", knowledge_bases=["kb"], tools=["rag_search"]),
    )
    # Exercise the paths that read the project.
    assert sidecar.project.name == "m"
    sidecar.build_capabilities()


def test_flag_off_yields_no_capabilities() -> None:
    """The modularity invariant, restated against a foreign host object.

    With the master flag off the hook list is empty and the agent is
    byte-identical to the unguarded baseline (paper, Section III-A).
    """
    sidecar = PalisadeSidecar(PalisadeSettings(), _foreign())
    assert sidecar.build_capabilities() == []


@pytest.mark.parametrize("attr", ["id", "name", "knowledge_bases", "tools"])
def test_contract_declares_every_attribute_the_sidecar_reads(attr: str) -> None:
    """Guard against the contract and the implementation drifting apart."""
    assert hasattr(HostProject, attr), (
        f"the sidecar reads project.{attr}; the HostProject contract must "
        f"declare it or hosts cannot know to provide it"
    )
