"""The host-application contract PALISADE is written against.

PALISADE is a sidecar: it wraps an existing agentic framework rather than
owning one. Two objects cross that boundary at session construction -- the
project (or workspace) the agent is acting within, and the authenticated
principal it is acting for.

Both are **structural** contracts (:class:`typing.Protocol`), not base classes
a host must inherit and not models a host must convert to. A host passes its
own objects unchanged; a type checker verifies they carry the attributes the
sidecar reads, and nothing is validated or copied at runtime. The reference
deployment's ``ProjectPublic`` and ``UserPublicWithConfig`` satisfy these as
they stand, with no adapter.

Concrete implementations (:class:`HostProjectModel`, :class:`HostUserModel`)
are provided for standalone tests and for hosts that have no such object of
their own. They are a convenience, never a requirement.

What the sidecar actually reads:

``project.tools`` and ``project.knowledge_bases``
    G2's allow-list derives the permitted tool patterns from the project, and
    a project with no knowledge base has ``rag_search`` withheld
    (``PalisadeSidecar._tool_patterns``).

``project.id``
    Keys the per-session capability registry, trust posteriors and provenance
    workflow.

``user``
    Identifies the authenticated principal. The trusted-principal exemption in
    the taint rule (paper, Section III-D) is keyed on the *source* label
    recorded at ingress, not on this object; the sidecar never asks the host
    whether a value should be trusted.

Nothing here is a security boundary. These are inputs the host supplies, and
the threat model places the host application outside the trusted computing
base for everything except session identity.
"""

from __future__ import annotations

import uuid
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


@runtime_checkable
class HostProject(Protocol):
    """The workspace a guarded agent session runs inside.

    Structural: any object carrying these attributes satisfies it. Note that
    ``runtime_checkable`` protocols check attribute *presence* only, so
    ``isinstance`` here is a smoke test, not validation.
    """

    @property
    def id(self) -> uuid.UUID:
        """Keys the session's capability registry, trust state and provenance."""

    @property
    def name(self) -> str: ...

    @property
    def knowledge_bases(self) -> list[str]:
        """Retrieval-corpus slugs in scope. Empty withholds ``rag_search``."""

    @property
    def tools(self) -> list[str]:
        """Tool-name patterns for the G2 allow-list (fnmatch; ``!name`` excludes)."""


@runtime_checkable
class HostUser(Protocol):
    """The authenticated principal a guarded session acts for."""

    @property
    def id(self) -> uuid.UUID: ...

    @property
    def email(self) -> str: ...


class HostProjectModel(BaseModel):
    """A concrete :class:`HostProject`, for hosts that have no project object
    of their own and for PALISADE's standalone tests."""

    model_config = ConfigDict(extra="allow")

    id: uuid.UUID
    name: str
    description: str | None = None
    system_prompt: str | None = None

    skills: list[str] = Field(default_factory=list)
    knowledge_bases: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)


class HostUserModel(BaseModel):
    """A concrete :class:`HostUser`, for the same purpose."""

    model_config = ConfigDict(extra="allow")

    id: uuid.UUID
    email: str
    is_admin: bool = False
    remote_hpc_jobs_dir: str | None = None


__all__ = [
    "HostProject",
    "HostUser",
    "HostProjectModel",
    "HostUserModel",
]
