"""The host-application contract PALISADE is written against.

PALISADE is a sidecar: it wraps an existing agentic framework rather than
owning one. Two objects cross that boundary at session construction -- the
project (or workspace) the agent is acting within, and the authenticated
principal it is acting for. This module states the minimum shape the sidecar
reads from each, so a host application can satisfy the contract without
PALISADE depending on the host's own models.

The reference deployment is VISTA, whose ``ProjectPublic`` and
``UserPublicWithConfig`` are structurally supersets of the models here; a host
integrating PALISADE may pass its own objects instead, provided they carry
these attributes. Extra fields are preserved, so passing a richer host model
loses nothing.

What the sidecar actually reads:

``HostProject.tools`` and ``HostProject.knowledge_bases``
    G2's allow-list derives the permitted tool patterns from the project, and
    a project with no knowledge base has ``rag_search`` withheld
    (``PalisadeSidecar._tool_patterns``).

``HostProject.id``
    Keys the per-session capability registry, trust posteriors and
    provenance workflow.

``HostUser``
    Identifies the authenticated principal. The trusted-principal exemption
    in the taint rule (Section III-D of the paper) is keyed on the *source*
    label recorded at ingress, not on this object; the sidecar never asks
    the host whether a value should be trusted.

Nothing here is a security boundary. These are inputs the host supplies, and
the threat model places the host application outside the trusted computing
base for everything except session identity.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field


class HostProject(BaseModel):
    """The workspace a guarded agent session runs inside."""

    model_config = ConfigDict(extra="allow")

    id: uuid.UUID
    name: str
    description: str | None = None
    system_prompt: str | None = None

    skills: list[str] = Field(default_factory=list)
    """Skill identifiers exposed to the agent."""

    knowledge_bases: list[str] = Field(default_factory=list)
    """Retrieval-corpus slugs in scope. An empty list withholds ``rag_search``."""

    tools: list[str] = Field(default_factory=list)
    """Tool-name patterns for the G2 allow-list (fnmatch; ``!name`` excludes)."""


class HostUser(BaseModel):
    """The authenticated principal a guarded session acts for."""

    model_config = ConfigDict(extra="allow")

    id: uuid.UUID
    email: str
    is_admin: bool = False

    remote_hpc_jobs_dir: str | None = None
    """Where the host stages batch scripts, when it submits to a scheduler."""


__all__ = ["HostProject", "HostUser"]
