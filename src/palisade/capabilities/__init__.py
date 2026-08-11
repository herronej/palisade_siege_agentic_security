"""
PALISADE capabilities package.
"""

from palisade.capabilities.registry import (
    CapabilityRegistry,
    CapabilityTag,
    DualUseMarker,
    SensitivityTier,
    TrustTier,
)
from palisade.capabilities.propagation import (
    propagate_taint,
    shares_distinctive_content,
    tainted_sources_for,
)
from palisade.capabilities.approval import (
    ApprovalOutcome,
    RequestApprovalFn,
    PalisadeApprovalCapability,
)
from palisade.capabilities.base import PalisadeCapability
from palisade.capabilities.exceptions import PalisadeDeny
from palisade.capabilities.g1_prompt import G1PromptCapability
from palisade.capabilities.g2_tool import G2ToolCapability
from palisade.capabilities.g3_rag import G3RagCapability
from palisade.capabilities.g4_code import G4CodeCapability
from palisade.capabilities.g5_hpc import G5HpcCapability
from palisade.capabilities.g6_egress import G6EgressCapability

__all__ = [
    "ApprovalOutcome",
    "CapabilityRegistry",
    "CapabilityTag",
    "DualUseMarker",
    "G1PromptCapability",
    "G2ToolCapability",
    "G3RagCapability",
    "G4CodeCapability",
    "G5HpcCapability",
    "G6EgressCapability",
    "RequestApprovalFn",
    "SensitivityTier",
    "TrustTier",
    "PalisadeApprovalCapability",
    "PalisadeCapability",
    "PalisadeDeny",
    "propagate_taint",
    "shares_distinctive_content",
    "tainted_sources_for",
]
