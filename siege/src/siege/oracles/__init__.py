"""
SIEGE correctness oracle + ground-truth tables (WI11).

Public surface:

- ``CorrectnessOracle`` / ``OracleVerdict`` — run the contract registry
  over scientific claims, returning ``OK | VIOLATION(reason)``.
- ``extract_instance_claims`` — pull the ``payload["claim"]`` declarations
  off a SIEGE instance.
- ``ground_truth`` — the dependency-free lookup API the WI11 contracts call.
"""

from __future__ import annotations

from siege.oracles import ground_truth
from siege.oracles.correctness_oracle import (
    CorrectnessOracle,
    OracleVerdict,
    extract_instance_claims,
)

__all__ = [
    "CorrectnessOracle",
    "OracleVerdict",
    "extract_instance_claims",
    "ground_truth",
]
