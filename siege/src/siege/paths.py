"""Filesystem anchors for the SIEGE benchmark and the analysis modules.

SIEGE is not a runtime dependency of anything — it replays a corpus through a
gate stack offline — so unlike :mod:`palisade.paths` this module may require a
checkout, and does.

``SIEGE_DIR`` is derived from this file rather than from the repository root,
so the package can be relocated within the tree without repointing the corpus.
"""

from __future__ import annotations

from pathlib import Path

from palisade.paths import require_repo_root

SIEGE_DIR = Path(__file__).resolve().parent
"""The installed ``siege`` package directory."""

REPO_ROOT = require_repo_root(SIEGE_DIR)
"""Root of the artifact checkout."""

CORPUS_DIR = SIEGE_DIR / "corpus"
"""SIEGE-S: 42 attack classes (205 instances) plus the 181-task benign
control, which lives here as two further directories -- ``benign_workload``
(the deployment's own 24-task example workload) and ``benign_diverse`` (the
157-task near-manifold set). One directory is one class; one file is one
instance."""

CONTROLS_DIR = SIEGE_DIR / "controls"
"""Targeted controls added beyond the 181-task benign set: the
provenance-carrying benign control and the expanded citation-grounding
control. These are *not* part of the 181 and are not in any reported FPR."""

GENERATED_DIR = SIEGE_DIR / "corpus_generated"
"""The reward-ranked evasive pools (4 classes x 20)."""

DOCS_DIR = REPO_ROOT / "docs" / "palisade"
"""Where the analysis modules write their Markdown reports. See
``docs/palisade/README.md`` for the map from a reported number to the module
that produces it."""

CONTRACTS_DIR = REPO_ROOT / "palisade_contracts"
"""Operator contract directory, loaded beside the built-in contract library."""

__all__ = [
    "SIEGE_DIR",
    "REPO_ROOT",
    "CORPUS_DIR",
    "CONTROLS_DIR",
    "GENERATED_DIR",
    "DOCS_DIR",
    "CONTRACTS_DIR",
]
