"""Filesystem anchors for the PALISADE/SIEGE artifact.

Every anchor is found by walking up to the repository marker rather than by
counting parent directories, so the checkout can be relocated, vendored or
installed without silently repointing the analysis modules at nothing.

The analysis modules under ``tools/`` read the corpus and write their reports
through these constants; see ``docs/palisade/README.md`` for the map from a
reported number to the module that produces it.
"""

from __future__ import annotations

from pathlib import Path

_MARKERS = ("pyproject.toml", "src")


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if all((candidate / m).exists() for m in _MARKERS):
            return candidate
    raise RuntimeError(
        f"could not locate the repository root above {start}: expected a "
        f"directory containing {' and '.join(_MARKERS)}"
    )


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
"""Root of the artifact checkout."""

SRC_DIR = REPO_ROOT / "src"
PALISADE_DIR = SRC_DIR / "palisade"
SIEGE_DIR = SRC_DIR / "siege"

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

DOCS_DIR = REPO_ROOT / "docs" / "palisade"
"""Where the analysis modules write their Markdown reports."""

CONTRACTS_DIR = REPO_ROOT / "palisade_contracts"
"""Operator contract directory, loaded beside the built-in contract library."""

__all__ = [
    "REPO_ROOT",
    "SRC_DIR",
    "PALISADE_DIR",
    "SIEGE_DIR",
    "CORPUS_DIR",
    "CONTROLS_DIR",
    "DOCS_DIR",
    "CONTRACTS_DIR",
]
