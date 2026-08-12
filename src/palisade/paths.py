"""Filesystem anchors for the PALISADE runtime.

PALISADE ships as a dependency of a host application, so this module must
import cleanly from ``site-packages`` where there is no repository around it.
Only anchors that survive that live here:

``PALISADE_DIR``
    The installed package directory. Always valid. Gate code resolves its
    bundled assets (jailbreak signatures, the Semgrep ruleset) relative to
    this, not to a checkout.

``find_repo_root`` / ``require_repo_root``
    A checkout root, when there is one. The benchmark and the analysis
    modules need it and are only ever run from a checkout; the runtime does
    not. ``find_repo_root`` returns ``None`` rather than raising, so nothing
    breaks at import time for an installed deployment.

The corpus, control and report anchors live in :mod:`siege.paths`, which is
free to require a checkout because SIEGE is not a runtime dependency.
"""

from __future__ import annotations

from pathlib import Path

PALISADE_DIR = Path(__file__).resolve().parent
"""The installed ``palisade`` package directory."""

# The artifact root, distinctively. `pyproject.toml` alone is not enough: the
# repository is a uv workspace, so a member directory (siege/) carries one too
# and a walk upward would stop there. `tools/` and `docs/` exist only at the
# root, and both are load-bearing for the analysis modules that call this.
_MARKERS = ("pyproject.toml", "tools", "docs")


def find_repo_root(start: Path | None = None) -> Path | None:
    """The checkout root above ``start``, or ``None`` when installed.

    Found by walking up to the repository markers rather than by counting
    parent directories, so a checkout can be relocated or vendored without
    silently repointing anything at the wrong tree.
    """
    origin = (start or PALISADE_DIR).resolve()
    for candidate in (origin, *origin.parents):
        if all((candidate / m).exists() for m in _MARKERS):
            return candidate
    return None


def require_repo_root(start: Path | None = None) -> Path:
    """The checkout root, or a clear error naming why there isn't one.

    For callers that genuinely need the repository — the benchmark, the
    analysis modules, the operator CLIs run from a clone. Runtime code must
    not call this: PALISADE is installed as a library into host applications
    that have no PALISADE checkout.
    """
    root = find_repo_root(start)
    if root is None:
        raise RuntimeError(
            f"no PALISADE checkout above {(start or PALISADE_DIR).resolve()}. "
            f"This code path needs the repository (corpus, reports, or "
            f"operator policy files) and PALISADE appears to be installed as "
            f"a library instead. Run it from a clone of the artifact."
        )
    return root


__all__ = ["PALISADE_DIR", "find_repo_root", "require_repo_root"]
