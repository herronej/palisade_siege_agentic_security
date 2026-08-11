"""The result-to-module map has to stay true, or it is worse than nothing.

`docs/palisade/README.md` is the artifact's promise that every reported number
names the module producing it. These tests check the two ways that promise
rots: a module named in the map that does not exist, and a report named in the
map that was never committed.
"""

from __future__ import annotations

import re
from pathlib import Path

from palisade.paths import DOCS_DIR, REPO_ROOT

MAP_PATH = DOCS_DIR / "README.md"
TOOLS_DIR = REPO_ROOT / "tools"

# Named in the map's prose as generating no result of their own.
NO_REPORT = {"palisade_screen", "eval_figures", "ablation_plots", "freeze_tool_manifest"}


def _map_text() -> str:
    return MAP_PATH.read_text()


def test_map_exists() -> None:
    assert MAP_PATH.is_file(), f"the Artifact Description promises {MAP_PATH}"


def test_every_named_module_exists() -> None:
    """No row may point at a module that isn't here."""
    named = sorted(set(re.findall(r"`tools\.([a-z0-9_]+)`", _map_text())))
    assert named, "the map names no modules at all"
    missing = [m for m in named if not (TOOLS_DIR / f"{m}.py").is_file()]
    assert not missing, f"map names nonexistent modules: {missing}"


def test_every_named_report_exists() -> None:
    """No row may point at a report that was never committed."""
    claimed = set(re.findall(r"\|\s*`([a-z0-9_]+\.md)`\s*\|", _map_text()))
    assert claimed, "the map names no reports at all"
    missing = sorted(c for c in claimed if not (DOCS_DIR / c).is_file())
    assert not missing, (
        f"map names reports with no committed artifact: {missing}. Either "
        f"generate them or mark the row as uncommitted -- an unresolvable "
        f"pointer is what the map exists to prevent."
    )


def test_every_analysis_module_is_accounted_for() -> None:
    """And no module may be silently absent from the map."""
    named = set(re.findall(r"`tools\.([a-z0-9_]+)`", _map_text()))
    have = {
        p.stem
        for p in TOOLS_DIR.glob("*.py")
        if p.stem != "__init__"
    }
    unmapped = sorted(have - named - NO_REPORT)
    assert not unmapped, (
        f"analysis modules missing from the result map: {unmapped}. Every "
        f"module either produces a reported number or is listed as producing "
        f"none."
    )


def test_shipped_reports_carry_their_provenance_banner() -> None:
    """A generated report must say which module regenerates it."""
    hand_written = {"README.md", "RELEASE_SAFETY.md", "g4_semgrep_rules.md"}
    for report in sorted(DOCS_DIR.glob("*.md")):
        if report.name in hand_written:
            continue
        head = report.read_text()[:400]
        assert "Source module:" in head, (
            f"{report.name} has no provenance banner naming the module that "
            f"generates it"
        )
