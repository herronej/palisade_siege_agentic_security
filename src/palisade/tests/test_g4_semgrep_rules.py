"""
Validation tests for the PALISADE G4 Semgrep ruleset.

Covers the acceptance criteria of
``Add Semgrep as optional dependency and write PALISADE
ruleset``:

1. ``pip install vista-backend[palisade-g4]`` installs Semgrep --
   verified by parsing ``backend/pyproject.toml`` and asserting the
   ``palisade-g4`` optional-dependency group contains a
   ``semgrep`` constraint.
2. Default install does not install Semgrep -- verified by asserting
   ``semgrep`` is not in ``[project.dependencies]``.
3. At least 6 PALISADE-specific rules covering the six listed
   threat surfaces.
4. Each rule documented in the markdown doc.

The tests don't actually run Semgrep -- the package is optional and
typically absent. They parse rule YAMLs and the markdown doc and
assert structural invariants.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from palisade.paths import PALISADE_DIR, REPO_ROOT


# -----------------------------------------------------------------
# Module-level layout constants
# -----------------------------------------------------------------


PYPROJECT = REPO_ROOT / "pyproject.toml"
RULES_DIR = PALISADE_DIR / "contracts" / "semgrep"
DOC_PATH = REPO_ROOT / "docs" / "palisade" / "g4_semgrep_rules.md"

# Required threat-surface coverage from the work-item AC.
REQUIRED_THREAT_SURFACES: frozenset[str] = frozenset(
    {
        "code-execution",          # subprocess shell metacharacters
        "credential-exfiltration", # credential file reads
        "supply-chain",            # typo-squatted package imports
        "data-exfiltration",       # outbound network / Globus
        "sandbox-escape",          # malicious env-var manipulation
    }
)


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------


def _read_pyproject_text() -> str:
    assert PYPROJECT.exists(), f"missing pyproject.toml at {PYPROJECT}"
    return PYPROJECT.read_text(encoding="utf-8")


def _parse_pyproject() -> dict:
    """
    Parse the TOML file. Python 3.11+ ships ``tomllib`` in the stdlib;
    we use it rather than depending on ``tomli`` to avoid a build-time
    extra.
    """
    import tomllib

    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def _load_rule_files() -> list[tuple[Path, dict]]:
    """
    Read every ``*.yml`` under the rules directory and return a list
    of (path, parsed-YAML) tuples.
    """
    assert RULES_DIR.exists(), f"missing rules directory at {RULES_DIR}"
    out: list[tuple[Path, dict]] = []
    for path in sorted(RULES_DIR.rglob("*.yml")):
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        out.append((path, data))
    return out


def _iter_rules() -> list[tuple[Path, dict]]:
    """Flatten the rules from every YAML file."""
    flat: list[tuple[Path, dict]] = []
    for path, data in _load_rule_files():
        assert isinstance(data, dict), f"{path}: top-level must be a mapping"
        rules = data.get("rules")
        assert isinstance(rules, list), (
            f"{path}: missing top-level `rules:` list"
        )
        for rule in rules:
            assert isinstance(rule, dict), (
                f"{path}: each rule must be a mapping; got {type(rule).__name__}"
            )
            flat.append((path, rule))
    return flat


# -----------------------------------------------------------------
# AC 1 + AC 2: pyproject.toml extras
# -----------------------------------------------------------------


def test_palisade_g4_optional_dependency_declared() -> None:
    """AC1: the `palisade-g4` extra includes semgrep."""
    data = _parse_pyproject()
    extras = data["project"]["optional-dependencies"]
    assert "palisade-g4" in extras, (
        "missing `[palisade-g4]` optional-dependency group in pyproject.toml"
    )
    members = extras["palisade-g4"]
    assert any("semgrep" in m.lower() for m in members), (
        f"`palisade-g4` group must include semgrep; got {members}"
    )


def test_semgrep_not_in_default_dependencies() -> None:
    """AC2: default install does not pull semgrep."""
    data = _parse_pyproject()
    deps = data["project"]["dependencies"]
    assert not any("semgrep" in d.lower() for d in deps), (
        f"semgrep leaked into default `[project.dependencies]`: {deps}"
    )


# -----------------------------------------------------------------
# AC 3: rule count + threat-surface coverage
# -----------------------------------------------------------------


def test_at_least_six_rules_shipped() -> None:
    """AC3: at least 6 PALISADE-specific rules."""
    rules = _iter_rules()
    assert len(rules) >= 6, (
        f"expected at least 6 rules; found {len(rules)} "
        f"in {sorted({str(p) for p, _ in rules})}"
    )


def test_rule_ids_are_unique_and_vista_prefixed() -> None:
    """Each rule has a unique `vista-` prefixed ID."""
    seen: set[str] = set()
    for path, rule in _iter_rules():
        rid = rule.get("id")
        assert isinstance(rid, str), f"{path}: rule missing string `id`"
        assert rid.startswith("vista-"), (
            f"{path}: rule id {rid!r} must start with `vista-` so it "
            f"doesn't collide with community rules"
        )
        assert rid not in seen, f"{path}: duplicate rule id {rid!r}"
        seen.add(rid)


def test_every_rule_has_required_fields() -> None:
    """Each rule has id / message / severity / languages / metadata."""
    required = ("id", "message", "severity", "languages")
    valid_severities = {"ERROR", "WARNING", "INFO"}
    for path, rule in _iter_rules():
        for field in required:
            assert field in rule, (
                f"{path}: rule {rule.get('id')!r} missing required field "
                f"{field!r}"
            )
        assert rule["severity"] in valid_severities, (
            f"{path}: rule {rule['id']!r} severity "
            f"{rule['severity']!r} not in {valid_severities}"
        )
        assert "python" in rule["languages"], (
            f"{path}: rule {rule['id']!r} must target python"
        )
        meta = rule.get("metadata", {})
        assert isinstance(meta, dict), (
            f"{path}: rule {rule['id']!r} metadata must be a mapping"
        )
        assert "vista-threat-surface" in meta, (
            f"{path}: rule {rule['id']!r} missing "
            f"`metadata.vista-threat-surface`"
        )
        assert "vista-taxonomy" in meta, (
            f"{path}: rule {rule['id']!r} missing "
            f"`metadata.vista-taxonomy`"
        )


def test_every_rule_has_pattern_form() -> None:
    """Each rule defines at least one of pattern / patterns / pattern-either."""
    pattern_keys = {
        "pattern",
        "patterns",
        "pattern-either",
        "pattern-regex",
        "pattern-not",
    }
    for path, rule in _iter_rules():
        present = pattern_keys & set(rule)
        assert present, (
            f"{path}: rule {rule['id']!r} has no pattern* directive "
            f"(saw keys: {sorted(rule)})"
        )


def test_threat_surface_coverage() -> None:
    """AC3 coverage: every required threat surface is named by at
    least one rule's metadata."""
    surfaces = {
        rule["metadata"]["vista-threat-surface"]
        for _, rule in _iter_rules()
    }
    missing = REQUIRED_THREAT_SURFACES - surfaces
    assert not missing, (
        f"the following required threat surfaces are not covered by "
        f"any rule: {sorted(missing)}. Saw: {sorted(surfaces)}"
    )


@pytest.mark.parametrize(
    "expected_id",
    [
        "vista-subprocess-shell-true",        # shell metacharacters
        "vista-credential-file-read",         # credential file reads
        "vista-typo-squat-import",            # typo-squatted imports
        "vista-globus-untrusted-endpoint",    # Globus to unknown endpoint
        "vista-env-var-preload-hijack",       # malicious env-var manipulation
        "vista-direct-outbound-network",      # outbound network
    ],
)
def test_canonical_rules_are_present(expected_id: str) -> None:
    """The 6 AC-listed canonical rules ship by ID."""
    ids = {rule["id"] for _, rule in _iter_rules()}
    assert expected_id in ids, (
        f"missing canonical rule {expected_id!r}; saw: {sorted(ids)}"
    )


# -----------------------------------------------------------------
# AC 4: markdown documentation
# -----------------------------------------------------------------


def test_documentation_exists() -> None:
    """AC4: the threat-surface mapping doc exists."""
    assert DOC_PATH.exists(), (
        f"missing documentation at {DOC_PATH}"
    )


def test_documentation_references_every_rule() -> None:
    """AC4: every rule ID appears in the markdown doc."""
    doc_text = DOC_PATH.read_text(encoding="utf-8")
    missing: list[str] = []
    for _, rule in _iter_rules():
        rid = rule["id"]
        if rid not in doc_text:
            missing.append(rid)
    assert not missing, (
        f"the following rule IDs are not mentioned in {DOC_PATH.name}: "
        f"{missing}"
    )


def test_documentation_mentions_each_threat_surface() -> None:
    """Each required threat surface is named in the doc."""
    doc_text = DOC_PATH.read_text(encoding="utf-8")
    for surface in REQUIRED_THREAT_SURFACES:
        assert surface in doc_text, (
            f"threat surface {surface!r} not mentioned in {DOC_PATH.name}"
        )


def test_documentation_mentions_install_command() -> None:
    """Operators reading the doc can find the install command."""
    doc_text = DOC_PATH.read_text(encoding="utf-8")
    assert "pip install vista-backend[palisade-g4]" in doc_text


# -----------------------------------------------------------------
# YAML structural pin: every file has a top-level `rules:` list
# -----------------------------------------------------------------


def test_every_yaml_file_has_top_level_rules_list() -> None:
    """Pin the Semgrep YAML shape so an editor mistake fires here."""
    files = _load_rule_files()
    assert files, "no rule YAML files found"
    for path, data in files:
        assert isinstance(data, dict)
        assert "rules" in data, f"{path}: missing top-level `rules:` key"
        assert isinstance(data["rules"], list)
        assert data["rules"], f"{path}: empty `rules:` list"


# -----------------------------------------------------------------
# Negative tests: known anti-patterns to keep in the metadata
# -----------------------------------------------------------------


_TAXONOMY_RE = re.compile(r"^\d+(\.\d+)*$")


def test_taxonomy_metadata_is_a_section_reference() -> None:
    """`vista-taxonomy` values look like dotted section refs."""
    for path, rule in _iter_rules():
        tx = str(rule["metadata"]["vista-taxonomy"])
        assert _TAXONOMY_RE.match(tx), (
            f"{path}: rule {rule['id']!r} taxonomy {tx!r} is not a "
            f"dotted section reference"
        )
