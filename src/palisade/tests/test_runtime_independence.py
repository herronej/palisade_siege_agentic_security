"""PALISADE must stay installable, and shippable, on its own.

Two properties hold this repository's split together, and both fail silently
if broken — the suite still passes from a checkout, where everything is on the
path and a repository is always around, and the breakage only shows up in a
host application's production environment.

1. The runtime never imports the benchmark. If it did, a host installing
   `palisade` would need `siege` and its 205 attack instances too.
2. The runtime imports without a checkout. `palisade` lands in a host's
   site-packages, where there is no `pyproject.toml` above it.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from palisade.paths import PALISADE_DIR

RUNTIME_FILES = sorted(
    p for p in PALISADE_DIR.rglob("*.py") if "tests" not in p.parts
)


def test_there_are_runtime_files_to_check() -> None:
    """Guard the guard: an empty scan must not read as a pass."""
    assert len(RUNTIME_FILES) > 40, f"only found {len(RUNTIME_FILES)} runtime modules"


def _guarded_import_nodes(tree: ast.AST) -> set[int]:
    """Import nodes inside a ``try`` whose handler catches ImportError.

    An optional dependency reached this way degrades when absent, which is a
    legitimate pattern; an unguarded one crashes a host at request time.
    """
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        catches_import_error = any(
            h.type is None
            or (isinstance(h.type, ast.Name) and h.type.id in {"ImportError", "Exception"})
            or (
                isinstance(h.type, ast.Tuple)
                and any(
                    isinstance(e, ast.Name) and e.id in {"ImportError", "Exception"}
                    for e in h.type.elts
                )
            )
            for h in node.handlers
        )
        if not catches_import_error:
            continue
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    guarded.add(id(sub))
    return guarded


@pytest.mark.parametrize("path", RUNTIME_FILES, ids=lambda p: p.name)
def test_runtime_does_not_import_the_benchmark(path: Path) -> None:
    """`palisade` must not *require* `siege`, at module scope or inside a
    function. The dependency runs the other way: the benchmark measures the
    sidecar. An import guarded by ``except ImportError`` is allowed, because a
    deployment without the benchmark then degrades instead of failing.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    guarded = _guarded_import_nodes(tree)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.split(".")[0] == "siege"]
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "siege":
                offenders.append(node.module or "")
    assert not offenders, (
        f"{path.name} requires {offenders} — the runtime sidecar must not depend "
        f"on the benchmark, or a host installing palisade pulls in the corpus. "
        f"If it is genuinely optional, guard it with `except ImportError` and "
        f"degrade."
    )


@pytest.mark.parametrize("path", RUNTIME_FILES, ids=lambda p: p.name)
def test_runtime_does_not_import_the_host_application(path: Path) -> None:
    """Nor on its reference deployment. PALISADE is written against the
    `palisade.host` protocols, not against VISTA."""
    text = path.read_text(encoding="utf-8")
    assert "import vista_backend" not in text and "from vista_backend" not in text, (
        f"{path.name} imports the reference host application; the sidecar must "
        f"depend only on the structural contract in palisade.host"
    )


def test_runtime_imports_without_a_checkout(tmp_path: Path) -> None:
    """Import every runtime module from a directory with no repository above it.

    `palisade.paths` once raised at import when it could not find a checkout,
    which made the package unusable as a dependency while every test here
    still passed.
    """
    modules = [
        "palisade",
        "palisade.config",
        "palisade.sidecar",
        "palisade.host",
        "palisade.paths",
        "palisade.pin_corpus",
        "palisade.trust",
        "palisade.incidents",
        "palisade.provenance",
    ]
    script = (
        "import importlib, sys\n"
        f"for m in {modules!r}:\n"
        "    importlib.import_module(m)\n"
        "from palisade.paths import find_repo_root\n"
        "print('OK', find_repo_root() is None)\n"
    )
    # cwd is an empty tmp dir, so nothing resolves a repo root by accident.
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"runtime modules failed to import outside a checkout:\n{proc.stderr}"
    )
