"""
Operator CLI: pin a Knowledge Base's corpus hash for PALISADE G3.

Computes the integrity hash of a KB's *served* ChromaDB store (see
``corpus_integrity.compute_corpus_hash``) and records it in the G3
corpus-policy file as ``corpus_manifests[<slug>]``. Once pinned, the
sidecar (and the siege eval) recompute the live hash and deny any
``rag_search`` whose corpus no longer matches the pin — the corpus was
re-indexed with different content, poisoned, or swapped.

Run it once against a corpus you trust — e.g. right after a reviewed
re-index — then commit / deploy the updated policy file::

    # dry run: print the hash, leave the policy file untouched
    python -m palisade.pin_corpus --kb molten-salt-papers

    # write corpus_manifests["molten-salt-papers"] into g3_kb_policy.json
    python -m palisade.pin_corpus --kb molten-salt-papers --write

Paths default to the standard repo layout (the same locations the backend
resolves to when launched from ``backend/``) and can be overridden for
non-standard deployments via ``--knowledge-bases-dir`` / ``--contracts-dir``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from palisade.corpus_integrity import compute_corpus_hash, resolve_kb_chroma_path
from palisade.gates.g3_rag import KB_POLICY_FILENAME, KB_POLICY_VERSION
from palisade.config import PalisadeSettings
from palisade.paths import find_repo_root

# Defaults come from the deployment's own settings (`PALISADE_CONTRACTS_DIR`,
# `PALISADE_KNOWLEDGE_BASES_DIR`), so this CLI works when PALISADE is installed
# as a library into a host application and there is no PALISADE checkout to
# measure paths against. In a checkout the repo root is a better anchor than
# the cwd, so it is preferred when one exists; both are only argparse
# defaults, and `--knowledge-bases-dir` / `--contracts-dir` override either.
_SETTINGS = PalisadeSettings()
_ANCHOR = find_repo_root() or Path.cwd()


def _resolve(configured: str) -> Path:
    p = Path(configured)
    return p if p.is_absolute() else (_ANCHOR / p).resolve()


_DEFAULT_KB_DIR = _resolve(_SETTINGS.knowledge_bases_dir)
_DEFAULT_CONTRACTS_DIR = _resolve(_SETTINGS.contracts_dir)


def _load_policy(policy_path: Path) -> dict:
    """Load the existing policy file, or return a fresh v1 skeleton."""
    if policy_path.is_file():
        data = json.loads(policy_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SystemExit(f"{policy_path} is not a JSON object.")
        data.setdefault("version", KB_POLICY_VERSION)
        data.setdefault("kb_sensitivity_tiers", {})
        data.setdefault("corpus_manifests", {})
        return data
    return {
        "version": KB_POLICY_VERSION,
        "kb_sensitivity_tiers": {},
        "corpus_manifests": {},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m palisade.pin_corpus",
        description="Pin a KB's corpus hash into the G3 corpus-policy file.",
    )
    parser.add_argument(
        "--kb", required=True,
        help="Knowledge Base slug, e.g. molten-salt-papers",
    )
    parser.add_argument(
        "--knowledge-bases-dir", type=Path, default=_DEFAULT_KB_DIR,
        help=f"KB root (default: {_DEFAULT_KB_DIR})",
    )
    parser.add_argument(
        "--contracts-dir", type=Path, default=_DEFAULT_CONTRACTS_DIR,
        help=f"PALISADE contracts dir holding {KB_POLICY_FILENAME} "
             f"(default: {_DEFAULT_CONTRACTS_DIR})",
    )
    parser.add_argument(
        "--write", action="store_true",
        help="Write the pin into the policy file (default: dry run).",
    )
    args = parser.parse_args(argv)

    store = resolve_kb_chroma_path(args.knowledge_bases_dir, args.kb)
    if store is None:
        print(
            f"ERROR: no ChromaDB store for KB {args.kb!r} under "
            f"{args.knowledge_bases_dir} (looked for <slug>/rag_db and <slug>).",
            file=sys.stderr,
        )
        return 2
    try:
        corpus_hash, n_chunks = compute_corpus_hash(store)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    policy_path = args.contracts_dir / KB_POLICY_FILENAME
    print(f"KB slug      : {args.kb}")
    print(f"Served store : {store}")
    print(f"Chunks       : {n_chunks}")
    print(f"Corpus hash  : {corpus_hash}")
    print(f"Policy file  : {policy_path}")

    if not args.write:
        print("\n(dry run — re-run with --write to pin this hash)")
        return 0

    data = _load_policy(policy_path)
    manifests = data["corpus_manifests"]
    prev = manifests.get(args.kb)
    manifests[args.kb] = corpus_hash
    args.contracts_dir.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if prev == corpus_hash:
        print("\nPin unchanged (corpus already matches the existing pin).")
    elif prev:
        print(f"\nUPDATED pin (previous: {prev}).")
    else:
        print("\nPinned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
