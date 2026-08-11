"""
Tests for the PALISADE G3 corpus-integrity helpers and the pin CLI.

The gate-level manifest check (match -> allow, mismatch -> SEV2 deny) is
covered in test_g3_fast.py; these tests cover the *producer* side: hashing
a real ChromaDB store, resolving the served path, and the operator CLI
writing a policy file the sidecar's loader can read back.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from palisade import corpus_integrity as ci
from palisade import pin_corpus
from palisade.gates.g3_rag import KB_POLICY_FILENAME, load_kb_policy


def _make_store(path: Path, docs: list[str]) -> None:
    """Create a `text_chunks` ChromaDB collection at `path` with `docs`.

    Explicit dummy embeddings are passed so Chroma never invokes its
    default (model-downloading) embedding function — these tests stay
    hermetic and fast, and the hash only reads documents anyway.
    """
    import chromadb

    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    col = client.get_or_create_collection(
        "text_chunks", metadata={"hnsw:space": "cosine"}
    )
    col.add(
        ids=[f"c{i}" for i in range(len(docs))],
        documents=list(docs),
        embeddings=[[float(i + 1), 1.0, 0.0, 0.0] for i in range(len(docs))],
    )


# ---------------------------------------------------------------- aggregate

def test_aggregate_is_order_independent() -> None:
    assert ci.aggregate_chunk_hashes(["a", "b", "c"]) == ci.aggregate_chunk_hashes(
        ["c", "a", "b"]
    )


def test_aggregate_changes_when_a_chunk_changes() -> None:
    # one byte of poisoning must move the hash
    assert ci.aggregate_chunk_hashes(["alpha", "beta"]) != ci.aggregate_chunk_hashes(
        ["alpha", "beta!"]
    )


# ------------------------------------------------------------ compute_corpus_hash

def test_compute_corpus_hash_matches_aggregate(tmp_path: Path) -> None:
    docs = ["one", "two", "three"]
    store = tmp_path / "kb" / "rag_db"
    _make_store(store, docs)
    corpus_hash, n_chunks = ci.compute_corpus_hash(store)
    assert n_chunks == 3
    assert corpus_hash == ci.aggregate_chunk_hashes(docs)


def test_compute_corpus_hash_missing_store_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ci.compute_corpus_hash(tmp_path / "not-a-store")


# ----------------------------------------------------------- resolve_kb_chroma_path

def test_resolve_prefers_rag_db_over_slug_dir(tmp_path: Path) -> None:
    kb = tmp_path / "molten"
    (kb / "rag_db").mkdir(parents=True)
    (kb / "rag_db" / "chroma.sqlite3").touch()
    (kb / "chroma.sqlite3").touch()  # shadowed slug-dir store also present
    assert ci.resolve_kb_chroma_path(tmp_path, "molten") == kb / "rag_db"


def test_resolve_falls_back_to_slug_dir(tmp_path: Path) -> None:
    kb = tmp_path / "molten"
    kb.mkdir()
    (kb / "chroma.sqlite3").touch()
    assert ci.resolve_kb_chroma_path(tmp_path, "molten") == kb


def test_resolve_none_when_absent(tmp_path: Path) -> None:
    assert ci.resolve_kb_chroma_path(tmp_path, "ghost") is None


# ----------------------------------------------------------- live_hashes_for_pinned

def test_live_hashes_computes_present_skips_missing(tmp_path: Path) -> None:
    _make_store(tmp_path / "present" / "rag_db", ["x", "y"])
    out = ci.live_hashes_for_pinned(tmp_path, ["present", "missing"])
    assert set(out) == {"present"}  # missing KB is skipped, not fatal
    assert out["present"] == ci.aggregate_chunk_hashes(["x", "y"])


# ---------------------------------------------------------------------- pin CLI

def test_pin_cli_write_creates_loadable_policy(tmp_path: Path) -> None:
    kbdir = tmp_path / "kbs"
    _make_store(kbdir / "molten-salt-papers" / "rag_db", ["chunk a", "chunk b"])
    contracts = tmp_path / "contracts"

    rc = pin_corpus.main([
        "--kb", "molten-salt-papers",
        "--knowledge-bases-dir", str(kbdir),
        "--contracts-dir", str(contracts),
        "--write",
    ])
    assert rc == 0

    policy_file = contracts / KB_POLICY_FILENAME
    assert policy_file.is_file()
    # The sidecar reads the file via load_kb_policy — it must round-trip.
    policy = load_kb_policy(policy_file)
    assert policy.corpus_manifests["molten-salt-papers"] == ci.aggregate_chunk_hashes(
        ["chunk a", "chunk b"]
    )


def test_pin_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    kbdir = tmp_path / "kbs"
    _make_store(kbdir / "molten-salt-papers" / "rag_db", ["a"])
    contracts = tmp_path / "contracts"

    rc = pin_corpus.main([
        "--kb", "molten-salt-papers",
        "--knowledge-bases-dir", str(kbdir),
        "--contracts-dir", str(contracts),
    ])
    assert rc == 0
    assert not (contracts / KB_POLICY_FILENAME).exists()


def test_pin_cli_unknown_kb_errors(tmp_path: Path) -> None:
    rc = pin_corpus.main([
        "--kb", "ghost",
        "--knowledge-bases-dir", str(tmp_path),
        "--contracts-dir", str(tmp_path / "contracts"),
        "--write",
    ])
    assert rc == 2  # no store -> non-zero, nothing written
    assert not (tmp_path / "contracts" / KB_POLICY_FILENAME).exists()
