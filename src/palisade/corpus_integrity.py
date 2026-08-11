"""
PALISADE G3 corpus-integrity helper.

Computes a deterministic, order-independent hash over a Knowledge Base's
retrievable chunk content — the ChromaDB ``text_chunks`` documents. That
hash is the operator-pinned integrity check G3 enforces:

  * the **operator** pins it once, against a trusted corpus, into
    ``<contracts_dir>/g3_kb_policy.json`` (``corpus_manifests[slug]``) via
    the ``pin_corpus`` CLI; and
  * the **sidecar** (and the siege eval harness) recompute it live
    and bind it through ``G3RagGate(kb_chunk_hashes=...)``.

When the two diverge, the served corpus no longer matches what the
operator pinned — poisoning, tampering, or an unreviewed re-index — and
G3 denies the ``rag_search`` SEV2 ([gates/g3_rag.py] step 4).

The hash covers chunk *text* only (the surface ``rag_search`` actually
feeds the model), not embeddings or citation metadata, so it is stable
across re-embedding and citation re-extraction of identical text — two
stores built from the same PDFs hash identically even if their HNSW
graphs and citation passes differ.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

#: ChromaDB collection holding the retrievable chunk text. Matches the
#: name build_rag.py and vista_mcp_server.rag_mcp use.
CHUNK_COLLECTION = "text_chunks"


def hash_chunk(text: str) -> str:
    """
    Per-chunk content hash. Identical to ``gates.g3_rag.hash_chunk``
    (sha256 of the UTF-8 text); inlined here so this module — and the
    operator CLI that imports it — stays free of the gate's heavier
    import graph.
    """
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def aggregate_chunk_hashes(chunk_texts) -> str:
    """
    Order-independent aggregate over a corpus's per-chunk hashes.

    sha256 over the newline-joined, **sorted** per-chunk hashes. Sorting
    makes the result independent of the order ChromaDB returns documents
    in (segment/HNSW order is not stable across rebuilds), so an
    untampered corpus hashes identically however it was indexed.
    """
    per = sorted(hash_chunk(t) for t in chunk_texts)
    return hashlib.sha256("\n".join(per).encode("utf-8")).hexdigest()


def compute_corpus_hash(chroma_path: str | Path) -> tuple[str, int]:
    """
    Open the ChromaDB store at ``chroma_path`` and return
    ``(corpus_hash, n_chunks)`` over its ``text_chunks`` documents.

    Raises ``FileNotFoundError`` when the path is not a Chroma store and
    ``RuntimeError`` when the collection can't be read. Bind-path callers
    (sidecar / eval) catch both and degrade to "no live hash" — a missing
    or unreadable corpus must not crash gate construction.
    """
    path = Path(chroma_path)
    if not (path / "chroma.sqlite3").is_file():
        raise FileNotFoundError(
            f"No ChromaDB store at {path} (expected chroma.sqlite3)."
        )

    # Lazy import: chromadb is a heavy dependency and this module is also
    # imported by the lightweight pin CLI before it needs a client.
    import chromadb

    client = chromadb.PersistentClient(path=str(path))
    try:
        collection = client.get_or_create_collection(
            CHUNK_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        raw = collection.get(include=["documents"])
    except Exception as exc:  # noqa: BLE001 — surface as RuntimeError
        raise RuntimeError(
            f"Could not read {CHUNK_COLLECTION!r} at {path}: {exc}"
        ) from exc

    documents = raw.get("documents") or []
    return aggregate_chunk_hashes(documents), len(documents)


def resolve_kb_chroma_path(
    knowledge_bases_dir: str | Path, slug: str
) -> Path | None:
    """
    Resolve a ``kb_slug`` to the ChromaDB store the MCP server actually
    serves, mirroring ``vista_mcp_server.rag_mcp._discover_kb_paths``:

      1. ``<knowledge_bases_dir>/<slug>/rag_db/`` — canonical per-KB layout
      2. ``<knowledge_bases_dir>/<slug>/``        — legacy in-place store

    Returns ``None`` when neither looks like a Chroma store. The order
    matters: G3 must hash the *same* store the agent queries, so the pin
    and the live check agree. Keep this in lockstep with the MCP server's
    discovery if that ever changes.
    """
    base = Path(knowledge_bases_dir) / slug
    for candidate in (base / "rag_db", base):
        if (candidate / "chroma.sqlite3").is_file():
            return candidate
    return None


def live_hashes_for_pinned(
    knowledge_bases_dir: str | Path,
    pinned_slugs,
    *,
    log=None,
) -> dict[str, str]:
    """
    Recompute live corpus hashes for ``pinned_slugs`` from their served
    ChromaDB stores. Returns ``{slug: corpus_hash}`` for every slug whose
    store could be located and read; a slug whose store is missing or
    unreadable is **skipped** (logged via ``log`` when supplied) rather
    than raising — gate construction must survive a missing corpus.

    Shared by the sidecar (production bind) and the siege eval so
    both compute the live side identically.
    """
    hashes: dict[str, str] = {}
    for slug in pinned_slugs:
        store = resolve_kb_chroma_path(knowledge_bases_dir, slug)
        if store is None:
            if log is not None:
                log.warning(
                    "G3 integrity: pinned KB %r has no served ChromaDB store "
                    "under %s; live integrity check disabled for it.",
                    slug, knowledge_bases_dir,
                )
            continue
        try:
            corpus_hash, n_chunks = compute_corpus_hash(store)
        except (FileNotFoundError, RuntimeError) as exc:
            if log is not None:
                log.warning(
                    "G3 integrity: could not hash pinned KB %r at %s (%s); "
                    "live integrity check disabled for it.",
                    slug, store, exc,
                )
            continue
        hashes[slug] = corpus_hash
        if log is not None:
            log.info(
                "G3 integrity: bound live hash for KB %r (%d chunks) from %s",
                slug, n_chunks, store,
            )
    return hashes
