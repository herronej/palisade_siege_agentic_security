"""
G3 parser / hash helper unit tests.

The R4 migration moved G3's ``rag_search`` gating onto `G3RagCapability`,
and R5 removed the sidecar's ``process_tool_call`` entirely (gate
enforcement now runs via the Agent's capability hooks). The behavioral
G3 coverage lives in ``capabilities/tests/test_g3_capability.py``.

What remains here are the parse/hash helper unit tests, which are
independent of the sidecar and still belong with the G3 wiring tests.
"""

from __future__ import annotations

import hashlib

from palisade.gates.g3_rag import hash_chunk, parse_rag_search_result


def _format_rag_result(chunks: list[tuple[str, str, str]]) -> str:
    """Build a `rag_search`-style formatted response."""
    separator = "\n\n" + "—" * 60 + "\n\n"
    return separator.join(
        f"[{i}] {source}, page {page}\n\n{text}"
        for i, (source, page, text) in enumerate(chunks, 1)
    )


def test_parse_rag_search_result_extracts_chunks() -> None:
    text = _format_rag_result([
        ("paper_a.pdf", "3", "chunk a content"),
        ("paper_b.pdf", "7", "chunk b content"),
    ])
    chunks = parse_rag_search_result(text)
    assert len(chunks) == 2
    assert chunks[0].text == "chunk a content"
    assert chunks[0].source_file == "paper_a.pdf"
    assert chunks[0].page == "3"
    assert chunks[1].text == "chunk b content"
    assert chunks[1].source_file == "paper_b.pdf"
    assert chunks[1].page == "7"


def test_hash_chunk_is_stable_and_sha256() -> None:
    h1 = hash_chunk("chunk text")
    h2 = hash_chunk("chunk text")
    h3 = hash_chunk("different text")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)
    assert h1 == hashlib.sha256("chunk text".encode("utf-8")).hexdigest()
