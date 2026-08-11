"""
``palisade/ingestion.py`` -- validation for files entering via the UI
upload endpoint.

Files uploaded through the UI land in the agent sandbox at
``/mnt/data/uploads`` and become inputs the agent reads and turns into
authoritative artifacts (datacards, plots, reports). This is an **ingress
surface the per-turn gates never see** -- G1's attached-file MIME/size
policy is for *chat* attachments, which this product does not use (the
capability hands G1 an empty ``attached_files`` list), so an uploaded file
reaches the agent entirely ungated.

``upload_violation`` is the ingestion checkpoint: a **deny-list** of
executable / script / archive / macro / active-content types, plus a light
structural check on JSON data files, run on the bytes before they are
written to the uploads directory. The upload endpoint (``api/uploads.py``)
rejects a violation with HTTP 400.

Why a deny-list, not an allow-list: a molten-salt assistant's legitimate
uploads are open-ended scientific data (``.json`` / ``.csv`` / ``.xlsx`` /
``.h5`` / ``.nc`` / ``.dat`` / ``.parquet`` …), documents, and figures --
enumerating them in an allow-list would reject real formats and break the
workflow. The threat is narrow and enumerable: content that executes or
that exploits a parser. So we block *that* and let data through. (Malicious
*data* -- wrong numbers -- is a separate concern, caught downstream by the
egress correctness contracts, not at ingestion.)

Pure and deterministic so it unit-tests without a server.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Extensions refused at ingestion. None has a legitimate place as a
#: data / reference upload, and each is a code-execution or parser-exploit
#: risk if it reaches the sandbox.
BLOCKED_EXTENSIONS = frozenset(
    {
        # native executables / libraries / installers
        ".exe", ".dll", ".so", ".dylib", ".bin", ".com", ".msi", ".app",
        ".deb", ".rpm",
        # scripts -- the upload-then-execute path; running code belongs to a
        # reviewed path, not an arbitrary upload
        ".sh", ".bash", ".zsh", ".bat", ".cmd", ".ps1",
        ".py", ".pyc", ".pl", ".rb", ".php", ".jar",
        # archives -- zip-bomb / unpack-traversal / payload carrier
        ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar",
        # macro-enabled office (VBA execution on open)
        ".xlsm", ".docm", ".pptm", ".xlsb", ".dotm",
        # active web / scriptable markup (injection if rendered or run)
        ".html", ".htm", ".xhtml", ".svg", ".mhtml", ".js", ".mjs",
        # arbitrary-code deserialization
        ".pickle", ".pkl", ".joblib",
    }
)


def upload_violation(
    filename: str, content: bytes, content_type: str | None = None
) -> str | None:
    """Return a human-readable rejection reason if the upload should be
    refused, else None.

    - The file extension must not be in :data:`BLOCKED_EXTENSIONS`.
    - A ``.json`` payload must parse as JSON (it is going to be read as a
      data file; a malformed blob masquerading as ``.json`` is rejected at
      the door rather than corrupting a downstream datacard).
    """
    ext = Path(filename or "").suffix.lower()
    if ext in BLOCKED_EXTENSIONS:
        return (
            f"file type {ext!r} is not an allowed upload "
            "(executable / script / archive / macro / active content)"
        )
    if ext == ".json":
        try:
            json.loads(content)
        except (ValueError, UnicodeDecodeError) as exc:
            return f"file {filename!r} is not valid JSON: {exc}"
    return None


__all__ = ["upload_violation", "BLOCKED_EXTENSIONS"]
