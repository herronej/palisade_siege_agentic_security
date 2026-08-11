"""
``palisade/provenance_manifest.py`` -- durable record of untrusted file
uploads.

Files uploaded via the UI are untrusted external input that the agent reads
and turns into artifacts (datacards, plots). This module records each upload
-- name, sha256, content-type, size, time -- in a JSON manifest stored
**outside the agent's sandbox mount** (a sibling of the volume directory, not
under ``/mnt``), so the agent cannot tamper with its own provenance record.

This is the durable form of the "untrusted upload" tag (the piece deferred
out of WB1): G6 surfaces reliance on uploaded data at egress, and audit /
incident review can read the manifest to see exactly what untrusted input
entered a session and when.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def manifest_path_for(volume_root: Path) -> Path:
    """Tamper-resistant manifest location: a sibling of the agent-mounted
    volume directory, so it is **not** reachable from inside ``/mnt``."""
    volume_root = Path(volume_root)
    return volume_root.parent / f"{volume_root.name}.uploads.json"


def load_manifest(manifest_path: Path) -> list[dict]:
    """Return the recorded upload entries, or ``[]`` if none / unreadable."""
    try:
        data = json.loads(Path(manifest_path).read_text())
    except (FileNotFoundError, ValueError, OSError):
        return []
    return data if isinstance(data, list) else []


def record_upload(
    manifest_path: Path,
    name: str,
    content: bytes,
    content_type: str | None = None,
) -> dict:
    """Append (or replace, by ``name``) an untrusted-upload entry and persist
    the manifest. Returns the entry."""
    entry = {
        "name": name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
        "content_type": content_type or "",
        "uploaded_at": datetime.now(tz=timezone.utc).isoformat(),
        "untrusted": True,
    }
    manifest = [e for e in load_manifest(manifest_path) if e.get("name") != name]
    manifest.append(entry)
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return entry


__all__ = ["manifest_path_for", "load_manifest", "record_upload"]
