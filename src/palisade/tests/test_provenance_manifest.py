"""
WB5: durable upload provenance manifest.
"""

from __future__ import annotations

import hashlib

from palisade.provenance_manifest import load_manifest, manifest_path_for, record_upload


def test_manifest_path_is_outside_volume_mount(tmp_path):
    volume_root = tmp_path / "volumes" / "proj-user"
    volume_root.mkdir(parents=True)
    mpath = manifest_path_for(volume_root)
    # The agent mounts volume_root at /mnt, so the manifest must be a SIBLING
    # of the volume dir (not under it) to be tamper-resistant.
    assert mpath.parent == volume_root.parent
    assert volume_root not in mpath.parents


def test_record_and_load_roundtrip(tmp_path):
    mpath = tmp_path / "m.json"
    entry = record_upload(mpath, "data.json", b'{"a":1}', "application/json")
    assert entry["name"] == "data.json"
    assert entry["sha256"] == hashlib.sha256(b'{"a":1}').hexdigest()
    assert entry["size"] == 7
    assert entry["untrusted"] is True
    loaded = load_manifest(mpath)
    assert len(loaded) == 1 and loaded[0]["name"] == "data.json"


def test_record_replaces_by_name(tmp_path):
    mpath = tmp_path / "m.json"
    record_upload(mpath, "data.json", b"v1")
    record_upload(mpath, "data.json", b"v2longer")
    record_upload(mpath, "other.csv", b"x")
    loaded = load_manifest(mpath)
    names = sorted(e["name"] for e in loaded)
    assert names == ["data.json", "other.csv"]  # data.json replaced, not duped
    d = next(e for e in loaded if e["name"] == "data.json")
    assert d["sha256"] == hashlib.sha256(b"v2longer").hexdigest()  # latest content


def test_load_missing_manifest_is_empty(tmp_path):
    assert load_manifest(tmp_path / "nope.json") == []
