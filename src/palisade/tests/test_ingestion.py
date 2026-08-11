"""
Upload ingestion gate tests (WB1): executable/script/archive/active-content
deny-list + JSON data-file sanity, for files entering via the UI upload
endpoint.
"""

from __future__ import annotations

import pytest

from palisade.ingestion import upload_violation


@pytest.mark.parametrize(
    "name",
    [
        # data the assistant legitimately consumes -- open-ended, so a
        # deny-list lets these through where an allow-list would not.
        "Molten_Salt_Thermophysical_Properties.json",
        "data.csv",
        "table.tsv",
        "results.xlsx",   # plain (non-macro) spreadsheet
        "dataset.h5",
        "raw.dat",
        "big.parquet",
        "notes.md",
        "readme.txt",
        "ref.pdf",
        "plot.png",
        "fig.jpeg",
        "cfg.yaml",
        "extensionless_data",  # no suffix -> not blocked
    ],
)
def test_allowed_types_pass(name):
    content = b"{}" if name.endswith(".json") else b"data"
    assert upload_violation(name, content) is None


@pytest.mark.parametrize(
    "name",
    [
        "evil.exe",
        "macro.xlsm",       # VBA-enabled office
        "payload.zip",
        "archive.tar.gz",   # Path.suffix is ".gz"
        "script.sh",
        "run.py",
        "page.html",
        "fig.svg",          # scriptable markup
        "lib.so",
        "model.pkl",        # pickle deserialization
        "installer.msi",
        "app.js",
    ],
)
def test_disallowed_types_rejected(name):
    assert upload_violation(name, b"data") is not None


def test_malformed_json_rejected():
    v = upload_violation("data.json", b"{not valid json,,,}")
    assert v is not None and "JSON" in v


def test_valid_json_object_passes():
    assert upload_violation("data.json", b'{"salts": [{"name": "FLiBe"}]}') is None


def test_valid_json_array_passes():
    assert upload_violation("data.json", b"[1, 2, 3]") is None
