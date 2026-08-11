"""Tests that exercise PALISADE *inside* its reference host application.

Everything here imports ``vista_backend`` -- the agent loop, the control-plane
router, the approval elicitation round-trip. They verify that the sidecar's
hooks fire correctly against a real PydanticAI agent rather than against the
replayed action sequences the rest of the suite uses, so they cannot run from
the standalone artifact checkout.

They are skipped, not deleted, so that a VISTA deployment can run the full
suite with no change: install PALISADE beside ``vista_backend`` and these
collect normally.

Nothing the paper reports depends on this directory. The SIEGE corpus is
replayed through the unmodified gate stack (Section V-A), which needs no host.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "vista_backend",
    reason=(
        "host-integration tests require the VISTA reference deployment; "
        "the standalone PALISADE/SIEGE artifact does not vendor it"
    ),
)
