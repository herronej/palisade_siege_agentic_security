"""
Tests for the PALISADE state + re-auth API endpoints.

Acceptance criteria pinned here:

  * ``GET  /projects/{name}/palisade/state`` returns per-capability
    scores and tiers for the caller's session;
  * ``POST /projects/{name}/palisade/reauth`` clears sticky lock-in;
  * both endpoints are documented in the OpenAPI schema when PALISADE
    is enabled, and absent entirely when ``enabled=false``;
  * the re-auth endpoint reuses the existing project authorization (a
    403 from `get_project_by_name` propagates).

The endpoints read/mutate the per-session `TrustScorer` held on the
project agent. These tests stand in a fake agent pool so they exercise
the HTTP layer + authorization without a database or a live agent.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from vista_backend.api import palisade as palisade_api
from vista_backend.api.palisade import router as palisade_router
from vista_backend.api.auth import get_user
from palisade.config import PalisadeSettings
from vista_backend.db.db import _get_session
from palisade.host import HostProject
from vista_backend.services import project as project_service
from palisade.trust import TrustScorer

PROJECT_ID = uuid.uuid4()
USER_ID = uuid.uuid4()
STATE_URL = "/projects/demo/palisade/state"
REAUTH_URL = "/projects/demo/palisade/reauth"


def _project() -> HostProject:
    return HostProject(
        id=PROJECT_ID, name="demo", description=None, system_prompt=None,
        skills=[], knowledge_bases=[], tools=[], usage_limits={},
    )


class _FakePool:
    """Stand-in for `project_agent_pool` keyed by (project_id, user_id)."""

    def __init__(self, scorer: TrustScorer | None = None) -> None:
        self._d: dict[tuple, object] = {}
        if scorer is not None:
            agent = SimpleNamespace(sidecar=SimpleNamespace(trust_scorer=scorer))
            self._d[(PROJECT_ID, USER_ID)] = agent

    def keys(self):
        return self._d.keys()

    @asynccontextmanager
    async def get(self, key):
        yield self._d[key]


def _make_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scorer: TrustScorer | None = None,
    enabled: bool = True,
    project_error: HTTPException | None = None,
) -> TestClient:
    """Build an app with only the PALISADE router (conditionally
    mounted, mirroring `api.api`), with auth/session/project/pool stubbed
    out."""
    app = FastAPI()
    if enabled:
        app.include_router(palisade_router)

    app.dependency_overrides[get_user] = lambda: SimpleNamespace(id=USER_ID)
    app.dependency_overrides[_get_session] = lambda: None

    async def _fake_get_project(session, name, user):
        if project_error is not None:
            raise project_error
        return _project()

    monkeypatch.setattr(project_service, "get_project_by_name", _fake_get_project)
    monkeypatch.setattr(palisade_api, "project_agent_pool", _FakePool(scorer))
    return TestClient(app)


# -----------------------------------------------------------------
# GET /state
# -----------------------------------------------------------------


def test_state_reports_per_capability_scores_and_tiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = TrustScorer(PalisadeSettings(enabled=True))
    scorer.record_violation(1, capability_kind="G5", high_stakes=True)
    scorer.record_clean_call("G2")

    client = _make_client(monkeypatch, scorer=scorer)
    resp = client.get(STATE_URL)
    assert resp.status_code == 200
    body = resp.json()

    assert 0.0 <= body["score"] <= 1.0
    assert body["tier"] == body["capabilities"]["G5"]["tier"]  # worst capability
    g5 = body["capabilities"]["G5"]
    assert g5["sticky"] is True
    assert g5["floor"] is not None
    assert g5["tier"] != "normal"
    assert body["capabilities"]["G2"]["sticky"] is False


def test_state_without_live_session_reports_initial_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No agent in the pool -> the fresh (all-NORMAL) state, no agent built."""
    client = _make_client(monkeypatch, scorer=None)
    resp = client.get(STATE_URL)
    assert resp.status_code == 200
    body = resp.json()
    assert body["tier"] == "normal"
    assert body["capabilities"] == {}
    assert body["sticky_denied"] == []


# -----------------------------------------------------------------
# POST /reauth
# -----------------------------------------------------------------


def test_reauth_clears_sticky_lock_in(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = TrustScorer(PalisadeSettings(enabled=True))
    scorer.record_violation(1, capability_kind="G5", high_stakes=True)
    scorer.mark_sticky_denied("g5:above_ceiling_resource")
    assert scorer.current_tier_for("G5").value != "normal"

    client = _make_client(monkeypatch, scorer=scorer)
    resp = client.post(REAUTH_URL)
    assert resp.status_code == 200
    assert resp.json()["unlocked_capabilities"] == ["G5"]

    # The lock is actually lifted on the live scorer.
    assert scorer.current_tier_for("G5").value == "normal"
    assert scorer.sticky_denied == ()


def test_reauth_without_live_session_is_a_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, scorer=None)
    resp = client.post(REAUTH_URL)
    assert resp.status_code == 200
    assert resp.json()["unlocked_capabilities"] == []


def test_reauth_requires_project_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC: a 403 from the existing project-access check propagates -- the
    endpoint does not introduce a separate auth path."""
    client = _make_client(
        monkeypatch,
        scorer=TrustScorer(PalisadeSettings(enabled=True)),
        project_error=HTTPException(status_code=403, detail="no access"),
    )
    assert client.post(REAUTH_URL).status_code == 403
    assert client.get(STATE_URL).status_code == 403


# -----------------------------------------------------------------
# Enable-flag gating + OpenAPI documentation
# -----------------------------------------------------------------


def test_endpoints_documented_in_openapi_when_enabled() -> None:
    # `api.api` includes the router when `settings.palisade.enabled`.
    app = FastAPI()
    app.include_router(palisade_router)
    paths = app.openapi()["paths"]
    assert "/projects/{project_name}/palisade/state" in paths
    assert "/projects/{project_name}/palisade/reauth" in paths
    schemas = app.openapi()["components"]["schemas"]
    assert "PalisadeStateResponse" in schemas
    assert "PalisadeReauthResponse" in schemas


def test_endpoints_absent_when_disabled() -> None:
    # When disabled, `api.api` simply does not mount the router.
    app = FastAPI()
    paths = app.openapi().get("paths", {})
    assert "/projects/{project_name}/palisade/state" not in paths
    assert "/projects/{project_name}/palisade/reauth" not in paths

    # And a request 404s rather than returning state.
    app.dependency_overrides[get_user] = lambda: SimpleNamespace(id=USER_ID)
    app.dependency_overrides[_get_session] = lambda: None
    client = TestClient(app)
    assert client.get(STATE_URL).status_code == 404
    assert client.post(REAUTH_URL).status_code == 404


def test_router_object_carries_the_palisade_tag() -> None:
    # Sanity: the routes hang off the dedicated tagged router.
    route_paths = {r.path for r in palisade_router.routes}
    assert "/projects/{project_name}/palisade/state" in route_paths
    assert "/projects/{project_name}/palisade/reauth" in route_paths
