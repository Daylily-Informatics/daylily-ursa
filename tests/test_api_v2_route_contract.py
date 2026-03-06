"""Contract tests enforcing v2-only API routing."""

from unittest.mock import MagicMock

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from daylily_ursa.workset_api import create_app
from daylily_ursa.workset_state_db import WorksetStateDB


def _create_test_app():
    state_db = MagicMock(spec=WorksetStateDB)
    state_db.get_queue_depth.return_value = {"ready": 0, "in_progress": 0, "completed": 0}
    return create_app(state_db=state_db, enable_auth=False)


def _collect_api_paths(app):
    api_like_prefixes = ("/api", "/worksets")
    api_like_exact_paths = {"/queue/stats", "/scheduler/stats"}
    return sorted(
        {
            route.path
            for route in app.routes
            if isinstance(route, APIRoute)
            and (
                route.path.startswith(api_like_prefixes)
                or route.path in api_like_exact_paths
            )
        }
    )


def test_all_api_routes_are_v2_versioned():
    app = _create_test_app()
    api_paths = _collect_api_paths(app)
    assert api_paths, "Expected at least one API route to be registered."

    non_v2_paths = [path for path in api_paths if not path.startswith("/api/v2/")]
    assert not non_v2_paths, (
        "Found non-v2 API routes that violate the /api/v2 contract: "
        f"{non_v2_paths}"
    )


def test_legacy_api_paths_return_not_found():
    app = _create_test_app()
    client = TestClient(app, base_url="https://testserver")

    legacy_calls = [
        ("get", "/api/v1/auth/tokens", {}),
        ("patch", "/api/v1/customers/cust-001", {"json": {"customer_name": "Acme"}}),
        ("patch", "/api/files/file-001", {"json": {"tags": []}}),
        ("get", "/worksets/ws-123", {}),
        ("get", "/queue/stats", {}),
        ("get", "/scheduler/stats", {}),
    ]

    for method_name, path, kwargs in legacy_calls:
        method = getattr(client, method_name)
        response = method(path, **kwargs)
        assert response.status_code == 404, (
            f"Legacy path should be removed after v2 cutover: {method_name.upper()} {path} "
            f"returned {response.status_code}"
        )
