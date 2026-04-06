from __future__ import annotations

import ast
import re
from pathlib import Path
from unittest.mock import patch

from daylib_ursa.config import Settings
from daylib_ursa.workset_api import create_app


class DummyStore:
    def get_analysis(self, analysis_euid: str):
        _ = analysis_euid
        return None


class DummyBloomClient:
    def resolve_run_assignment(self, *_args, **_kwargs):  # pragma: no cover - not used
        raise AssertionError("not used")


def _settings() -> Settings:
    return Settings(
        cors_origins="*",
        ursa_internal_api_key="ursa-test-key",
        bloom_base_url="https://bloom.example",
        atlas_base_url="https://atlas.example",
        ursa_internal_output_bucket="ursa-internal",
        ursa_tapdb_mount_enabled=False,
    )


def _create_test_app():
    with patch("daylib_ursa.workset_api.RegionAwareS3Client", return_value=object()):
        return create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=_settings())


def test_public_routes_are_versioned_and_legacy_customer_routes_are_absent() -> None:
    app = _create_test_app()
    paths = {getattr(route, "path", "") for route in app.routes if getattr(route, "path", "")}
    public_api_paths = {path for path in paths if path.startswith("/api/")}
    allowed_unversioned = {
        "/api/anomalies",
        "/api/anomalies/{anomaly_id}",
    }
    assert public_api_paths
    assert all(
        path.startswith("/api/v1/") or path in allowed_unversioned for path in public_api_paths
    )
    assert not any(path.startswith("/api/customers/") for path in public_api_paths)


def test_phase_one_route_families_exist() -> None:
    app = _create_test_app()
    paths = {getattr(route, "path", "") for route in app.routes if getattr(route, "path", "")}
    expected = {
        "/api/v1/me",
        "/api/v1/analyses",
        "/api/v1/analyses/ingest",
        "/api/v1/worksets",
        "/api/v1/manifests",
        "/api/v1/buckets",
        "/api/v1/artifacts/import",
        "/api/v1/clusters",
        "/api/v1/clusters/jobs",
        "/api/v1/user-tokens",
        "/api/v1/admin/user-tokens",
        "/api/v1/admin/users",
        "/api/v1/admin/client-registrations",
        "/api/v1/admin/client-registrations/{client_registration_euid}",
        "/api/v1/admin/client-registrations/{client_registration_euid}/tokens",
    }
    assert expected.issubset(paths)


def test_all_decorated_routes_have_direct_request_coverage() -> None:
    def iter_routes(module_path: str) -> set[tuple[str, str]]:
        tree = ast.parse(Path(module_path).read_text())
        routes: set[tuple[str, str]] = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call) or not isinstance(
                    decorator.func, ast.Attribute
                ):
                    continue
                method = decorator.func.attr.upper()
                if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"}:
                    continue
                if not decorator.args or not isinstance(decorator.args[0], ast.Constant):
                    continue
                if not isinstance(decorator.args[0].value, str):
                    continue
                routes.add((method, decorator.args[0].value))
        return routes

    def sample_path(expr: ast.AST) -> str | None:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return expr.value.split("?", 1)[0]
        if isinstance(expr, ast.JoinedStr):
            parts: list[str] = []
            for value in expr.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                elif isinstance(value, ast.FormattedValue):
                    parts.append("SEGMENT")
                else:
                    return None
            return "".join(parts).split("?", 1)[0]
        return None

    def iter_direct_request_samples() -> set[tuple[str, str]]:
        samples: set[tuple[str, str]] = set()
        for path in Path("tests").glob("test_*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                method = node.func.attr.upper()
                if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"}:
                    continue
                if not node.args:
                    continue
                sample = sample_path(node.args[0])
                if sample is None:
                    continue
                samples.add((method, sample))
        return samples

    def route_matches(route: str, sample: str) -> bool:
        pattern = re.escape(route)
        pattern = re.sub(r"\\\{[^{}]+\\\}", r"[^/]+", pattern)
        return re.fullmatch(pattern, sample) is not None

    decorated_routes = iter_routes("daylib_ursa/workset_api.py") | iter_routes(
        "daylib_ursa/gui_app.py"
    )
    request_samples = iter_direct_request_samples() | {
        ("GET", "/auth/error"),
        ("GET", "/auth/logout"),
        ("POST", "/auth/logout"),
        ("GET", "/logout"),
    }
    missing = sorted(
        (method, route)
        for method, route in decorated_routes
        if not any(
            method == sample_method and route_matches(route, sample_route)
            for sample_method, sample_route in request_samples
        )
    )

    assert missing == []
