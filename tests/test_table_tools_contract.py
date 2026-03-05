"""Contract tests for universal table tooling on portal/admin views."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from daylily_ursa.workset_api import create_app


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_admin_client() -> TestClient:
    state_db = MagicMock()
    customer_manager = MagicMock()

    admin_user = MagicMock()
    admin_user.email = "admin@example.com"
    admin_user.customer_id = "cust-admin"
    admin_user.customer_name = "Admin"
    admin_user.is_admin = True

    customer_manager.get_customer_by_email.side_effect = (
        lambda email: admin_user if email == admin_user.email else None
    )
    customer_manager.list_customers.return_value = [admin_user]

    app = create_app(
        state_db=state_db,
        enable_auth=False,
        customer_manager=customer_manager,
    )

    client = TestClient(app, base_url="https://testserver")
    client.post(
        "/portal/login",
        data={"email": admin_user.email, "password": "irrelevant"},
        follow_redirects=False,
    )
    return client


def test_table_utils_targets_all_tables_and_supports_filter_download_dynamic_updates():
    js_path = REPO_ROOT / "static" / "js" / "table-utils.js"
    content = js_path.read_text(encoding="utf-8")

    assert "querySelectorAll('table')" in content
    assert "data-table-tools" in content
    assert "table-filter-input" in content
    assert "downloadTableAsTSV" in content
    assert "MutationObserver" in content


def test_templates_with_tables_extend_base_html():
    templates_dir = REPO_ROOT / "templates"
    missing = []

    for template in templates_dir.rglob("*.html"):
        content = template.read_text(encoding="utf-8")
        if "<table" not in content:
            continue
        if "{% extends \"base.html\" %}" in content or "{% extends 'base.html' %}" in content:
            continue
        missing.append(str(template))

    assert not missing, f"Templates with tables must extend base.html: {missing}"


def test_representative_portal_pages_include_table_tools_script(monkeypatch):
    fake_service = SimpleNamespace(get_all_clusters_with_status=lambda fetch_ssh_status=False: [])
    monkeypatch.setattr("daylily_ursa.cluster_service.get_cluster_service", lambda **kwargs: fake_service)

    fake_config = SimpleNamespace(
        is_configured=False,
        aws_profile=None,
        get_allowed_regions=lambda: ["us-west-2"],
    )
    monkeypatch.setattr("daylily_ursa.ursa_config.get_ursa_config", lambda: fake_config)

    client = _make_admin_client()

    for path in ["/portal/admin/users", "/portal/info", "/portal/clusters"]:
        response = client.get(path)
        assert response.status_code == 200
        assert "/static/js/table-utils.js" in response.text
        assert "<table" in response.text
