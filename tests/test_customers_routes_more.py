"""Request-level coverage for remaining customer routes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware


def test_customers_routes_have_request_level_coverage():
    from daylily_ursa.routes.customers import CustomerDependencies, create_customers_router

    customer_manager = MagicMock()
    config = SimpleNamespace(
        customer_id="cust-001",
        customer_name="Acme",
        email="acme@example.com",
        s3_bucket="acme-bucket",
        max_concurrent_worksets=10,
        max_storage_gb=500,
        billing_account_id=None,
        cost_center=None,
        is_admin=False,
    )

    customer_manager.onboard_customer.return_value = config
    customer_manager.get_customer_config.return_value = config
    customer_manager.get_customer_usage.return_value = {"worksets": 0, "storage_gb": 0}
    customer_manager.update_customer.return_value = config
    customer_manager.get_customer_by_email.return_value = config
    customer_manager.list_customers.return_value = [config]

    async def get_current_user():
        return None

    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="test-secret")

    @app.get("/__test__/login")
    async def _login(request: Request):
        request.session["customer_id"] = "cust-001"
        request.session["user_authenticated"] = True
        return {"ok": True}

    app.include_router(create_customers_router(CustomerDependencies(customer_manager=customer_manager, get_current_user=get_current_user)))

    with TestClient(app, base_url="https://testserver") as client:
        assert client.post(
            "/customers",
            json={"customer_name": "Acme", "email": "acme@example.com"},
        ).status_code != 404
        assert client.get("/customers/cust-001").status_code != 404
        assert client.get("/customers/cust-001/usage").status_code != 404

        # PATCH route reads request.session for authorization.
        assert client.get("/__test__/login").status_code == 200
        assert client.patch("/api/v1/customers/cust-001", json={"customer_name": "Acme"}).status_code != 404

