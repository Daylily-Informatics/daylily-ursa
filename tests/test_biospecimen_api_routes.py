"""Request-level coverage for biospecimen API routes.

These tests are intentionally lightweight: they exercise route registration and
basic request handling without requiring real TapDB/AWS.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from daylily_ursa.biospecimen import Biosample, Biospecimen, BiospecimenRegistry, Library, Subject
from daylily_ursa.biospecimen_api import create_biospecimen_router


def test_biospecimen_api_routes_have_request_level_coverage():
    registry = MagicMock(spec=BiospecimenRegistry)

    # Return owned entities so handlers avoid 403/404.
    registry.list_subjects.return_value = []
    registry.get_subject.return_value = Subject(subject_id="subj-001", customer_id="cust-001", display_name="HG002")
    registry.create_subject.return_value = True
    registry.update_subject.return_value = True
    registry.get_subject_hierarchy.return_value = {"subject_id": "subj-001", "biospecimens": []}

    registry.list_biosamples.return_value = []
    registry.get_biospecimen.return_value = Biospecimen(
        biospecimen_id="bspec-001",
        customer_id="cust-001",
        subject_id="subj-001",
    )
    registry.create_biosample.return_value = True
    registry.get_biosample.return_value = Biosample(
        biosample_id="bio-001",
        customer_id="cust-001",
        biospecimen_id="bspec-001",
        subject_id="subj-001",
    )
    registry.update_biosample.return_value = True

    registry.list_libraries.return_value = []
    registry.list_libraries_for_biosample.return_value = []
    registry.create_library.return_value = True
    registry.get_library.return_value = Library(
        library_id="lib-001",
        customer_id="cust-001",
        biosample_id="bio-001",
    )
    registry.update_library.return_value = True

    registry.get_statistics.return_value = {"subjects": 0, "biospecimens": 0, "biosamples": 0, "libraries": 0}

    def get_customer_id(_: Request) -> str:
        return "cust-001"

    app = FastAPI()
    app.include_router(create_biospecimen_router(registry=registry, get_customer_id=get_customer_id))

    with TestClient(app, base_url="https://testserver") as client:
        # Subjects
        assert client.get("/api/biospecimen/subjects").status_code != 404
        assert client.get("/api/biospecimen/subjects/subj-001").status_code != 404
        assert client.get("/api/biospecimen/subjects/subj-001/hierarchy").status_code != 404
        assert client.post("/api/biospecimen/subjects", json={"identifier": "HG002"}).status_code != 404
        assert client.put("/api/biospecimen/subjects/subj-001", json={"identifier": "HG002"}).status_code != 404

        # Biosamples
        assert client.get("/api/biospecimen/biosamples").status_code != 404
        assert client.get("/api/biospecimen/biosamples/bio-001").status_code != 404
        assert client.post(
            "/api/biospecimen/biosamples",
            json={"identifier": "S1", "biospecimen_id": "bspec-001"},
        ).status_code != 404
        assert client.put(
            "/api/biospecimen/biosamples/bio-001",
            json={"identifier": "S1", "biospecimen_id": "bspec-001"},
        ).status_code != 404

        # Libraries
        assert client.get("/api/biospecimen/libraries").status_code != 404
        assert client.get("/api/biospecimen/libraries/lib-001").status_code != 404
        assert client.post(
            "/api/biospecimen/libraries",
            json={"identifier": "L1", "biosample_id": "bio-001"},
        ).status_code != 404
        assert client.put(
            "/api/biospecimen/libraries/lib-001",
            json={"identifier": "L1", "biosample_id": "bio-001"},
        ).status_code != 404

        # Statistics
        assert client.get("/api/biospecimen/statistics").status_code != 404
