# tests/test_api.py
"""API tests via FastAPI TestClient — agent.run and the RAG retriever are
mocked, so no ONNX model, no DashScope calls, no network."""

import pytest
from fastapi.testclient import TestClient

import app.api.server as server


@pytest.fixture
def client():
    return TestClient(server.app)


@pytest.fixture(autouse=True)
def _uploads_to_tmp(tmp_path, monkeypatch):
    # uploaded files must not pollute the real uploads/ dir
    monkeypatch.setattr(server, "UPLOAD_DIR", str(tmp_path / "uploads"))


FAKE_STATE = {
    "det_result": {
        "image": "x.jpg", "inference_ms": 12.3,
        "counts": {"person": 2},
        "detections": [
            {"class": "person", "confidence": 0.9, "bbox": [1, 2, 3, 4]},
            {"class": "person", "confidence": 0.8, "bbox": [5, 6, 7, 8]},
        ],
    },
    "risk_level": "high",
    "vlm_report": "### 1. Scene Description\nA busy construction site.",
    "standards": [{
        "text": "Workers must wear helmets." + " " * 50,
        "standard": "Construction Site Safety",
        "source": "construction_site_safety.md",
        "distance": 0.2,
    }],
    "decisions": ["Risk level: high -> retrieving top 5 standards"],
    "saved": {"markdown": "reports/x.md"},
}


class TestHealth:
    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestInspect:
    def test_happy_path_schema(self, client, monkeypatch):
        called = {}

        def fake_run(path, conf_thres=0.25, save=True):
            called["path"] = path
            called["conf"] = conf_thres
            return dict(FAKE_STATE)

        monkeypatch.setattr(server.agent, "run", fake_run)
        resp = client.post(
            "/inspect",
            files={"image": ("site.jpg", b"\xff\xd8\xff\xe0 fake jpeg",
                             "image/jpeg")},
            data={"conf": "0.4", "save": "false"},
        )
        assert resp.status_code == 200
        body = resp.json()

        assert called["conf"] == 0.4
        assert body["detection"]["object_count"] == 2
        assert body["detection"]["counts"] == {"person": 2}
        assert body["risk_level"] == "high"
        assert body["vlm_report"].startswith("### 1.")
        # relevance = 1 - distance
        assert body["standards"][0]["relevance"] == 0.8
        assert body["agent_decisions"] == FAKE_STATE["decisions"]

    def test_unsupported_extension_415(self, client):
        resp = client.post(
            "/inspect",
            files={"image": ("evil.exe", b"payload", "application/octet-stream")},
        )
        assert resp.status_code == 415
        assert "Unsupported file type" in resp.json()["detail"]

    def test_agent_exception_500(self, client, monkeypatch):
        def exploding_run(path, conf_thres=0.25, save=True):
            raise RuntimeError("VLM backend down")

        monkeypatch.setattr(server.agent, "run", exploding_run)
        resp = client.post(
            "/inspect",
            files={"image": ("site.jpg", b"\xff\xd8 fake", "image/jpeg")},
        )
        assert resp.status_code == 500
        assert "Agent failed" in resp.json()["detail"]

    def test_defaults_conf_and_save(self, client, monkeypatch):
        seen = {}

        def fake_run(path, conf_thres=0.25, save=True):
            seen["conf"], seen["save"] = conf_thres, save
            return dict(FAKE_STATE)

        monkeypatch.setattr(server.agent, "run", fake_run)
        resp = client.post("/inspect",
                           files={"image": ("a.png", b"x", "image/png")})
        assert resp.status_code == 200
        assert seen["conf"] == 0.25
        assert seen["save"] is True


class TestStandards:
    def test_lists_deduplicated_sources(self, client, monkeypatch):
        class FakeCollection:
            def get(self, include=None):
                return {"metadatas": [
                    {"source": "a.md", "standard": "Std A"},
                    {"source": "a.md", "standard": "Std A"},
                    {"source": "b.md", "standard": "Std B"},
                ]}

        class FakeRetriever:
            collection = FakeCollection()

        monkeypatch.setattr(server, "_get_retriever", lambda: FakeRetriever())
        resp = client.get("/standards")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert {s["source"] for s in body["standards"]} == {"a.md", "b.md"}


class TestOpenAPISchema:
    def test_docs_generated(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json()["paths"]
        assert set(paths) >= {"/health", "/inspect", "/standards"}
