from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.brain_tumor_ops.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_predict_accepts_image(client):
    image_path = Path(__file__).resolve().parent / "fixtures" / "sample.jpg"

    with image_path.open("rb") as image_file:
        response = client.post(
            "/predict", files={"file": ("sample.jpg", image_file, "image/jpeg")}
        )

    assert response.status_code == 200, response.text

    result = response.json()

    assert "classification_result" in result
    assert isinstance(result["classification_result"], str)
    assert result["classification_result"] != ""


def test_predict_rejects_text_file(client):
    response = client.post(
        "/predict",
        files={"file": ("notes.md", b"# It's not an image.", "text/markdown")},
    )

    assert response.status_code == 400, response.text
    assert response.json() == {
        "detail": "Provided file is not an image.",
    }
