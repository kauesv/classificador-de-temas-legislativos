import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main


class FakeModel:
    def predict_proba(self, texts):
        return [[0.15, 0.85]]


class PredictEndpointTests(unittest.TestCase):
    def setUp(self):
        main.app.state.model_bundle = {
            "model": FakeModel(),
            "classes": ["1", "2"],
            "label_names": {"1": "Saúde", "2": "Educação"},
        }
        self.client = TestClient(main.app)

    def tearDown(self):
        main.app.state.model_bundle = None

    def test_predict_returns_highest_scoring_theme(self):
        response = self.client.post("/predict", json={"texto": "Acesso à educação básica"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["tema"], "Educação")
        self.assertEqual(response.json()["confianca"], 0.85)

    def test_predict_rejects_blank_text(self):
        response = self.client.post("/predict", json={"texto": "   "})

        self.assertEqual(response.status_code, 422)

    def test_health_reports_unreadable_model_as_not_ready(self):
        temporary_root = Path(__file__).resolve().parents[1] / ".testtmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary_directory:
            corrupt_model = Path(temporary_directory) / "corrupt.joblib"
            corrupt_model.write_text("invalid", encoding="utf-8")
            main.app.state.model_bundle = None
            with patch.object(main, "MODEL_PATH", corrupt_model), patch(
                "joblib.load", side_effect=ValueError("modelo inválido")
            ):
                response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["model_ready"])

    def test_openrouter_endpoint_is_removed(self):
        response = self.client.post(
            "/predict/openrouter", json={"texto": "Acesso à educação básica"}
        )

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
