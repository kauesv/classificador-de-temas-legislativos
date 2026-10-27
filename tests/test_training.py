import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.train import split_indices, train_model


class TrainingTests(unittest.TestCase):
    def test_identical_texts_never_cross_dataset_splits(self):
        texts = [f"ementa {index // 2}" for index in range(40)]
        train, validation, test = split_indices(texts)
        groups = [{" ".join(texts[index].casefold().split()) for index in split} for split in (train, validation, test)]

        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])

    def test_training_saves_predictable_pipeline_and_test_metrics(self):
        temporary_root = Path(__file__).resolve().parents[1] / ".testtmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary_directory:
            root = Path(temporary_directory)
            dataset_path = root / "dataset.csv"
            with dataset_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=["id", "texto", "codigos_tema", "temas"])
                writer.writeheader()
                for code in range(1, 6):
                    for sample in range(20):
                        writer.writerow(
                            {
                                "id": f"{code}-{sample}",
                                "texto": f"assunto tema{code} projeto exemplo {sample}",
                                "codigos_tema": json.dumps([str(code)]),
                                "temas": json.dumps([f"Tema {code}"]),
                            }
                        )

            result = train_model(
                dataset_path=dataset_path,
                model_path=root / "model.joblib",
                reports_dir=root / "reports",
                top_classes=5,
                min_class_samples=5,
                log_mlflow=False,
            )

            self.assertTrue((root / "model.joblib").is_file())
            self.assertTrue(any("Logistic Regression" in name for name in result["validacao"]))
            self.assertIn("teste_balanced_accuracy", result)
            self.assertTrue((root / "reports" / "test_metrics.json").is_file())
            with (root / "reports" / "error_review.csv").open(encoding="utf-8-sig") as errors_file:
                self.assertLessEqual(len(list(csv.DictReader(errors_file))), 30)

    def test_training_logs_mlflow_to_sqlite(self):
        temporary_root = Path(__file__).resolve().parents[1] / ".testtmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary_directory:
            root = Path(temporary_directory)
            dataset_path = root / "dataset.csv"
            with dataset_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=["id", "texto", "codigos_tema", "temas"])
                writer.writeheader()
                for code in range(1, 6):
                    for sample in range(20):
                        writer.writerow(
                            {
                                "id": f"{code}-{sample}",
                                "texto": f"assunto tema{code} projeto exemplo {sample}",
                                "codigos_tema": json.dumps([str(code)]),
                                "temas": json.dumps([f"Tema {code}"]),
                            }
                        )

            script = (
                "from app.train import train_model; "
                f"train_model({str(dataset_path)!r}, {str(root / 'model.joblib')!r}, "
                f"{str(root / 'reports')!r}, top_classes=5, min_class_samples=5)"
            )
            environment = os.environ | {"MLFLOW_DISABLE_AGENT_HINT": "1"}
            subprocess.run(
                [sys.executable, "-c", script],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue((root / "reports" / "mlruns" / "mlflow.db").is_file())


if __name__ == "__main__":
    unittest.main()
