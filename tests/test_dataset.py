import csv
import json
import tempfile
import unittest
from pathlib import Path

from app.dataset import build_dataset


class BuildDatasetTests(unittest.TestCase):
    def test_joins_topics_by_proposition_id_and_drops_empty_ementas(self):
        temporary_root = Path(__file__).resolve().parents[1] / ".testtmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary_directory:
            root = Path(temporary_directory)
            (root / "temas").mkdir()
            (root / "proposicoes-2026.json").write_text(
                json.dumps(
                    {
                        "dados": [
                            {"id": 1, "ano": 2026, "ementa": "  Texto da saúde.\n"},
                            {"id": 2, "ano": 2026, "ementa": "   "},
                            {"id": 3, "ano": 2026, "ementa": "Texto da educação."},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "temas" / "proposicoesTemas-2026.json").write_text(
                json.dumps(
                    {
                        "dados": [
                            {"uriProposicao": "https://camara/proposicoes/1", "codTema": 1, "tema": "Saúde"},
                            {"uriProposicao": "https://camara/proposicoes/1", "codTema": 1, "tema": "Saúde"},
                            {"uriProposicao": "https://camara/proposicoes/3", "codTema": 2, "tema": "Educação"},
                            {"uriProposicao": "https://camara/proposicoes/999", "codTema": 3, "tema": "Órfã"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_path = root / "processed" / "dataset.csv"
            summary = build_dataset(root, output_path, min_samples=1, top_classes=10)

            rows = output_path.read_text(encoding="utf-8-sig").splitlines()
            with output_path.open(encoding="utf-8-sig", newline="") as source:
                dataset = list(csv.DictReader(source))
            self.assertEqual(summary["proposicoes_com_texto_e_tema"], 2)
            self.assertEqual(summary["ementas_vazias"], 1)
            self.assertEqual(summary["temas_sem_proposicao"], 1)
            self.assertEqual(len(rows), 3)  # cabeçalho e duas proposições
            self.assertIn('"1"', rows[1])
            self.assertEqual(dataset[0]["texto"], "Texto da saúde.")

    def test_rejects_conflicting_names_for_same_theme_code(self):
        temporary_root = Path(__file__).resolve().parents[1] / ".testtmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary_directory:
            root = Path(temporary_directory)
            (root / "temas").mkdir()
            (root / "proposicoes-2026.json").write_text(
                json.dumps(
                    {
                        "dados": [
                            {"id": 1, "ano": 2026, "ementa": "Texto um"},
                            {"id": 2, "ano": 2026, "ementa": "Texto dois"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "temas" / "proposicoesTemas-2026.json").write_text(
                json.dumps(
                    {
                        "dados": [
                            {"uriProposicao": "https://camara/proposicoes/1", "codTema": 1, "tema": "Saúde"},
                            {"uriProposicao": "https://camara/proposicoes/2", "codTema": 1, "tema": "Saude"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "nomes diferentes"):
                build_dataset(root, root / "processed" / "dataset.csv", min_samples=1)


if __name__ == "__main__":
    unittest.main()
