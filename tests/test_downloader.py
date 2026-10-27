import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.baixar_dados import download_dataset, download_years


class DownloaderTests(unittest.TestCase):
    def test_download_writes_valid_archive(self):
        payload = json.dumps({"dados": [{"id": 123}]}).encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch("app.baixar_dados.urlopen", return_value=io.BytesIO(payload)) as urlopen:
                path = download_dataset("proposicoes", 2026, data_root=temporary_directory)

            self.assertEqual(
                path.read_bytes(), payload,
            )
            urlopen.assert_called_once_with(
                "https://dadosabertos.camara.leg.br/arquivos/proposicoes/json/proposicoes-2026.json",
                timeout=120,
            )
            self.assertFalse(path.with_suffix(path.suffix + ".tmp").exists())

    def test_download_rejects_unexpected_json_before_writing(self):
        payload = json.dumps({"wrong": []}).encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch("app.baixar_dados.urlopen", return_value=io.BytesIO(payload)):
                with self.assertRaisesRegex(ValueError, "Resposta JSON inesperada"):
                    download_dataset("temas", 2026, data_root=temporary_directory)

            self.assertFalse((Path(temporary_directory) / "proposicoes" / "temas" / "proposicoesTemas-2026.json").exists())

    def test_download_years_skips_unused_author_archive(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch(
                "app.baixar_dados.download_dataset",
                side_effect=lambda dataset, year, **kwargs: Path(temporary_directory) / dataset,
            ) as download:
                download_years(2026, 2026)

        self.assertEqual(
            [call.args[0] for call in download.call_args_list],
            ["proposicoes", "temas"],
        )


if __name__ == "__main__":
    unittest.main()
