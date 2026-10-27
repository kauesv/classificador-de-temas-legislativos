"""Download Câmara JSON archives without replacing existing files by default."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "proposicoes": ("proposicoes", "proposicoes", "proposicoes-{year}.json"),
    "temas": ("proposicoesTemas", "proposicoes/temas", "proposicoesTemas-{year}.json"),
}


def download_dataset(
    dataset: str,
    year: int,
    *,
    data_root: str | Path = ROOT / "data",
    overwrite: bool = False,
) -> Path | None:
    if dataset not in DATASETS:
        raise ValueError(f"Tipo de arquivo desconhecido: {dataset}")

    route, relative_directory, filename = DATASETS[dataset]
    destination = Path(data_root) / relative_directory / filename.format(year=year)
    if destination.exists() and not overwrite:
        return None

    url = (
        "https://dadosabertos.camara.leg.br/arquivos/"
        f"{route}/json/"
        f"{filename.format(year=year)}"
    )

    with urlopen(url, timeout=120) as response:
        content = response.read()
    payload = json.loads(content)

    if not isinstance(payload, dict) or not isinstance(payload.get("dados"), list):
        raise ValueError(f"Resposta JSON inesperada em {url}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def download_years(start_year: int, end_year: int, *, overwrite: bool = False) -> list[Path]:
    if start_year > end_year:
        raise ValueError("O ano inicial deve ser menor ou igual ao ano final")
    downloaded = []
    for year in range(start_year, end_year + 1):
        for dataset in DATASETS:
            path = download_dataset(dataset, year, overwrite=overwrite)
            if path is None:
                print(f"Já existe, mantendo: {year} / {dataset}")
            else:
                downloaded.append(path)
                print(f"Baixado: {path}")
    return downloaded

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baixa dados anuais da Câmara dos Deputados")
    parser.add_argument("--inicio", type=int, required=True, help="Primeiro ano, inclusive")
    parser.add_argument("--fim", type=int, required=True, help="Último ano, inclusive")
    parser.add_argument("--sobrescrever", action="store_true", help="Atualiza arquivos já existentes")
    args = parser.parse_args()
    download_years(args.inicio, args.fim, overwrite=args.sobrescrever)
