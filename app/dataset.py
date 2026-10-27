"""Build a reproducible text/labels table from Câmara download archives."""

import json
import argparse
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


def _records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as source:
        payload = json.load(source)
    if not isinstance(payload, dict) or not isinstance(payload.get("dados"), list):
        raise ValueError(f"Formato JSON inesperado em {path}")
    return payload["dados"]


def _proposition_id(uri: str) -> str:
    return urlparse(uri).path.rstrip("/").split("/")[-1]


def build_dataset(
    data_dir: str | Path = "data/proposicoes",
    output_path: str | Path = "data/processed/dataset.csv",
    *,
    top_classes: int = 10,
    min_samples: int = 50,
) -> dict:
    """Join proposition ementas and theme codes by Câmara proposition ID."""
    root = Path(data_dir)
    output = Path(output_path)
    proposition_files = sorted(root.glob("proposicoes-*.json"))
    if not proposition_files:
        raise FileNotFoundError(f"Nenhum arquivo proposicoes-AAAA.json em {root}")

    propositions: dict[str, dict] = {}
    themes_by_id: dict[str, dict[str, str]] = defaultdict(dict)
    theme_names: dict[str, str] = {}
    missing_theme_files = []
    themes_without_proposition = 0

    for proposition_file in proposition_files:
        match = re.fullmatch(r"proposicoes-(\d{4})\.json", proposition_file.name)
        if not match:
            continue
        year = match.group(1)
        theme_file = root / "temas" / f"proposicoesTemas-{year}.json"
        if not theme_file.is_file():
            missing_theme_files.append(year)
            continue

        year_propositions = _records(proposition_file)
        year_ids = set()
        for proposition in year_propositions:
            proposition_id = str(proposition.get("id", "")).strip()
            if not proposition_id:
                continue
            if proposition_id in propositions:
                raise ValueError(f"ID de proposição duplicado em anos diferentes: {proposition_id}")
            propositions[proposition_id] = proposition
            year_ids.add(proposition_id)

        for theme in _records(theme_file):
            proposition_id = _proposition_id(str(theme.get("uriProposicao", "")))
            if proposition_id not in year_ids:
                themes_without_proposition += 1
                continue
            code = str(theme.get("codTema", "")).strip()
            name = str(theme.get("tema", "")).strip()
            if code and name:
                previous_name = theme_names.setdefault(code, name)
                if previous_name != name:
                    raise ValueError(
                        f"Código de tema {code} associado a nomes diferentes: "
                        f"{previous_name!r} e {name!r}"
                    )
                themes_by_id[proposition_id][code] = name

    if not themes_by_id:
        reason = f"; sem arquivo de temas para os anos {', '.join(missing_theme_files)}" if missing_theme_files else ""
        raise FileNotFoundError(f"Nenhuma associação proposição/tema encontrada em {root}{reason}")

    rows = []
    empty_ementas = 0
    unlabeled = 0
    for proposition_id, proposition in propositions.items():
        text = " ".join(str(proposition.get("ementa") or "").split())
        if not text:
            empty_ementas += 1
            continue
        labels = themes_by_id.get(proposition_id, {})
        if not labels:
            unlabeled += 1
            continue
        ordered = sorted(labels.items(), key=lambda item: item[0])
        rows.append(
            {
                "id": proposition_id,
                "ano": proposition.get("ano", ""),
                "texto": text,
                "codigos_tema": json.dumps([code for code, _ in ordered], ensure_ascii=False),
                "temas": json.dumps([name for _, name in ordered], ensure_ascii=False),
            }
        )

    if not rows:
        raise ValueError("Nenhuma proposição com ementa e tema após a limpeza")

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")

    counts = Counter(code for row in rows for code in json.loads(row["codigos_tema"]))
    name_by_code = {
        code: name
        for row in rows
        for code, name in zip(json.loads(row["codigos_tema"]), json.loads(row["temas"]))
    }
    eligible = [(code, count) for code, count in counts.items() if count >= min_samples]
    selected = sorted(eligible, key=lambda item: (-item[1], item[0]))[:top_classes]
    summary = {
        "data_processamento": date.today().isoformat(),
        "anos": sorted({str(row["ano"]) for row in rows}),
        "arquivos_proposicoes": len(proposition_files),
        "anos_sem_arquivo_de_temas": missing_theme_files,
        "proposicoes_lidas": len(propositions),
        "proposicoes_com_texto_e_tema": len(rows),
        "ementas_vazias": empty_ementas,
        "proposicoes_sem_tema": unlabeled,
        "temas_sem_proposicao": themes_without_proposition,
        "temas_distintos": len(counts),
        "classes_selecionadas": [
            {"codigo": code, "tema": name_by_code[code], "amostras": count}
            for code, count in selected
        ],
        "distribuicao_temas": [
            {"codigo": code, "tema": name_by_code[code], "amostras": count}
            for code, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ],
    }
    summary_path = output.with_name("dataset_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Junta ementas e temas oficiais da Câmara")
    parser.add_argument("--data-dir", default="data/proposicoes")
    parser.add_argument("--output", default="data/processed/dataset.csv")
    parser.add_argument("--top-classes", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=50)
    args = parser.parse_args()
    summary = build_dataset(
        args.data_dir,
        args.output,
        top_classes=args.top_classes,
        min_samples=args.min_samples,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
