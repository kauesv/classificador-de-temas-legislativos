"""Train and compare multilabel TF-IDF models for Câmara themes."""

import argparse
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    multilabel_confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer


MODEL_NAMES = ("TF-IDF + Logistic Regression", "TF-IDF + Linear SVM")


def _code_sort(code: str):
    return (0, int(code)) if code.isdigit() else (1, code)


def _text_group(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def split_indices(texts: list[str], random_state: int = 42):
    """Return train/validation/test indices without splitting identical texts."""
    groups = np.asarray([_text_group(text) for text in texts])
    all_indices = np.arange(len(texts))
    train_validation, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state).split(
            all_indices, groups=groups
        )
    )
    train, validation = next(
        GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=random_state + 1).split(
            train_validation, groups=groups[train_validation]
        )
    )
    return train_validation[train], train_validation[validation], test


def _pipeline(model_name: str) -> Pipeline:
    if model_name == MODEL_NAMES[0]:
        classifier = OneVsRestClassifier(
            LogisticRegression(class_weight="balanced", max_iter=1000)
        )
    else:
        classifier = OneVsRestClassifier(
            CalibratedClassifierCV(
                estimator=LinearSVC(class_weight="balanced"), cv=3, method="sigmoid"
            )
        )
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    strip_accents="unicode", ngram_range=(1, 2), min_df=2,
                    max_features=100_000,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def _macro_f1(model: Pipeline, texts: list[str], targets: np.ndarray) -> float:
    probabilities = model.predict_proba(texts)
    predictions = probabilities >= 0.5
    return float(f1_score(targets, predictions, average="macro", zero_division=0))


def _top1_accuracy(model: Pipeline, texts: list[str], targets: np.ndarray) -> float:
    probabilities = model.predict_proba(texts)
    top_indices = probabilities.argmax(axis=1)
    return float(targets[np.arange(len(targets)), top_indices].mean())


def train_model(
    dataset_path: str | Path = "data/processed/dataset.csv",
    model_path: str | Path = "models/theme_classifier.joblib",
    reports_dir: str | Path = "outputs",
    *,
    top_classes: int = 10,
    min_class_samples: int = 50,
    random_state: int = 42,
    log_mlflow: bool = True,
) -> dict:
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    reports_dir = Path(reports_dir)
    if not 5 <= top_classes <= 10:
        raise ValueError("top_classes deve estar entre 5 e 10")

    frame = pd.read_csv(dataset_path, dtype={"id": str})
    required = {"id", "texto", "codigos_tema", "temas"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {', '.join(sorted(missing))}")
    if frame["id"].duplicated().any():
        raise ValueError("O dataset deve conter uma linha por ID de proposição")

    frame = frame.dropna(subset=["texto", "codigos_tema", "temas"]).copy()
    frame["texto"] = frame["texto"].astype(str).str.strip()
    frame = frame[frame["texto"] != ""].copy()
    frame["codigos_tema"] = frame["codigos_tema"].map(json.loads)
    frame["temas"] = frame["temas"].map(json.loads)
    if frame.empty:
        raise ValueError("O dataset não contém proposições com texto e tema")

    counts = Counter(code for labels in frame["codigos_tema"] for code in set(labels))
    selected_codes = [
        code
        for code, count in sorted(counts.items(), key=lambda item: (-item[1], _code_sort(item[0])))
        if count >= min_class_samples
    ][:top_classes]
    if len(selected_codes) < 5:
        raise ValueError(
            f"Só {len(selected_codes)} temas têm pelo menos {min_class_samples} exemplos; "
            "ajuste o período ou min_class_samples"
        )

    name_by_code = {}
    selected_rows = []
    for row in frame.to_dict(orient="records"):
        name_by_code.update(dict(zip(row["codigos_tema"], row["temas"])))
        labels = sorted(set(row["codigos_tema"]) & set(selected_codes))
        if labels:
            row["labels_selecionados"] = labels
            selected_rows.append(row)

    selected_frame = pd.DataFrame(selected_rows)
    texts = selected_frame["texto"].tolist()
    encoder = MultiLabelBinarizer(classes=selected_codes)
    targets = encoder.fit_transform(selected_frame["labels_selecionados"])
    if any(targets.sum(axis=0) == 0):
        raise ValueError("Há tema selecionado sem exemplos após filtrar o dataset")

    train_indices, validation_indices, test_indices = split_indices(texts, random_state)
    if any(targets[train_indices].sum(axis=0) == 0):
        raise ValueError("A divisão deixou um tema sem exemplos de treino; baixe mais anos")

    models = {name: _pipeline(name) for name in MODEL_NAMES}
    validation_metrics = {}
    validation_top1 = {}
    for name, model in models.items():
        model.fit([texts[i] for i in train_indices], targets[train_indices])
        validation_metrics[name] = _macro_f1(
            model,
            [texts[i] for i in validation_indices],
            targets[validation_indices],
        )
        validation_top1[name] = _top1_accuracy(
            model,
            [texts[i] for i in validation_indices],
            targets[validation_indices],
        )

    best_name = max(MODEL_NAMES, key=lambda name: validation_metrics[name])
    train_validation_indices = np.concatenate((train_indices, validation_indices))
    test_metrics = {}
    test_balanced_accuracy = {}
    balanced_accuracy_by_theme = {}
    test_top1 = {}
    test_reports = {}
    test_confusions = {}
    test_predictions = {}
    class_names = [name_by_code[code] for code in selected_codes]

    for name, model in models.items():
        model.fit(
            [texts[i] for i in train_validation_indices],
            targets[train_validation_indices],
        )
        probabilities = model.predict_proba([texts[i] for i in test_indices])
        predictions = probabilities >= 0.5
        test_predictions[name] = predictions
        test_targets = targets[test_indices]
        test_metrics[name] = float(
            f1_score(test_targets, predictions, average="macro", zero_division=0)
        )
        balanced_by_theme = {
            theme: (
                float(balanced_accuracy_score(test_targets[:, index], predictions[:, index]))
                if np.unique(test_targets[:, index]).size == 2
                else None
            )
            for index, theme in enumerate(class_names)
        }
        balanced_accuracy_by_theme[name] = balanced_by_theme
        test_balanced_accuracy[name] = float(
            np.mean([score for score in balanced_by_theme.values() if score is not None])
        )
        test_top1[name] = float(
            test_targets[np.arange(len(test_targets)), probabilities.argmax(axis=1)].mean()
        )
        test_reports[name] = classification_report(
            test_targets,
            predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        confusion = multilabel_confusion_matrix(test_targets, predictions)
        test_confusions[name] = {
            theme: matrix.tolist()
            for theme, matrix in zip(class_names, confusion)
        }

        if log_mlflow:
            import mlflow

            tracking_dir = reports_dir / "mlruns"
            tracking_dir.mkdir(parents=True, exist_ok=True)
            database_path = (tracking_dir / "mlflow.db").resolve().as_posix()
            mlflow.set_tracking_uri(f"sqlite:///{database_path}")
            mlflow.set_experiment("classificador-temas-legislativos")
            with mlflow.start_run(run_name=name):
                mlflow.log_params(
                    {
                        "modelo": name,
                        "temas": len(selected_codes),
                        "min_class_samples": min_class_samples,
                        "random_state": random_state,
                        "split": "65/15/20 por texto normalizado",
                    }
                )
                mlflow.log_metric("macro_f1_validacao", validation_metrics[name])
                mlflow.log_metric("top1_validacao", validation_top1[name])
                mlflow.log_metric("macro_f1_teste", test_metrics[name])
                mlflow.log_metric("balanced_accuracy_teste", test_balanced_accuracy[name])
                mlflow.log_metric("top1_teste", test_top1[name])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": models[best_name],
            "classes": selected_codes,
            "label_names": {code: name_by_code[code] for code in selected_codes},
        },
        model_path,
    )

    result = {
        "dataset": str(dataset_path),
        "amostras": len(selected_frame),
        "classes": [
            {"codigo": code, "tema": name_by_code[code], "amostras": counts[code]}
            for code in selected_codes
        ],
        "divisao": {"treino": len(train_indices), "validacao": len(validation_indices), "teste": len(test_indices)},
        "validacao": validation_metrics,
        "validacao_top1": validation_top1,
        "teste_macro_f1": test_metrics,
        "teste_balanced_accuracy": test_balanced_accuracy,
        "balanced_accuracy_por_tema": balanced_accuracy_by_theme[best_name],
        "teste_top1": test_top1,
        "modelo_escolhido": best_name,
        "relatorio_teste": test_reports[best_name],
        "matrizes_confusao_teste": test_confusions[best_name],
        "limiar_multirrótulo": 0.5,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "test_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "modelo": name,
                "macro_f1_validacao": validation_metrics[name],
                "top1_validacao": validation_top1[name],
                "macro_f1_teste": test_metrics[name],
                "balanced_accuracy_teste": test_balanced_accuracy[name],
                "top1_teste": test_top1[name],
            }
            for name in MODEL_NAMES
        ]
    ).to_csv(reports_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    best_predictions = test_predictions[best_name]
    errors = []
    for index, predicted in zip(test_indices, best_predictions):
        actual_codes = set(selected_frame.iloc[index]["labels_selecionados"])
        predicted_codes = {
            code for code, is_predicted in zip(selected_codes, predicted) if is_predicted
        }
        if actual_codes != predicted_codes:
            errors.append(
                {
                    "id": selected_frame.iloc[index]["id"],
                    "texto": texts[index],
                    "temas_reais": json.dumps(
                        [name_by_code[code] for code in sorted(actual_codes, key=_code_sort)],
                        ensure_ascii=False,
                    ),
                    "temas_previstos": json.dumps(
                        [name_by_code[code] for code in sorted(predicted_codes, key=_code_sort)],
                        ensure_ascii=False,
                    ),
                }
            )
    if len(errors) > 30:
        errors = random.Random(random_state).sample(errors, 30)
    pd.DataFrame(
        errors, columns=["id", "texto", "temas_reais", "temas_previstos"]
    ).to_csv(
        reports_dir / "error_review.csv", index=False, encoding="utf-8-sig"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina e compara os classificadores de temas")
    parser.add_argument("--dataset", default="data/processed/dataset.csv")
    parser.add_argument("--model", default="models/theme_classifier.joblib")
    parser.add_argument("--reports", default="outputs")
    parser.add_argument("--top-classes", type=int, default=10)
    parser.add_argument("--min-class-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train_model(
        args.dataset,
        args.model,
        args.reports,
        top_classes=args.top_classes,
        min_class_samples=args.min_class_samples,
        random_state=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
