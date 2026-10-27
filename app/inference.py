from collections.abc import Mapping


def predict_text(bundle: Mapping, text: str) -> dict:
    probabilities = bundle["model"].predict_proba([text])[0]
    best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    code = bundle["classes"][best_index]
    return {
        "tema": bundle["label_names"][code],
        "confianca": round(float(probabilities[best_index]), 4),
    }
