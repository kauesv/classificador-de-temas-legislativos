import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.inference import predict_text


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "theme_classifier.joblib"))

app = FastAPI(
    title="Classificador de Temas Legislativos",
    version="1.0.0",
)
app.state.model_bundle = None


class PredictionRequest(BaseModel):
    texto: str = Field(min_length=1, max_length=20_000)

    @field_validator("texto")
    @classmethod
    def texto_nao_pode_ser_vazio(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("O texto não pode estar vazio")
        return value


class PredictionResponse(BaseModel):
    tema: str
    confianca: float | None


def _model_bundle() -> dict:
    if app.state.model_bundle is None:
        if not MODEL_PATH.is_file():
            raise HTTPException(status_code=503, detail="Modelo ainda não foi treinado")
        import joblib

        app.state.model_bundle = joblib.load(MODEL_PATH)
    return app.state.model_bundle


@app.get("/health")
async def health():
    try:
        _model_bundle()
    except Exception:
        return {"status": "online", "model_ready": False}
    return {"status": "online", "model_ready": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    return predict_text(_model_bundle(), request.texto)
