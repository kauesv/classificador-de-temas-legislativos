from fastapi import FastAPI

app = FastAPI(
    title="Classificador de Legislativos",
    version="1.0.0",
)

@app.get("/health")
async def root():
    return {"status": "online"}

@app.post("/predict")
async def predict():
    return {"status": "Feito"}