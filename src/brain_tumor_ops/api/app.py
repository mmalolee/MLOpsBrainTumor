from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from src.brain_tumor_ops.configs.model import ModelConfig
from src.brain_tumor_ops.configs.paths import PathsConfig
from src.brain_tumor_ops.configs.prediction import PredictionConfig
from src.brain_tumor_ops.configs.preprocessing import DataPreprocessingConfig
from src.brain_tumor_ops.pipelines.prediction_pipeline import PredictionPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = PredictionPipeline(
        PredictionConfig(), PathsConfig(), ModelConfig(), DataPreprocessingConfig()
    )

    yield

    del app.state.pipeline


app = FastAPI(description="eee", lifespan=lifespan)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile, request: Request):
    try:
        with Image.open(file.file) as uploaded_image:
            image = uploaded_image.convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Provided file is not an image.")

    classification_pipeline = request.app.state.pipeline
    classification = classification_pipeline.predict_image(image)

    return {"classification_result": classification}
