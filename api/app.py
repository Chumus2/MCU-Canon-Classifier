import joblib
import pandas as pd

from src.utils import load_model

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict


app = FastAPI(
    title="MCU Canon Classificator",
    description="Predicts whether a Marvel movie/series belongs to the official MCU canon.",
    version="1.0.0"
)


# loaded once at startup, not on every request - model.pkl holds
# {"model": ..., "threshold": ..., "feature_names": [...]}
artifact = load_model("../machine-learning/models/model.pkl")
model = artifact["model"]
threshold = artifact["threshold"]
feature_names = artifact["feature_names"]


class Movie_Features(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    # One row of input features, matching exactly what the model was trained on.
    # Field names must match `feature_names` saved alongside the model.
    runtime_min: float = Field(..., description="Runtime in minutes")
    year: int = Field(..., description="Release year")
    age_years: int = Field(..., description="Movie age in years")
    is_animated: int = Field(..., ge=0, le=1)
    is_tv_series: int = Field(..., ge=0, le=1)
    budget_log: float = Field(..., description="Budget log")
    revenue_log: float = Field(..., description="Revenue log")
    box_office_log: float = Field(..., description="Box ofice log")
    imdb_votes: float = Field(..., description="IMDb votes")
    tmdb_votes: float = Field(..., description="TMDB votes")
    popularity: float = Field(..., description="Popularity")
    IMDb: float = Field(..., description="IMDb rating")
    Metacritic: float = Field(..., description="Metacritic")
    Rotten_Tomatoes: float = Field(..., alias="Rotten Tomatoes")
    TMDB: float = Field(..., description="TMDB rating")

class PredictionResponse(BaseModel):
    is_mcu_canon: int
    probability: float
    threshold_used: float


@app.get("/")
def root():
    return {"status": "ok", "model": "XGBClassifier", "threshold": threshold}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: Movie_Features):
    input_dict = features.dict(by_alias=True)

    # build a single-row DataFrame with columns in the exact order the model expects;
    # missing/extra fields would silently misalign features otherwise
    try:
        row = pd.DataFrame([[input_dict[col] for col in feature_names]], columns=feature_names)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing required column {e}")

    proba = model.predict_proba(row)[:, 1][0]
    prediction = int(proba >= threshold)

    return PredictionResponse(
        is_mcu_canon=prediction,
        probability=round(float(proba), 4),
        threshold_used=threshold,
    )


if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise