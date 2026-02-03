from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from catboost import CatBoostRegressor
import pathlib

app = FastAPI()

# -----------------------------
# CORS (browser-safe)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   # includes OPTIONS
    allow_headers=["*"],
)

BASE_DIR = pathlib.Path(__file__).parent

# -----------------------------
# Load model & data ONCE
# -----------------------------
model = CatBoostRegressor()
model.load_model("quali_q3_delta_model.cbm")

medians = pd.read_csv("circuit_medians.csv")

categorical_features = [
    "Driver", "Team", "Compound", "Event", "Session",
    "QualiSegment", "CircuitName", "Country",
    "TrackType", "LapSpeedClass",
    "Driver_Track", "Team_Track"
]

numeric_features = [
    "TyreLife", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "TrackLength_m", "NumCorners", "CornerDensity",
    "AvgCornerSpacing_m", "AirTemp", "TrackTemp",
    "WindSpeed", "Altitude_m", "DRSZones"
]

features = categorical_features + numeric_features

# -----------------------------
# Request schema
# -----------------------------
class PredictRequest(BaseModel):
    driver: str
    team: str
    event: str
    quali_segment: str

# -----------------------------
# Prediction logic
# -----------------------------
def predict_quali_time(driver, team, event, quali_segment):
    row = medians[
        (medians["Event"] == event) &
        (medians["QualiSegment"] == quali_segment)
    ]

    if row.empty:
        raise ValueError("No median data found")

    row = row.iloc[0]
    session_median = row["SessionMedianLap"]

    input_data = {
        "Driver": driver,
        "Team": team,
        "Compound": "SOFT",
        "Event": event,
        "Session": "Q",
        "QualiSegment": quali_segment,

        "CircuitName": row["CircuitName"],
        "Country": row["Country"],
        "TrackType": row["TrackType"],
        "LapSpeedClass": row["LapSpeedClass"],

        "Driver_Track": f"{driver}_{row['CircuitName']}",
        "Team_Track": f"{team}_{row['CircuitName']}",

        "TyreLife": 2,
        "SpeedI1": row["SpeedI1"],
        "SpeedI2": row["SpeedI2"],
        "SpeedFL": row["SpeedFL"],
        "SpeedST": row["SpeedST"],
        "TrackLength_m": row["TrackLength_m"],
        "NumCorners": row["NumCorners"],
        "CornerDensity": row["CornerDensity"],
        "AvgCornerSpacing_m": row["AvgCornerSpacing_m"],
        "AirTemp": row["AirTemp"],
        "TrackTemp": row["TrackTemp"],
        "WindSpeed": row["WindSpeed"],
        "Altitude_m": row["Altitude_m"],
        "DRSZones": row["DRSZones"],
    }

    X = pd.DataFrame([input_data])[features]
    predicted_delta = model.predict(X)[0]

    return round(session_median + predicted_delta, 3)

# -----------------------------
# Serve frontend (optional)
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return (BASE_DIR / "index.html").read_text()

# -----------------------------
# API endpoint (BOTH routes)
# -----------------------------
@app.post("/predict")
@app.post("/predict/")
def predict(req: PredictRequest):
    try:
        lap_time = predict_quali_time(
            req.driver,
            req.team,
            req.event,
            req.quali_segment
        )
        return JSONResponse({
            "predicted_lap_time_sec": lap_time,
            "real_lap_time_sec": None
        })
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
