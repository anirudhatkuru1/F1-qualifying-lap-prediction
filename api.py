from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from catboost import CatBoostRegressor

DRIVER_NAME_TO_CODE = {
    "Lando Norris": "NOR",
    "Max Verstappen": "VER",
    "Oscar Piastri": "PIA",
    "George Russell": "RUS",
    "Charles Leclerc": "LEC",
    "Lewis Hamilton": "HAM",
    "Andrea Kimi Antonelli": "ANT",
    "Alexander Albon": "ALB",
    "Carlos Sainz": "SAI",
    "Fernando Alonso": "ALO",
    "Nico Hulkenberg": "HUL",
    "Isack Hadjar": "HAD",
    "Oliver Bearman": "BEA",
    "Liam Lawson": "LAW",
    "Esteban Ocon": "OCO",
    "Lance Stroll": "STR",
    "Yuki Tsunoda": "TSU",
    "Pierre Gasly": "GAS",
    "Gabriel Bortoleto": "BOR",
    "Franco Colapinto": "COL",
    "Jack Doohan": "DOO"
}

CAT_FEATURES = [
    "Driver", "Team", "Compound", "Event", "Session",
    "QualiSegment", "CircuitName", "Country",
    "TrackType", "LapSpeedClass"
]

EXPECTED_FEATURES = CAT_FEATURES + [
    "TyreLife", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "TrackLength_m", "NumCorners", "CornerDensity",
    "AvgCornerSpacing_m", "AirTemp", "TrackTemp",
    "WindSpeed", "Altitude_m", "DRSZones"
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "F1 Qualifying Prediction API is running 🚀"}

# Load once
model = CatBoostRegressor()
model.load_model("quali_q3_delta_model.cbm")

medians = pd.read_csv("circuit_medians.csv")
real_2025 = pd.read_csv("real_lap_time_2025.csv")

class QualiRequest(BaseModel):
    driver: str
    team: str
    event: str
    quali_segment: str = "Q3"
    compound: str = "SOFT"
    session: str = "Q"

@app.post("/predict")
def predict_quali(req: QualiRequest):

    driver_code = DRIVER_NAME_TO_CODE.get(req.driver)
    if not driver_code:
        return {"error": "Unknown driver"}

    row = medians[medians["Event"] == req.event]
    if row.empty:
        return {"error": "Event not found"}

    row = row.iloc[0]

    input_data = {
        "Driver": driver_code,
        "Team": req.team,
        "Compound": req.compound,
        "Event": req.event,
        "Session": req.session,
        "QualiSegment": req.quali_segment,

        "CircuitName": row["CircuitName"],
        "Country": row["Country"],
        "TrackType": row["TrackType"],
        "LapSpeedClass": row["LapSpeedClass"],

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

    X = pd.DataFrame([input_data], columns=EXPECTED_FEATURES)

    predicted_delta = float(model.predict(X)[0])


    session_median = row["SessionMedianLap"]
    predicted_lap_time = session_median + predicted_delta

    real_row = real_2025[
        (real_2025["driver"] == driver_code) &
        (real_2025["race"] == req.event)
    ]

    real_time = (
        float(real_row.iloc[0]["real_time_seconds"])
        if not real_row.empty
        else None
    )

    delta = (
        round(predicted_lap_time - real_time, 3)
        if real_time is not None
        else None
    )

    return {
        "driver": driver_code,
        "event": req.event,
        "predicted_lap_time_sec": round(predicted_lap_time, 3),
        "real_lap_time_sec": round(real_time, 3) if real_time is not None else None,
        "delta_sec": delta
    }
