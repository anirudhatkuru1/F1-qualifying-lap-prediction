from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from catboost import CatBoostRegressor

# ----------------------------
# Driver name → 3-letter code
# ----------------------------
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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "F1 Qualifying Prediction API is running 🚀"}

# ----------------------------
# Load artifacts ONCE
# ----------------------------
model = CatBoostRegressor()
model.load_model("quali_model.cbm")

medians = pd.read_csv("circuit_medians.csv")
real_2025 = pd.read_csv("real_lap_time_2025.csv")

# ----------------------------
# Request schema
# ----------------------------
class QualiRequest(BaseModel):
    driver: str
    team: str
    event: str
    quali_segment: str
    compound: str = "SOFT"
    session: str = "Q"

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict_quali(req: QualiRequest):

    # Driver mapping
    driver_code = DRIVER_NAME_TO_CODE.get(req.driver)
    if driver_code is None:
        return {"error": "Unknown driver"}

    # Circuit medians
    row = medians[medians["Event"] == req.event]
    if row.empty:
        return {"error": "Event not found"}
    row = row.iloc[0]

    # Model input
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

    X = pd.DataFrame([input_data])
    predicted = float(model.predict(X)[0])

    # ----------------------------
    # 2025 real time lookup
    # ----------------------------
    real_row = real_2025[
        (real_2025["driver"] == driver_code) &
        (real_2025["race"] == req.event)
    ]

    real_time = (
        float(real_row.iloc[0]["real_time_seconds"])
        if not real_row.empty
        else None
    )

    # Delta
    delta = round(predicted - real_time, 3) if real_time is not None else None

    return {
        "driver": driver_code,
        "event": req.event,
        "predicted_lap_time_sec": round(predicted, 3),
        "real_lap_time_sec": round(real_time, 3) if real_time else None,
        "delta_sec": delta
    }
