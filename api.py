import pandas as pd
from catboost import CatBoostRegressor, Pool
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load model & data ONCE
# -----------------------------
model = CatBoostRegressor()
model.load_model("quali_q3_delta_model.cbm")

medians = pd.read_csv("circuit_medians.csv")
driver_stats = pd.read_csv("driver_track_stats.csv")
team_stats = pd.read_csv("team_track_stats.csv")

# -----------------------------
# Feature config (MUST MATCH TRAINING)
# -----------------------------
categorical_features = [
    "Driver", "Team", "Compound", "Event", "Session",
    "QualiSegment", "CircuitName", "Country",
    "TrackType", "LapSpeedClass",
    "Driver_Track", "Team_Track"
]

numeric_features = [
    "TyreLife",
    "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "TrackLength_m", "NumCorners", "CornerDensity",
    "AvgCornerSpacing_m", "AirTemp", "TrackTemp",
    "WindSpeed", "Altitude_m", "DRSZones",
    "DriverTrackAvgDelta", "DriverTrackStdDelta",
    "TeamTrackAvgDelta", "TeamTrackStdDelta"
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
# Metadata endpoint
# -----------------------------
@app.get("/metadata")
def get_metadata():
    return {
        "drivers": sorted(driver_stats["Driver"].unique().tolist()),
        "teams": sorted(team_stats["Team"].unique().tolist()),
        "events": sorted(medians["Event"].unique().tolist()),
        "quali_segments": ["Q1", "Q2", "Q3"]
    }

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(req: PredictRequest):

    row_df = medians[
        (medians["Event"] == req.event) &
        (medians["QualiSegment"] == req.quali_segment)
    ]

    if row_df.empty:
        return {"error": "No median data for this event / segment"}

    row = row_df.iloc[0]

    driver_track = f"{req.driver}_{row['CircuitName']}"
    team_track   = f"{req.team}_{row['CircuitName']}"

    drow = driver_stats[driver_stats["Driver_Track"] == driver_track]
    trow = team_stats[team_stats["Team_Track"] == team_track]

    input_data = {
        "Driver": req.driver,
        "Team": req.team,
        "Compound": "SOFT",
        "Event": req.event,
        "Session": "Q",
        "QualiSegment": req.quali_segment,

        "CircuitName": row["CircuitName"],
        "Country": row["Country"],
        "TrackType": row["TrackType"],
        "LapSpeedClass": row["LapSpeedClass"],

        "Driver_Track": driver_track,
        "Team_Track": team_track,

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

        "DriverTrackAvgDelta": float(drow["DriverTrackAvgDelta"].iloc[0]) if not drow.empty else 0.0,
        "DriverTrackStdDelta": float(drow["DriverTrackStdDelta"].iloc[0]) if not drow.empty else 0.15,

        "TeamTrackAvgDelta": float(trow["TeamTrackAvgDelta"].iloc[0]) if not trow.empty else 0.0,
        "TeamTrackStdDelta": float(trow["TeamTrackStdDelta"].iloc[0]) if not trow.empty else 0.15,
    }

    X = pd.DataFrame([input_data])[features]
    pool = Pool(X, cat_features=categorical_features)

    delta = model.predict(pool)[0]
    lap_time = row["SessionMedianLap"] + delta

    return {"predicted_lap_time_sec": round(lap_time, 3)}
