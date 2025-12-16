from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def load_data():
    quali = pd.read_csv(DATA_DIR / "processed_quali_2024.csv")
    drivers = pd.read_csv(DATA_DIR / "driver_team_mapping.csv")

    # merge driver names
    df = quali.merge(drivers, on="driver_number", how="left")

    return df


def predict_quali(race_number: int, driver: str | None = None):
    """
    race_number -> meeting_key
    driver -> partial name match (e.g. 'Hamilton')
    """

    df = load_data()

    # filter race
    race_df = df[df["meeting_key"] == race_number].copy()

    # actual best lap per driver
    result = (
        race_df.groupby(["driver_number", "driver_name", "team"])
        .agg(actual_lap=("lap_duration", "min"))
        .reset_index()
    )

    # VERY SIMPLE prediction baseline (mean improvement)
    result["predicted_lap"] = result["actual_lap"] * 0.995

    # predicted position
    result = result.sort_values("predicted_lap").reset_index(drop=True)
    result["predicted_position"] = result.index + 1

    # driver filter
    if driver:
        result = result[
            result["driver_name"]
            .str.contains(driver, case=False, na=False)
        ]

    return result


if __name__ == "__main__":
    output = predict_quali(race_number=5, driver="Hamilton")
    print(output)
