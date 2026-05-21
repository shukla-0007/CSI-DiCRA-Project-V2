import os
import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.database.db import SessionLocal, DistrictBaselines, RiskPredictions
from src.features.phenology_clipper import SUPPORTED_CROPS

# ── Per-crop yield thresholds for Risk Tier assignment ──────────────────────────
# Based on PMFBY / ICAR typical yield ranges for Madhya Pradesh.
# Tiers map: yield < threshold[0] → Tier 5 (Catastrophic)
#            yield < threshold[1] → Tier 4 (Severe)
#            yield < threshold[2] → Tier 3 (Moderate)
#            yield < threshold[3] → Tier 2 (Mild)
#            else                 → Tier 1 (Normal)
# Thresholds roughly correspond to <50%, 50-70%, 70-80%, 80-90%, >90% of
# typical district yield (mid-range of PMFBY data for MP).
CROP_YIELD_THRESHOLDS = {
    "Soybean": [500,  700,  800,  900],   # typical ~1000 kg/ha
    "Rice":    [600,  840,  960, 1080],   # typical ~1200 kg/ha
    "Maize":   [1000, 1400, 1600, 1800],  # typical ~2000 kg/ha
    "Wheat":   [1000, 1400, 1600, 1800],  # typical ~2000 kg/ha
    "Gram":    [500,  700,  800,  900],   # typical ~1000 kg/ha
    "Mustard": [550,  770,  880,  990],   # typical ~1100 kg/ha
}

MP_DISTRICTS = [
    "Agar Malwa", "Alirajpur", "Anuppur", "Ashoknagar", "Balaghat", "Barwani",
    "Betul", "Bhind", "Bhopal", "Burhanpur", "Chhatarpur", "Chhindwara",
    "Damoh", "Datia", "Dewas", "Dhar", "Dindori", "Guna", "Gwalior", "Harda",
    "Hoshangabad", "Indore", "Jabalpur", "Jhabua", "Katni", "Khandwa",
    "Khargone", "Mandla", "Mandsaur", "Morena", "Narsinghpur", "Neemuch",
    "Niwari", "Panna", "Raisen", "Rajgarh", "Ratlam", "Rewa", "Sagar",
    "Satna", "Sehore", "Seoni", "Shahdol", "Shajapur", "Sheopur", "Shivpuri",
    "Sidhi", "Singrauli", "Tikamgarh", "Ujjain", "Umaria", "Vidisha",
]


def yield_to_tier(predicted_yield: float, crop: str) -> int:
    """Map a predicted yield (kg/ha) to a 1-5 Risk Tier using crop thresholds."""
    thresholds = CROP_YIELD_THRESHOLDS[crop]
    if predicted_yield < thresholds[0]: return 5
    if predicted_yield < thresholds[1]: return 4
    if predicted_yield < thresholds[2]: return 3
    if predicted_yield < thresholds[3]: return 2
    return 1


def load_cmip6_mock_data(scenario: str, crop: str) -> pd.DataFrame:
    """
    Mock CMIP6 data for 2026-2041.  In production this reads a per-crop
    downscaled NetCDF from IITM Pune for the given SSP scenario.
    Warming multiplier is slightly higher for SSP5_8.5.
    """
    years      = list(range(2026, 2042))
    multiplier = 1.5 if scenario == "SSP5_8.5" else 1.1

    rows = []
    for district in MP_DISTRICTS:
        for year in years:
            base_events  = 5
            future_events= int(base_events * multiplier * (year - 2025) / 5)
            rows.append({
                "district":       district,
                "year":           year,
                "compound_events": max(0, min(20, future_events)),
                "mean_ndvi":       np.random.uniform(0.25, 0.70),
            })
    return pd.DataFrame(rows)


def generate_forecasts():
    """
    For every combination of (scenario, crop), load the crop-specific model,
    pull season-aware baselines, and write RiskPredictions rows.

    Total rows: 2 scenarios × 6 crops × 52 districts × 16 years = 9,984 rows.
    """
    db = SessionLocal()

    for scenario in ["SSP2_4.5", "SSP5_8.5"]:
        for crop in SUPPORTED_CROPS:
            model_path = f"models/xgboost_{crop.lower()}_model.pkl"
            if not os.path.exists(model_path):
                print(f"⚠️  Model not found for {crop} ({model_path}). "
                      f"Run train_all_models() first. Skipping.")
                continue

            model    = joblib.load(model_path)
            cmip6_df = load_cmip6_mock_data(scenario, crop)
            print(f"🔮 Generating: {scenario} / {crop} ({len(cmip6_df)} rows)...")

            for _, row in cmip6_df.iterrows():
                baseline = db.query(DistrictBaselines).filter(
                    DistrictBaselines.district == row["district"],
                    DistrictBaselines.crop     == crop,
                ).first()
                if not baseline:
                    continue

                features = np.array([[
                    row["compound_events"],
                    row["mean_ndvi"],
                    baseline.tmax_95th,
                    baseline.sm_5th,
                ]])

                predicted_yield = float(model.predict(features)[0])
                tier            = yield_to_tier(predicted_yield, crop)
                csi_score       = round(tier / 5.0, 3)

                existing = db.query(RiskPredictions).filter(
                    RiskPredictions.district == row["district"],
                    RiskPredictions.year     == row["year"],
                    RiskPredictions.scenario == scenario,
                    RiskPredictions.crop     == crop,
                ).first()

                if existing:
                    existing.risk_tier = tier
                    existing.csi_score = csi_score
                else:
                    db.add(RiskPredictions(
                        district  = row["district"],
                        year      = row["year"],
                        scenario  = scenario,
                        crop      = crop,
                        csi_score = csi_score,
                        risk_tier = tier,
                    ))

            db.commit()
            print(f"  ✅ {scenario} / {crop} saved.")

    db.close()
    print("✅ All forecasts generated.")


if __name__ == "__main__":
    generate_forecasts()
