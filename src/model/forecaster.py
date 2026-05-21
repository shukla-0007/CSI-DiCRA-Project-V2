"""
forecaster.py
=============
Generates CSI risk tier predictions for 2026–2041 using real CMIP6 climate
projections from the Open-Meteo Climate API.

CMIP6 Model Mapping (Open-Meteo HighResMIP ensemble):
    SSP2_4.5  →  EC_Earth3P_HR   (ECMWF high-res — moderate warming scenario)
    SSP5_8.5  →  MRI_AGCM3_2_S  (Japan MRI high-res — high warming scenario,
                                   best spatial resolution for South Asia)

Both models provide:
    - temperature_2m_max         (daily Tmax in °C)
    - precipitation_sum          (daily total precipitation in mm)
    - soil_moisture_0_to_10cm_mean (volumetric m³/m³)

API endpoint: https://climate-api.open-meteo.com/v1/climate
No API key required. Free for non-commercial use.
"""

import os
import time
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session

from src.database.db import SessionLocal, DistrictBaselines, RiskPredictions, log_api_status
from src.features.phenology_clipper import SUPPORTED_CROPS, CROP_CALENDARS, get_season_months

# ── Forecast window ────────────────────────────────────────────────────────────
FORECAST_START_YEAR = 2026
FORECAST_END_YEAR   = 2041   # inclusive

# ── CMIP6 model ↔ SSP scenario mapping ────────────────────────────────────────
SCENARIO_MODELS = {
    "SSP2_4.5": "EC_Earth3P_HR",   # ECMWF high-res — moderate warming
    "SSP5_8.5": "MRI_AGCM3_2_S",  # MRI high-res — high warming, best for Asia
}

# ── Open-Meteo Climate API ─────────────────────────────────────────────────────
OPEN_METEO_CLIMATE_URL = "https://climate-api.open-meteo.com/v1/climate"

# ── All 52 MP districts ────────────────────────────────────────────────────────
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

# ── Per-crop yield thresholds for Risk Tier assignment ──────────────────────────
# Based on PMFBY / ICAR typical yield ranges for Madhya Pradesh.
# Tiers: yield < threshold[0] → Tier 5 (Catastrophic)
#        yield < threshold[1] → Tier 4 (Severe)
#        yield < threshold[2] → Tier 3 (Moderate)
#        yield < threshold[3] → Tier 2 (Mild)
#        else                 → Tier 1 (Normal)
CROP_YIELD_THRESHOLDS = {
    "Soybean": [500,  700,  800,  900],   # typical ~1000 kg/ha
    "Rice":    [600,  840,  960, 1080],   # typical ~1200 kg/ha
    "Maize":   [1000, 1400, 1600, 1800],  # typical ~2000 kg/ha
    "Wheat":   [1000, 1400, 1600, 1800],  # typical ~2000 kg/ha
    "Gram":    [500,  700,  800,  900],   # typical ~1000 kg/ha
    "Mustard": [550,  770,  880,  990],   # typical ~1100 kg/ha
}


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def yield_to_tier(predicted_yield: float, crop: str) -> int:
    """Map a predicted yield (kg/ha) to a 1–5 Risk Tier."""
    t = CROP_YIELD_THRESHOLDS[crop]
    if predicted_yield < t[0]: return 5
    if predicted_yield < t[1]: return 4
    if predicted_yield < t[2]: return 3
    if predicted_yield < t[3]: return 2
    return 1


def _get_district_coordinates(district_name: str):
    """
    Returns (lat, lon) for an MP district.
    Delegates to dicra_fetcher to avoid code duplication.
    """
    from src.pipeline.dicra_fetcher import get_district_coordinates
    return get_district_coordinates(district_name)


# ══════════════════════════════════════════════════════════════════════════════
#  OPEN-METEO CMIP6 FETCHER
# ══════════════════════════════════════════════════════════════════════════════

def fetch_cmip6_district(
    district: str,
    scenario: str,
    start_year: int = FORECAST_START_YEAR,
    end_year:   int = FORECAST_END_YEAR,
) -> pd.DataFrame:
    """
    Fetches daily CMIP6 climate projections from Open-Meteo for one MP district.

    Returns a DataFrame with columns:
        date, tmax, precipitation, soil_moisture

    Falls back to physics-guided simulation if the API call fails.
    """
    model = SCENARIO_MODELS[scenario]
    lat, lon = _get_district_coordinates(district)
    start_str = f"{start_year}-01-01"
    end_str   = f"{end_year}-12-31"

    def _db_log(status: str, service: str, message: str):
        db = SessionLocal()
        try:
            log_api_status(db, service, district, status, message)
        except Exception as e:
            print(f"Logging error: {e}")
        finally:
            db.close()


    for attempt in range(2):   # up to 2 attempts (1 retry)
        try:
            resp = requests.get(
                OPEN_METEO_CLIMATE_URL,
                params={
                    "latitude":   lat,
                    "longitude":  lon,
                    "start_date": start_str,
                    "end_date":   end_str,
                    "models":     model,
                    "daily":      "temperature_2m_max,precipitation_sum,soil_moisture_0_to_10cm_mean",
                },
                timeout=60,
            )

            # Rate limit — wait and retry
            if resp.status_code == 429:
                wait = 5
                print(f"    ⏳ [{district}] Open-Meteo rate limit. Waiting {wait}s (retry {attempt+1}/1)...")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}: {resp.text[:200]}")

            body = resp.json()
            if "daily" not in body:
                raise ValueError(f"'daily' key missing: {list(body.keys())}")

            daily = body["daily"]
            dates = pd.to_datetime(daily["time"])
            tmax  = daily["temperature_2m_max"]
            prec  = daily["precipitation_sum"]
            sm    = daily["soil_moisture_0_to_10cm_mean"]

            df = pd.DataFrame({
                "date":          dates,
                "tmax":          [float(v) if v is not None else np.nan for v in tmax],
                "precipitation": [float(v) if v is not None else np.nan for v in prec],
                "soil_moisture": [float(v) if v is not None else np.nan for v in sm],
            })

            # Forward-fill any isolated nulls (rare in this API)
            df["tmax"]          = df["tmax"].interpolate(method="linear").bfill().ffill()
            df["precipitation"] = df["precipitation"].fillna(0.0)
            df["soil_moisture"] = df["soil_moisture"].interpolate(method="linear").bfill().ffill()

            _db_log("SUCCESS", "Open-Meteo CMIP6 API", f"Successfully fetched daily CMIP6 forecast for scenario {scenario}.")
            return df

        except Exception as e:
            if attempt < 1:
                continue
            print(f"    ⚠️  [{district}] Open-Meteo API failed ({e}). Using physics-guided fallback.")
            _db_log("FALLBACK", "Open-Meteo CMIP6 API", f"API failed: {e}. Reverting to physics-guided CMIP6 simulation.")
            return _physics_guided_fallback_df(district, scenario, start_year, end_year)

    # All retries exhausted
    _db_log("FALLBACK", "Open-Meteo CMIP6 API", "API retries exhausted. Reverting to physics-guided CMIP6 simulation.")
    return _physics_guided_fallback_df(district, scenario, start_year, end_year)


def _physics_guided_fallback_df(
    district: str,
    scenario: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Fallback: generates daily climate projections by applying published IPCC
    warming rates to climatological normals for MP.

    Warming rates (IPCC AR6, India regional):
        SSP2_4.5: +0.05°C/year Tmax,  -0.001/year soil moisture
        SSP5_8.5: +0.10°C/year Tmax,  -0.002/year soil moisture
    """
    dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq="D")
    n = len(dates)
    months = np.array([d.month for d in dates])
    years  = np.array([d.year  for d in dates])

    # Climatological baseline (MP normals)
    tmax_base = 30.0 + 11.0 * np.sin(np.pi * (months - 2) / 6)
    sm_base   = 0.18 + 0.10 * np.sin(np.pi * (months - 3) / 5)
    prec_base = np.where(
        np.isin(months, [6, 7, 8, 9]),
        np.random.exponential(8.0, n),
        np.random.exponential(0.5, n),
    )

    # Apply warming trend from 2025 baseline
    years_diff = years - 2025
    if scenario == "SSP2_4.5":
        tmax_trend = 0.05 * years_diff
        sm_trend   = -0.001 * years_diff
    else:  # SSP5_8.5
        tmax_trend = 0.10 * years_diff
        sm_trend   = -0.002 * years_diff

    tmax = np.clip(tmax_base + tmax_trend + np.random.normal(0, 1.2, n), 15.0, 50.0)
    sm   = np.clip(sm_base + sm_trend + np.random.normal(0, 0.015, n), 0.02, 0.50)

    return pd.DataFrame({
        "date":          dates,
        "tmax":          tmax,
        "precipitation": np.clip(prec_base, 0.0, None),
        "soil_moisture": sm,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION (CMIP6 → XGBoost features)
# ══════════════════════════════════════════════════════════════════════════════

def extract_crop_features(
    cmip6_df: pd.DataFrame,
    baseline: DistrictBaselines,
    district: str,
    year: int,
    crop: str,
) -> dict | None:
    """
    Slices the CMIP6 daily DataFrame to the crop's critical phenological window
    and extracts XGBoost features:
        mean_tmax, mean_sm, mean_ndvi, tot_precip, tmax_anomaly, sm_anomaly

    Returns None if no data falls in the window.
    """
    cal        = CROP_CALENDARS[crop]
    start_year = year + cal["start_year_offset"]

    try:
        window_start = pd.Timestamp(f"{start_year}-{cal['critical_start']}")
        window_end   = pd.Timestamp(f"{year}-{cal['critical_end']}")
    except Exception:
        return None

    window = cmip6_df[
        (cmip6_df["date"] >= window_start) &
        (cmip6_df["date"] <= window_end)
    ].copy()

    if window.empty:
        return None

    mean_tmax = window["tmax"].mean()
    mean_sm = window["soil_moisture"].mean()
    tot_precip = window["precipitation"].sum()

    # Handle NaNs if any (e.g. interpolate/fill)
    mean_tmax = float(mean_tmax) if pd.notna(mean_tmax) else 30.0
    mean_sm = float(mean_sm) if pd.notna(mean_sm) else 0.15
    tot_precip = float(tot_precip) if pd.notna(tot_precip) else 0.0

    # Approximate NDVI from mean precipitation during the window
    mean_precip = window["precipitation"].mean()
    mean_precip = float(mean_precip) if pd.notna(mean_precip) else 0.0
    season = cal["season"]
    if season == "Kharif":
        ndvi_approx = float(np.clip(0.30 + 0.004 * mean_precip, 0.20, 0.80))
    else:
        ndvi_approx = float(np.clip(0.40 + 0.002 * mean_precip, 0.30, 0.70))

    tmax_anomaly = mean_tmax - baseline.tmax_95th
    sm_anomaly = mean_sm - baseline.sm_5th

    return {
        "mean_tmax": mean_tmax,
        "mean_sm": mean_sm,
        "mean_ndvi": ndvi_approx,
        "tot_precip": tot_precip,
        "tmax_anomaly": tmax_anomaly,
        "sm_anomaly": sm_anomaly,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN FORECAST GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_forecasts():
    """
    For every (scenario, crop, district, year) combination in 2026–2041:
        1. Fetch CMIP6 daily projections from Open-Meteo (cached per scenario×district).
        2. Slice to the crop's phenological window.
        3. Compute compound heat+drought events.
        4. Run through the trained XGBoost model.
        5. Assign a Risk Tier (1–5) and CSI score (0.0–1.0).
        6. Upsert into RiskPredictions.
    """
    db = SessionLocal()
    years = list(range(FORECAST_START_YEAR, FORECAST_END_YEAR + 1))

    print(f"\n🔮 Generating CMIP6 forecasts ({FORECAST_START_YEAR}–{FORECAST_END_YEAR})")
    print(f"   Scenarios : {list(SCENARIO_MODELS.keys())}")
    print(f"   Crops     : {SUPPORTED_CROPS}")
    print(f"   Districts : {len(MP_DISTRICTS)}")
    print(f"   Years     : {len(years)}")
    print(f"   Total rows: {len(SCENARIO_MODELS) * len(SUPPORTED_CROPS) * len(MP_DISTRICTS) * len(years):,}\n")

    for scenario in SCENARIO_MODELS:
        model_name = SCENARIO_MODELS[scenario]
        print(f"\n{'='*60}")
        print(f"  📡 Scenario: {scenario}  (CMIP6 model: {model_name})")
        print(f"{'='*60}")

        # Cache for CMIP6 dataframes to avoid fetching the same district data multiple times for different crops
        cmip6_cache = {}

        for crop in SUPPORTED_CROPS:
            model_path = f"models/xgboost_{crop.lower()}_model.pkl"
            if not os.path.exists(model_path):
                print(f"  ⚠️  [{crop}] Model not found at {model_path}. Run train_all_models() first.")
                continue

            xgb_model = joblib.load(model_path)
            print(f"\n  🌾 [{crop}] Loaded model — processing {len(MP_DISTRICTS)} districts...")

            saved = 0
            for dist_idx, district in enumerate(MP_DISTRICTS, start=1):
                # Skip if prediction already exists for all years for this scenario/crop/district
                existing_count = db.query(RiskPredictions).filter(
                    RiskPredictions.district == district,
                    RiskPredictions.scenario == scenario,
                    RiskPredictions.crop     == crop,
                ).count()
                if existing_count == len(years):
                    print(f"    ⏭️  [{district}] Already has predictions for {crop} / {scenario} (skipping)...")
                    continue

                cache_key = (district, scenario)
                if cache_key not in cmip6_cache:
                    print(f"    📡 [{district}] Fetching CMIP6 daily projections from Open-Meteo...")
                    cmip6_df = fetch_cmip6_district(district, scenario)
                    cmip6_cache[cache_key] = cmip6_df
                    # Polite rate limit — Open-Meteo has 10 req/min limit on free tier
                    time.sleep(8.0)
                else:
                    cmip6_df = cmip6_cache[cache_key]

                if cmip6_df.empty:
                    print(f"    ⚠️  [{district}] No CMIP6 data returned — skipping.")
                    continue

                baseline = db.query(DistrictBaselines).filter(
                    DistrictBaselines.district == district,
                    DistrictBaselines.crop     == crop,
                ).first()
                if not baseline:
                    continue

                for year in years:
                    features_dict = extract_crop_features(cmip6_df, baseline, district, year, crop)
                    if features_dict is None:
                        continue

                    features = np.array([[
                        features_dict["mean_tmax"],
                        features_dict["mean_sm"],
                        features_dict["mean_ndvi"],
                        features_dict["tot_precip"],
                        features_dict["tmax_anomaly"],
                        features_dict["sm_anomaly"],
                    ]])

                    predicted_yield = float(xgb_model.predict(features)[0])
                    tier            = yield_to_tier(predicted_yield, crop)
                    csi_score       = round(tier / 5.0, 3)

                    existing = db.query(RiskPredictions).filter(
                        RiskPredictions.district == district,
                        RiskPredictions.year     == year,
                        RiskPredictions.scenario == scenario,
                        RiskPredictions.crop     == crop,
                    ).first()

                    if existing:
                        existing.risk_tier = tier
                        existing.csi_score = csi_score
                    else:
                        db.add(RiskPredictions(
                            district  = district,
                            year      = year,
                            scenario  = scenario,
                            crop      = crop,
                            csi_score = csi_score,
                            risk_tier = tier,
                        ))
                    saved += 1

                # Commit per district to avoid one giant transaction
                db.commit()

                # Polite rate limit — Open-Meteo is free, don't hammer it
                time.sleep(0.2)

            print(f"  ✅ [{crop} / {scenario}] {saved} risk predictions saved.")

    db.close()
    print(f"\n✅ All CMIP6 forecasts generated for {FORECAST_START_YEAR}–{FORECAST_END_YEAR}.")


if __name__ == "__main__":
    generate_forecasts()
