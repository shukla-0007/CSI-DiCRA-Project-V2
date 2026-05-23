from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Security, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from src.database.db import get_db, RiskPredictions, DistrictBaselines, APILog, log_api_status
from src.features.phenology_clipper import SUPPORTED_CROPS
import os
import secrets
import uvicorn

# ── API KEY SECURITY SETUP ───────────────────────────────────────────────────
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_or_create_api_key() -> str:
    """Retrieve API key from environment variable or read/generate from .csi_api_key file."""
    key = os.environ.get("CSI_API_KEY", "").strip()
    if key:
        return key

    key_file = ".csi_api_key"
    if os.path.exists(key_file):
        try:
            with open(key_file, "r") as f:
                val = f.read().strip()
                if val:
                    return val
        except Exception as e:
            print(f"⚠️ Error reading {key_file}: {e}")

    new_key = secrets.token_hex(32)
    try:
        with open(key_file, "w") as f:
            f.write(new_key)
        print(f"🔒 Generated new secure API key and wrote to {key_file}")
        return new_key
    except Exception as e:
        print(f"⚠️ Could not write {key_file}: {e}. Using transient key.")
        return new_key

CSI_API_KEY = get_or_create_api_key()

def validate_api_key(header_value: str = Security(api_key_header)):
    if not CSI_API_KEY:
        return
    if header_value == CSI_API_KEY:
        return header_value
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid or missing API Key. Add 'X-API-KEY' header."
    )

# ── APP INITIALIZATION & CORS ─────────────────────────────────────────────────
app = FastAPI(
    title       = "CSI-DiCRA Compound Stress Index API",
    description = "Actuarial climate risk tiers for agricultural insurance — Madhya Pradesh.",
    version     = "2.1.0",
    dependencies = [Depends(validate_api_key)],
)

allowed_origins = os.environ.get("CSI_ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalise_scenario(scenario: str) -> str:
    """Accept legacy RCP names alongside current SSP names."""
    return scenario.replace("RCP_4.5", "SSP2_4.5").replace("RCP_8.5", "SSP5_8.5")


@app.get("/api/v1/insurance/premium_risk")
def get_premium_risk(
    district: str,
    year:     int,
    crop:     str = "Soybean",
    scenario: str = "SSP5_8.5",
    db: Session   = Depends(get_db),
):
    """
    For insurance companies: returns the CSI risk tier and recommended
    premium adjustment for a given district / year / crop / scenario.
    """
    if crop not in SUPPORTED_CROPS:
        raise HTTPException(
            status_code=400,
            detail=f"Crop '{crop}' not supported. Choose from: {SUPPORTED_CROPS}",
        )
    scenario   = _normalise_scenario(scenario)
    prediction = db.query(RiskPredictions).filter(
        RiskPredictions.district == district,
        RiskPredictions.year     == year,
        RiskPredictions.scenario == scenario,
        RiskPredictions.crop     == crop,
    ).first()

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail=f"No projection found for {district} / {year} / {scenario} / {crop}.",
        )

    adj = (
        f"+{(prediction.risk_tier - 3) * 1.5}%"
        if prediction.risk_tier > 3
        else "Base Rate"
    )
    return {
        "district":                    prediction.district,
        "year":                        prediction.year,
        "scenario":                    prediction.scenario,
        "crop":                        prediction.crop,
        "csi_risk_tier":               prediction.risk_tier,
        "probabilistic_score":         round(prediction.csi_score, 3),
        "recommended_premium_adjustment": adj,
    }


@app.get("/api/v1/dashboard/trajectory")
def get_risk_trajectory(
    district: str,
    crop:     str = "Soybean",
    scenario: str = "SSP5_8.5",
    db: Session   = Depends(get_db),
):
    """
    For the Streamlit dashboard: returns the 15-year risk trajectory
    (2026-2041) for a given district / crop / scenario.
    """
    if crop not in SUPPORTED_CROPS:
        raise HTTPException(
            status_code=400,
            detail=f"Crop '{crop}' not supported. Choose from: {SUPPORTED_CROPS}",
        )
    scenario    = _normalise_scenario(scenario)
    predictions = db.query(RiskPredictions).filter(
        RiskPredictions.district == district,
        RiskPredictions.scenario == scenario,
        RiskPredictions.crop     == crop,
    ).order_by(RiskPredictions.year).all()

    return [
        {"year": p.year, "risk_tier": p.risk_tier, "score": p.csi_score}
        for p in predictions
    ]


@app.get("/api/v1/dashboard/map_data")
def get_map_data(
    crop: str = "Soybean",
    year: int = 2026,
    scenario: str = "SSP5_8.5",
    db: Session = Depends(get_db)
):
    """
    Returns bulk climate and risk metrics for all 52 districts under the
    selected crop, year, and climate scenario. Useful for mapping.
    """
    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    from src.database.db import DailyClimate
    from src.features.phenology_clipper import CROP_CALENDARS
    from src.pipeline.dicra_fetcher import MP_DISTRICTS

    if crop not in SUPPORTED_CROPS:
        raise HTTPException(
            status_code=400,
            detail=f"Crop '{crop}' not supported. Choose from: {SUPPORTED_CROPS}",
        )
    scenario = _normalise_scenario(scenario)

    # 1. Pre-fetch predictions for the year/scenario/crop in one query
    preds = db.query(RiskPredictions).filter(
        RiskPredictions.year == year,
        RiskPredictions.crop == crop,
        RiskPredictions.scenario == scenario
    ).all()
    pred_map = {p.district: p for p in preds}

    # 2. Pre-fetch baselines
    baselines = db.query(DistrictBaselines).filter(
        DistrictBaselines.crop == crop
    ).all()
    baseline_map = {b.district: b for b in baselines}

    cal = CROP_CALENDARS[crop]
    start_year = year + cal["start_year_offset"]
    window_start = pd.Timestamp(f"{start_year}-{cal['critical_start']}")
    window_end   = pd.Timestamp(f"{year}-{cal['critical_end']}")

    # Fetch historical observed precipitation for all districts at once to optimize DB roundtrips
    all_hist = db.query(DailyClimate.district, DailyClimate.date, DailyClimate.precipitation).order_by(DailyClimate.date).all()
    hist_by_district = {}
    for d, dt, p in all_hist:
        if d not in hist_by_district:
            hist_by_district[d] = []
        hist_by_district[d].append((dt, p))

    results = {}
    for district in MP_DISTRICTS:
        district_slug = district.replace(" ", "_").replace("-", "_").lower()
        cache_path = os.path.join("data", "processed", f"cmip6_{district_slug}_{scenario}.csv")

        if not os.path.exists(cache_path):
            continue

        try:
            # Read cached daily climate forecast
            df = pd.read_csv(cache_path)
            df["date"] = pd.to_datetime(df["date"])

            # Slice to critical phenological window
            window_df = df[(df["date"] >= window_start) & (df["date"] <= window_end)]
            if window_df.empty:
                continue

            mean_tmax = window_df["tmax"].mean()
            mean_sm = window_df["soil_moisture"].mean()
            mean_precip = window_df["precipitation"].mean()

            # Approximate NDVI using standard formulas
            if cal["season"] == "Kharif":
                ndvi_approx = float(np.clip(0.30 + 0.004 * mean_precip, 0.20, 0.80))
            else:
                ndvi_approx = float(np.clip(0.40 + 0.002 * mean_precip, 0.30, 0.70))

            # Fetch historical observed precipitation from batch dictionary for combined SPI-3
            hist_records = hist_by_district.get(district, [])

            if hist_records:
                hist_df = pd.DataFrame(hist_records, columns=["date", "precipitation"])
                hist_df["date"] = pd.to_datetime(hist_df["date"])
                combined_df = pd.concat([hist_df, df[["date", "precipitation"]]], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=["date"]).sort_values("date").set_index("date")
            else:
                combined_df = df[["date", "precipitation"]].copy().set_index("date")

            # Compute SPI-3 on the combined series
            combined_df["precip_90"] = combined_df["precipitation"].rolling(window=90, min_periods=1).sum()
            spi3_series = pd.Series(index=combined_df.index, dtype=float)

            for month in range(1, 13):
                month_mask = combined_df.index.month == month
                month_data = combined_df.loc[month_mask, "precip_90"]
                if month_data.empty:
                    continue

                N = len(month_data)
                zeros_mask = month_data <= 0.001
                m = zeros_mask.sum()
                q = m / N

                non_zero_data = month_data[~zeros_mask]

                if len(non_zero_data) >= 3:
                    try:
                        shape, loc, scale = stats.gamma.fit(non_zero_data, floc=0)
                        g_cdf = stats.gamma.cdf(month_data, shape, loc=loc, scale=scale)
                        h = q + (1 - q) * g_cdf
                    except Exception:
                        h = month_data.rank(method='average') / (N + 1)
                else:
                    h = np.zeros(N)
                    h[zeros_mask] = q
                    if N - m > 0:
                        h[~zeros_mask] = 0.5 + 0.5 * (month_data[~zeros_mask].rank() / (N - m + 1))

                h = np.clip(h, 0.00001, 0.99999)
                month_spi = stats.norm.ppf(h)
                spi3_series.loc[month_mask] = month_spi

            window_spi3 = spi3_series.loc[window_start:window_end].mean()

            # Retrieve predicted risk tier & score
            pred = pred_map.get(district)
            tier = pred.risk_tier if pred else 1
            score = pred.csi_score if pred else 0.0
            recommended_premium = (
                f"+{(tier - 3) * 1.5}%"
                if tier > 3
                else "Base Rate"
            )

            # Calculate SHAP explainability
            from src.model.explainability import get_shap_explanation
            baseline_rec = baseline_map.get(district)
            tmax_95th = baseline_rec.tmax_95th if baseline_rec else 35.0
            sm_5th = baseline_rec.sm_5th if baseline_rec else 0.08
            
            feat_arr = np.array([[
                float(mean_tmax),
                float(mean_sm),
                float(ndvi_approx),
                float(window_spi3) if pd.notna(window_spi3) else 0.0,
                float(mean_tmax - tmax_95th),
                float(mean_sm - sm_5th)
            ]])
            shap_result = get_shap_explanation(crop, feat_arr)

            # Round output variables
            results[district] = {
                "NDVI": round(float(ndvi_approx), 3),
                "Tmax": round(float(mean_tmax), 1),
                "Soil Moisture": round(float(mean_sm), 3),
                "SPI-3": round(float(window_spi3), 3) if pd.notna(window_spi3) else 0.0,
                "CSI Risk Tier": int(tier),
                "CSI Score": round(float(score), 3),
                "Recommended Premium Adjustment": recommended_premium,
                "explainability": shap_result
            }
        except Exception as e:
            # Log error and skip district
            print(f"⚠️ Failed to extract map data for {district}: {e}")
            continue

    return results


@app.get("/api/v1/baselines")
def get_baselines(
    crop: str | None = None,
    db:   Session    = Depends(get_db),
):
    """
    Returns season-aware baselines for all districts.
    Optionally filter by crop (e.g. ?crop=Wheat).
    """
    query = db.query(DistrictBaselines)
    if crop:
        if crop not in SUPPORTED_CROPS:
            raise HTTPException(
                status_code=400,
                detail=f"Crop '{crop}' not supported. Choose from: {SUPPORTED_CROPS}",
            )
        query = query.filter(DistrictBaselines.crop == crop)

    return [
        {
            "district":  b.district,
            "crop":      b.crop,
            "tmax_95th": b.tmax_95th,
            "sm_5th":    b.sm_5th,
        }
        for b in query.all()
    ]


@app.get("/api/v1/crops")
def list_supported_crops():
    """Returns the list of crops the system supports."""
    return {"supported_crops": SUPPORTED_CROPS}


@app.get("/api/v1/health/api_status")
def get_api_status(db: Session = Depends(get_db)):
    """
    Returns the latest 50 API logs from the system for developer diagnostics,
    including successful calls and fallback events.
    """
    logs = db.query(APILog).order_by(APILog.timestamp.desc()).limit(50).all()
    return [
        {
            "id":        log.id,
            "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S") if log.timestamp else None,
            "service":   log.service,
            "district":  log.district,
            "status":    log.status,
            "message":   log.message,
        }
        for log in logs
    ]


import threading
import time

def start_scheduler():
    def run_periodically():
        # Sleep for a few seconds to let FastAPI spin up fully
        time.sleep(5)
        print("⏰ Scheduler: Running initial startup climate sync and forecast regeneration...")
        try:
            from src.pipeline.initialize_data import run_pipeline
            run_pipeline(incremental=True, safety_days=15)
        except Exception as e:
            print(f"❌ Scheduler startup run failed: {e}")

        while True:
            # Sleep for 3 days
            # 3 days = 3 * 24 * 3600 seconds
            time.sleep(3 * 24 * 3600)
            print("⏰ Scheduler: Triggering 3-day periodic climate sync and forecast regeneration...")
            try:
                from src.pipeline.initialize_data import run_pipeline
                run_pipeline(incremental=True, safety_days=15)
            except Exception as e:
                print(f"❌ Scheduler periodic run failed: {e}")

    thread = threading.Thread(target=run_periodically, daemon=True)
    thread.start()


@app.on_event("startup")
def startup_event():
    start_scheduler()


@app.post("/api/v1/pipeline/sync")
def trigger_pipeline_sync(
    background_tasks: BackgroundTasks,
    incremental: bool = True,
    safety_days: int = 15,
):
    """
    Trigger the live pipeline sync manually in the background.
    """
    from src.pipeline.initialize_data import run_pipeline
    background_tasks.add_task(run_pipeline, incremental=incremental, safety_days=safety_days)
    return {"message": "Pipeline sync started in background."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)