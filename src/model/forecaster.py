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


def generate_physically_guided_fallback(db: Session, district: str, year: int, crop: str, scenario: str) -> dict:
    """
    Simulates future climate projections by shifting the historical baseline data
    for the crop's critical window by projected trend factors:
    - SSP2-4.5: +0.05°C/year warming, -0.001/year soil moisture decline.
    - SSP5-8.5: +0.10°C/year warming, -0.002/year soil moisture decline.
    """
    from src.database.db import DailyClimate
    from src.features.phenology_clipper import get_season_months
    
    baseline = db.query(DistrictBaselines).filter(
        DistrictBaselines.district == district,
        DistrictBaselines.crop     == crop
    ).first()
    
    baseline_tmax = baseline.tmax_95th if baseline else 35.0
    baseline_sm = baseline.sm_5th if baseline else 0.15

    # Fetch historical daily climate records to model crop critical window conditions
    hist_records = db.query(DailyClimate).filter(
        DailyClimate.district == district
    ).all()
    
    if not hist_records:
        # Randomized generation with trends if database is empty
        years_diff = year - 2025
        t_warming = 0.05 * years_diff if scenario == "SSP2_4.5" else 0.10 * years_diff
        sm_drying = 0.001 * years_diff if scenario == "SSP2_4.5" else 0.002 * years_diff
        
        sim_tmax = np.random.uniform(28.0 + t_warming, 42.0 + t_warming, 30)
        sim_sm = np.random.uniform(max(0.01, 0.15 - sm_drying - 0.05), max(0.02, 0.15 - sm_drying + 0.05), 30)
        
        comp_days = sum(1 for t, sm in zip(sim_tmax, sim_sm) if t > baseline_tmax and sm < baseline_sm)
        mean_ndvi = np.random.uniform(0.35, 0.65)
        return {"compound_events": comp_days, "mean_ndvi": mean_ndvi}

    # Filter to critical window months
    season_months = get_season_months(crop)
    window_data = [
        r for r in hist_records 
        if r.date.month in season_months and r.tmax is not None and r.soil_moisture is not None
    ]
    
    if not window_data:
        window_data = hist_records
        
    years_diff = year - 2025
    t_warming = 0.05 * years_diff if scenario == "SSP2_4.5" else 0.10 * years_diff
    sm_drying = 0.001 * years_diff if scenario == "SSP2_4.5" else 0.002 * years_diff
    
    # Shift historical data by climate trends
    future_tmax = [r.tmax + t_warming for r in window_data]
    future_sm = [max(0.01, r.soil_moisture - sm_drying) for r in window_data]
    
    comp_days_all = sum(1 for t, sm in zip(future_tmax, future_sm) if t > baseline_tmax and sm < baseline_sm)
    
    unique_years = len(set(r.date.year for r in window_data))
    scaled_comp_days = int(round(comp_days_all / unique_years)) if unique_years > 0 else comp_days_all
    scaled_comp_days = min(30, max(0, scaled_comp_days))
    
    ndvi_vals = [r.ndvi for r in window_data if r.ndvi is not None]
    mean_ndvi = float(np.mean(ndvi_vals)) + np.random.uniform(-0.05, 0.05) if ndvi_vals else np.random.uniform(0.35, 0.65)
    
    return {
        "compound_events": scaled_comp_days,
        "mean_ndvi": float(np.clip(mean_ndvi, 0.1, 0.9))
    }

def load_cmip6_netcdf(db: Session, scenario: str, crop: str) -> pd.DataFrame:
    """
    Attempts to read CMIP6 projection parameters from local Copernicus/IITM Pune NetCDF files
    and slices the gridded records to the district centroids.
    """
    import xarray as xr
    from src.pipeline.dicra_fetcher import get_district_coordinates
    from src.features.phenology_clipper import get_season_months
    
    possible_paths = [
        f"data/raw/cmip6/{scenario.lower()}_{crop.lower()}.nc",
        f"data/raw/cmip6/{scenario}_{crop}.nc",
        f"data/raw/cmip6/{scenario.lower()}.nc",
        f"data/raw/cmip6/{scenario}.nc",
    ]
    
    nc_path = None
    for p in possible_paths:
        if os.path.exists(p):
            nc_path = p
            break
        abs_p = f"/Users/sigma-7/Documents/CSI-DiCRA-Project/CSI-DiCRA-Project/{p}"
        if os.path.exists(abs_p):
            nc_path = abs_p
            break
            
    if not nc_path:
        return None
        
    print(f"📖 Ingesting CMIP6 projections from NetCDF: {nc_path}")
    try:
        ds = xr.open_dataset(nc_path)
        
        tmax_var = None
        sm_var = None
        for v in ds.variables:
            v_lower = v.lower()
            if 'tmax' in v_lower or 'tasmax' in v_lower:
                tmax_var = v
            elif 'soil' in v_lower or 'sm' in v_lower or 'mrso' in v_lower or 'gwetroot' in v_lower:
                sm_var = v
                
        if not tmax_var or not sm_var:
            print(f"⚠️ Temperature/Soil Moisture variables not found in NetCDF. Available: {list(ds.variables)}")
            ds.close()
            return None
            
        lat_coord = None
        lon_coord = None
        time_coord = None
        for c in ds.coords:
            c_lower = c.lower()
            if 'lat' in c_lower:
                lat_coord = c
            elif 'lon' in c_lower or 'lng' in c_lower:
                lon_coord = c
            elif 'time' in c_lower or 'date' in c_lower:
                time_coord = c
                
        if not lat_coord or not lon_coord or not time_coord:
            print(f"⚠️ NetCDF dimensions are missing lat/lon/time mapping. Available: {list(ds.coords)}")
            ds.close()
            return None
            
        years = list(range(2026, 2042))
        rows = []
        
        for district in MP_DISTRICTS:
            lat, lon = get_district_coordinates(district)
            ds_sliced = ds.sel({lat_coord: lat, lon_coord: lon}, method='nearest')
            
            baseline = db.query(DistrictBaselines).filter(
                DistrictBaselines.district == district,
                DistrictBaselines.crop     == crop
            ).first()
            baseline_tmax = baseline.tmax_95th if baseline else 35.0
            baseline_sm = baseline.sm_5th if baseline else 0.15
            
            times = pd.to_datetime(ds_sliced[time_coord].values)
            tmax_vals = ds_sliced[tmax_var].values
            sm_vals = ds_sliced[sm_var].values
            
            # Kelvin to Celsius conversion if applicable
            if np.nanmean(tmax_vals) > 100.0:
                tmax_vals = tmax_vals - 273.15
                
            slice_df = pd.DataFrame({
                "date": times,
                "tmax": tmax_vals,
                "soil_moisture": sm_vals
            }).dropna()
            
            for year in years:
                from src.features.phenology_clipper import CROP_CALENDARS
                cal = CROP_CALENDARS[crop]
                start_year = year + cal["start_year_offset"]
                start_date = pd.Timestamp(f"{start_year}-{cal['critical_start']}")
                end_date = pd.Timestamp(f"{year}-{cal['critical_end']}")
                
                window_df = slice_df[(slice_df['date'] >= start_date) & (slice_df['date'] <= end_date)]
                
                if window_df.empty:
                    fallback_res = generate_physically_guided_fallback(db, district, year, crop, scenario)
                    rows.append({
                        "district": district,
                        "year": year,
                        "compound_events": fallback_res["compound_events"],
                        "mean_ndvi": fallback_res["mean_ndvi"]
                    })
                else:
                    comp_events = sum(
                        1 for _, r in window_df.iterrows()
                        if r["tmax"] > baseline_tmax and r["soil_moisture"] < baseline_sm
                    )
                    fallback_res = generate_physically_guided_fallback(db, district, year, crop, scenario)
                    rows.append({
                        "district": district,
                        "year": year,
                        "compound_events": comp_events,
                        "mean_ndvi": fallback_res["mean_ndvi"]
                    })
                    
        ds.close()
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"❌ Error reading NetCDF: {e}. Falling back to physically-guided simulation.")
        return None

def load_cmip6_data(db: Session, scenario: str, crop: str) -> pd.DataFrame:
    """
    Ingests CMIP6 climate projections. Tries to load NetCDF files via xarray/netcdf4.
    Falls back to physically-guided simulation if NetCDF files are absent.
    """
    df = load_cmip6_netcdf(db, scenario, crop)
    if df is not None and not df.empty:
        return df
        
    print(f"🌱 Generating physically-guided CMIP6 projections for {scenario} / {crop}...")
    rows = []
    years = list(range(2026, 2042))
    for district in MP_DISTRICTS:
        for year in years:
            res = generate_physically_guided_fallback(db, district, year, crop, scenario)
            rows.append({
                "district": district,
                "year": year,
                "compound_events": res["compound_events"],
                "mean_ndvi": res["mean_ndvi"]
            })
    return pd.DataFrame(rows)

def generate_forecasts():
    """
    For every combination of (scenario, crop), load the crop-specific model,
    pull season-aware baselines, and write RiskPredictions rows.
    """
    db = SessionLocal()

    for scenario in ["SSP2_4.5", "SSP5_8.5"]:
        for crop in SUPPORTED_CROPS:
            model_path = f"models/xgboost_{crop.lower()}_model.pkl"
            if not os.path.exists(model_path):
                print(f"⚠️ Model not found for {crop} ({model_path}). "
                      f"Run train_all_models() first. Skipping.")
                continue

            model    = joblib.load(model_path)
            cmip6_df = load_cmip6_data(db, scenario, crop)
            print(f"🔮 Generating forecasts: {scenario} / {crop} ({len(cmip6_df)} rows)...")

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
