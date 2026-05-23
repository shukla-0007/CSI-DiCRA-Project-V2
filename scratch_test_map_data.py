import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from src.database.db import SessionLocal, DailyClimate, DistrictBaselines, RiskPredictions
from src.features.phenology_clipper import CROP_CALENDARS
from src.pipeline.dicra_fetcher import MP_DISTRICTS

def calculate_combined_spi3(db, district: str, cmip6_df: pd.DataFrame) -> pd.Series:
    # 1. Fetch historical precipitation from DB
    hist_records = db.query(DailyClimate.date, DailyClimate.precipitation).filter(
        DailyClimate.district == district
    ).order_by(DailyClimate.date).all()
    
    hist_df = pd.DataFrame(hist_records, columns=["date", "precipitation"])
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    
    # 2. Get future precipitation from cmip6_df
    fut_df = cmip6_df[["date", "precipitation"]].copy()
    fut_df["date"] = pd.to_datetime(fut_df["date"])
    
    # 3. Concatenate
    combined_df = pd.concat([hist_df, fut_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["date"]).sort_values("date").set_index("date")
    
    # 4. Compute 90-day rolling sum
    combined_df["precip_90"] = combined_df["precipitation"].rolling(window=90, min_periods=1).sum()
    
    spi3_series = pd.Series(index=combined_df.index, dtype=float)
    
    # 5. Fit Gamma distribution and map to Z-score
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
        
    return spi3_series

def test_extraction():
    import time
    db = SessionLocal()
    crop = "Soybean"
    year = 2026
    scenario = "SSP5_8.5"
    
    start_time = time.time()
    results = {}
    
    # Pre-fetch predictions for the year/scenario/crop in one query
    preds = db.query(RiskPredictions).filter(
        RiskPredictions.year == year,
        RiskPredictions.crop == crop,
        RiskPredictions.scenario == scenario
    ).all()
    pred_map = {p.district: p for p in preds}
    
    # Pre-fetch baselines in one query
    baselines = db.query(DistrictBaselines).filter(
        DistrictBaselines.crop == crop
    ).all()
    baseline_map = {b.district: b for b in baselines}
    
    cal = CROP_CALENDARS[crop]
    start_year = year + cal["start_year_offset"]
    window_start = pd.Timestamp(f"{start_year}-{cal['critical_start']}")
    window_end   = pd.Timestamp(f"{year}-{cal['critical_end']}")
    
    for district in MP_DISTRICTS:
        district_slug = district.replace(" ", "_").replace("-", "_").lower()
        cache_path = os.path.join("data", "processed", f"cmip6_{district_slug}_{scenario}.csv")
        
        if not os.path.exists(cache_path):
            continue
            
        df = pd.read_csv(cache_path)
        df["date"] = pd.to_datetime(df["date"])
        
        # Get baseline
        baseline = baseline_map.get(district)
        if not baseline:
            continue
            
        # Compute combined SPI3
        spi3_series = calculate_combined_spi3(db, district, df)
        
        # Slice features
        window_df = df[(df["date"] >= window_start) & (df["date"] <= window_end)]
        if window_df.empty:
            continue
            
        mean_tmax = window_df["tmax"].mean()
        mean_sm = window_df["soil_moisture"].mean()
        mean_precip = window_df["precipitation"].mean()
        
        if cal["season"] == "Kharif":
            ndvi_approx = float(np.clip(0.30 + 0.004 * mean_precip, 0.20, 0.80))
        else:
            ndvi_approx = float(np.clip(0.40 + 0.002 * mean_precip, 0.30, 0.70))
            
        window_spi3 = spi3_series.loc[window_start:window_end].mean()
        
        pred = pred_map.get(district)
        tier = pred.risk_tier if pred else 1
        score = pred.csi_score if pred else 0.0
        
        results[district] = {
            "NDVI": round(ndvi_approx, 3),
            "Tmax": round(mean_tmax, 1),
            "Soil Moisture": round(mean_sm, 3),
            "SPI-3": round(window_spi3, 3),
            "CSI Risk Tier": tier,
            "CSI Score": round(score, 3)
        }
        
    end_time = time.time()
    print(f"Processed {len(results)} districts in {end_time - start_time:.3f} seconds.")
    print("Sample result for Bhopal:", results.get("Bhopal"))
    
    db.close()

if __name__ == "__main__":
    test_extraction()
