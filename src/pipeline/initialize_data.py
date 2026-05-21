"""
initialize_data.py
==================
Master pipeline orchestrator for the CSI-DiCRA system.

Run order:
    1. init_db()                  — create SQLite tables (idempotent)
    2. sync_yield_history()       — load PMFBY yield data (real API or synthetic)
    3. generate_bulk_climate_data() — ingest daily climate variables (2018-01-01 to today)
        a. LST, Soil Moisture, NDVI  → DiCRA Live Sync API (Layer 232 / 268 / 214)
        b. Precipitation             → NASA POWER (PRECTOTCORR) for daily resolution
    4. calculate_district_baselines() — compute season-aware 95th/5th percentiles
    5. train_all_models()         — train one XGBoost Regressor per crop
    6. generate_forecasts()       — write RiskPredictions for 2026–2041 (SSP2_4.5 / SSP5_8.5)

Coverage:
    - State:    Madhya Pradesh (all 52 districts)
    - Training: 2018-01-01 → current date  (matches DiCRA satellite coverage start)
    - Forecast: 2026 → 2041
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from sqlalchemy import insert
from src.database.db import init_db, SessionLocal, DailyClimate, YieldHistory
from src.pipeline.pmfby_validation import sync_yield_history
from src.features.dynamic_baselines import calculate_district_baselines
from src.features.phenology_clipper import SUPPORTED_CROPS
from src.model.train_xgboost import train_all_models
from src.model.forecaster import generate_forecasts
from src.pipeline.dicra_fetcher import fetch_dicra_data, MP_DISTRICTS
from src.pipeline.imd_spi_calculator import fetch_chirps_precipitation


# ── Date range for historical training ────────────────────────────────────────
# 2018-01-01 aligns with the earliest MODIS/SMAP satellite records on DiCRA.
TRAINING_START = "2018-01-01"
TRAINING_END   = datetime.today().strftime("%Y-%m-%d")   # up to today


def fetch_and_prepare_district(district, start_date, end_date):
    """Worker function to fetch and prepare climate records for one district."""
    try:
        # 1. DiCRA API → LST, Soil Moisture, NDVI
        dicra_df = fetch_dicra_data(district, start_date, end_date)

        # 2. NASA POWER → daily precipitation
        precip_df = fetch_chirps_precipitation(district, start_date, end_date)

        # 3. Merge on (date, district)
        merged = pd.merge(dicra_df, precip_df, on=["date", "district"], how="inner")

        records = []
        for _, row in merged.iterrows():
            d_val = row["date"].date() if hasattr(row["date"], "date") else row["date"]
            records.append({
                "date":          d_val,
                "district":      row["district"],
                "tmax":          float(row["tmax"])          if pd.notna(row["tmax"])          else None,
                "soil_moisture": float(row["soil_moisture"]) if pd.notna(row["soil_moisture"]) else None,
                "precipitation": float(row["precipitation"]) if pd.notna(row["precipitation"]) else None,
                "ndvi":          float(row["ndvi"])          if pd.notna(row["ndvi"])          else None,
            })
        print(f"  ✅ {district}: {len(merged):,} daily records prepared.")
        return records
    except Exception as e:
        print(f"  ❌ Error processing district {district}: {e}")
        return []


def generate_bulk_climate_data():
    """
    Fetches and stores daily climate records for all 52 MP districts from
    2018-01-01 to the current date.

    Uses ThreadPoolExecutor for concurrent fetching to speed up the process.
    """
    db = SessionLocal()

    existing_count = db.query(DailyClimate).count()
    if existing_count > 0:
        print(f"ℹ️  daily_climate already has {existing_count:,} records. "
              f"Skipping bulk sync.  (Delete the DB or truncate the table to re-run.)")
        db.close()
        return

    print(f"📡 Ingesting climate data for all {len(MP_DISTRICTS)} MP districts concurrently "
          f"({TRAINING_START} → {TRAINING_END})...")
    print(f"   Sources: DiCRA API (LST/SM/NDVI) + NASA POWER (precipitation)\n")

    import concurrent.futures
    all_records = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_dist = {
            executor.submit(fetch_and_prepare_district, dist, TRAINING_START, TRAINING_END): dist
            for dist in MP_DISTRICTS
        }
        for future in concurrent.futures.as_completed(future_to_dist):
            district = future_to_dist[future]
            try:
                records = future.result()
                all_records.extend(records)
            except Exception as exc:
                print(f"  ❌ District {district} generated an exception: {exc}")

    if all_records:
        print(f"\n💾 Bulk inserting {len(all_records):,} climate records ...")
        db.execute(insert(DailyClimate), all_records)
        db.commit()
        print("✅ daily_climate table populated successfully.")
    else:
        print("⚠️  No climate records were prepared — check API connectivity.")

    db.close()


def main():
    print("=" * 65)
    print("  CSI-DiCRA — Full Pipeline Initialisation")
    print(f"  Coverage: All 52 MP Districts | {TRAINING_START} → {TRAINING_END}")
    print("=" * 65)

    # 1. Initialise DB schema
    init_db()

    # 2. Sync PMFBY yield history (real API or synthetic 52-district dataset)
    print("\n[Step 2] 🌾 Syncing yield history...")
    sync_yield_history()

    # 3. Bulk-ingest daily climate (DiCRA + NASA POWER)
    print("\n[Step 3] 📡 Fetching climate data from DiCRA Live Sync + NASA POWER...")
    generate_bulk_climate_data()

    # 4. Season-aware baseline calculation per (district, crop)
    print("\n[Step 4] 📊 Calculating season-aware baselines...")
    db = SessionLocal()
    calculate_district_baselines(db)
    db.close()

    # 5. Train one XGBoost model per crop
    print("\n[Step 5] 🚀 Training XGBoost models for all 6 crops...")
    train_all_models()

    # 6. Generate CMIP6 risk forecasts (2026-2041)
    print("\n[Step 6] 🔮 Generating CMIP6 CSI forecasts (SSP2_4.5 & SSP5_8.5)...")
    generate_forecasts()

    print("\n" + "=" * 65)
    print("  ✅  CSI-DiCRA pipeline initialisation complete!")
    print("     → FastAPI: uvicorn src.main:app --host 0.0.0.0 --port 8000")
    print("     → Dashboard: streamlit run src/frontend.py --server.port 8501")
    print("=" * 65)


if __name__ == "__main__":
    main()
