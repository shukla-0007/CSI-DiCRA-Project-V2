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
from src.database.db import init_db, SessionLocal, DailyClimate, YieldHistory, log_api_status
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


def generate_bulk_climate_data(incremental=True, safety_days=15):
    """
    Fetches and stores daily climate records for all 52 MP districts.
    If incremental=True, it finds the maximum date in DailyClimate,
    subtracts safety_days to find the start_date, and fetches from there to today.
    Existing records in that window are deleted first to avoid duplicates.
    """
    db = SessionLocal()

    start_date = TRAINING_START
    if incremental:
        from sqlalchemy import func
        max_date = db.query(func.max(DailyClimate.date)).scalar()
        if max_date is not None:
            start_dt = max_date - timedelta(days=safety_days)
            ts_dt = datetime.strptime(TRAINING_START, "%Y-%m-%d").date()
            if start_dt < ts_dt:
                start_dt = ts_dt
            start_date = start_dt.strftime("%Y-%m-%d")

    end_date = datetime.today().strftime("%Y-%m-%d")

    print(f"🔄 Syncing climate data from {start_date} to {end_date} (incremental={incremental}, safety_days={safety_days})...")

    # Delete existing records in the window before writing to prevent duplicate keys
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    del_count = db.query(DailyClimate).filter(
        DailyClimate.date >= start_dt,
        DailyClimate.date <= end_dt
    ).delete()
    db.commit()
    print(f"🗑️ Deleted {del_count:,} existing climate records in overlap window [{start_date} to {end_date}].")

    print(f"📡 Ingesting climate data for all {len(MP_DISTRICTS)} MP districts concurrently...")

    import concurrent.futures
    all_records = []

    # Use more workers if incremental since we have a smaller date window
    max_workers = 10 if incremental else 5
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dist = {
            executor.submit(fetch_and_prepare_district, dist, start_date, end_date): dist
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
        msg = f"Synced {len(all_records):,} daily records across {len(MP_DISTRICTS)} districts for window [{start_date} to {end_date}]."
        log_api_status(db, "Climate Ingestion", None, "SUCCESS", msg)
    else:
        print("⚠️ No climate records were prepared — check API connectivity.")
        log_api_status(db, "Climate Ingestion", None, "FAILED", "No climate records prepared from APIs.")

    db.close()
    return start_date, end_date


import threading
pipeline_lock = threading.Lock()


def run_pipeline(incremental=True, safety_days=15):
    """
    Thread-safe orchestrator to run the entire CSI-DiCRA pipeline.
    Returns True if completed successfully, False otherwise.
    """
    if not pipeline_lock.acquire(blocking=False):
        print("⚠️ Pipeline run is already in progress. Skipping execution.")
        db = SessionLocal()
        try:
            log_api_status(db, "Pipeline Scheduler", None, "FAILED", "Skipped run: another sync is currently running.")
        finally:
            db.close()
        return False

    db = SessionLocal()
    try:
        log_api_status(db, "Pipeline Scheduler", None, "IN_PROGRESS", f"Pipeline run started (incremental={incremental}, safety_days={safety_days}).")
        db.commit()

        # 1. Initialise DB schema
        init_db()

        # 2. Sync PMFBY yield history
        print("\n[Step 2] 🌾 Syncing yield history...")
        sync_yield_history()

        # 3. Bulk-ingest daily climate (DiCRA + NASA POWER)
        print("\n[Step 3] 📡 Fetching climate data from DiCRA Live Sync + NASA POWER...")
        start_date, end_date = generate_bulk_climate_data(incremental=incremental, safety_days=safety_days)

        # 4. Season-aware baseline calculation per (district, crop)
        print("\n[Step 4] 📊 Calculating season-aware baselines...")
        calculate_district_baselines(db)

        # 5. Train one XGBoost model per crop
        print("\n[Step 5] 🚀 Training XGBoost models for all 6 crops...")
        train_all_models()

        # 6. Generate CMIP6 risk forecasts (2026-2041)
        print("\n[Step 6] 🔮 Generating CMIP6 CSI forecasts (SSP2_4.5 & SSP5_8.5)...")
        generate_forecasts()

        msg = f"Pipeline sync complete. Overlap: {start_date} to {end_date}. Models retrained and forecasts regenerated."
        print(f"\n✅ {msg}")
        log_api_status(db, "Pipeline Scheduler", None, "SUCCESS", msg)
        return True
    except Exception as e:
        import traceback
        err_msg = f"Pipeline execution failed: {e}\n{traceback.format_exc()}"
        print(f"\n❌ {err_msg}")
        log_api_status(db, "Pipeline Scheduler", None, "FAILED", f"Error: {e}")
        return False
    finally:
        db.close()
        pipeline_lock.release()


def main():
    import sys
    incremental = True
    if "--full" in sys.argv:
        incremental = False
    run_pipeline(incremental=incremental)


if __name__ == "__main__":
    main()
