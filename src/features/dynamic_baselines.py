import pandas as pd
from datetime import date
from sqlalchemy.orm import Session
from sqlalchemy import func
from src.database.db import SessionLocal, DailyClimate, DistrictBaselines
from src.features.phenology_clipper import SUPPORTED_CROPS, get_season_months


def calculate_district_baselines(db: Session):
    """
    Season-aware baseline calculation.

    For each (district, crop) pair, only the DailyClimate records whose
    calendar month falls inside that crop's critical flowering window are used
    to compute percentiles.  This ensures:
      - Wheat thresholds reflect January–February temperature norms (~25°C),
        not the year-round 95th percentile (~42°C).
      - Soybean thresholds reflect July–August monsoon-season norms.

    Baselines are upserted into DistrictBaselines with composite PK
    (district, crop).
    """
    print("🔄 Calculating Season-Aware Baselines (95th Heat, 5th Drought) per crop...")

    districts = [d[0] for d in db.query(DailyClimate.district).distinct().all()]
    if not districts:
        print("⚠️  No climate data found in daily_climate table. Run the fetcher first.")
        return

    updated = 0
    for district in districts:
        for crop in SUPPORTED_CROPS:
            season_months = get_season_months(crop)
            # Convert to zero-padded strings for SQLite strftime comparison
            month_strs = [f"{m:02d}" for m in season_months]

            records = db.query(DailyClimate).filter(
                DailyClimate.district == district,
                func.strftime("%m", DailyClimate.date).in_(month_strs),
            ).all()

            if not records:
                continue

            df = pd.DataFrame([
                {"tmax": r.tmax, "soil_moisture": r.soil_moisture}
                for r in records
            ]).dropna()

            if df.empty:
                continue

            tmax_95th = float(df["tmax"].quantile(0.95))
            sm_5th    = float(df["soil_moisture"].quantile(0.05))

            existing = db.query(DistrictBaselines).filter(
                DistrictBaselines.district == district,
                DistrictBaselines.crop     == crop,
            ).first()

            if existing:
                existing.tmax_95th     = tmax_95th
                existing.sm_5th        = sm_5th
                existing.calculated_on = date.today()
            else:
                db.add(DistrictBaselines(
                    district      = district,
                    crop          = crop,
                    tmax_95th     = tmax_95th,
                    sm_5th        = sm_5th,
                    calculated_on = date.today(),
                ))

            updated += 1
            print(f"  📍 {district} / {crop}: "
                  f"Tmax 95th={tmax_95th:.1f}°C  SM 5th={sm_5th:.4f}  "
                  f"(months {season_months})")

    db.commit()
    print(f"✅ Baseline calculation complete — {updated} (district, crop) pairs updated.")


if __name__ == "__main__":
    db = SessionLocal()
    calculate_district_baselines(db)
    db.close()
