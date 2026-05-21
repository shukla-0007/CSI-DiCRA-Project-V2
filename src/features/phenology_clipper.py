from datetime import datetime
from sqlalchemy.orm import Session
from src.database.db import SessionLocal, DailyClimate

# FAO Crop Calendars for Madhya Pradesh
# critical_start / critical_end : MM-DD of the flowering / grain-fill window
# start_year_offset : -1 for Rabi crops whose window opens in December of the
#   PREVIOUS calendar year (Gram, Mustard). year param = harvest year.
CROP_CALENDARS = {
    "Soybean": {"season": "Kharif", "critical_start": "07-25", "critical_end": "08-31", "start_year_offset":  0},
    "Rice":    {"season": "Kharif", "critical_start": "09-01", "critical_end": "09-30", "start_year_offset":  0},
    "Maize":   {"season": "Kharif", "critical_start": "08-10", "critical_end": "09-10", "start_year_offset":  0},
    "Wheat":   {"season": "Rabi",   "critical_start": "01-20", "critical_end": "02-20", "start_year_offset":  0},
    "Gram":    {"season": "Rabi",   "critical_start": "12-20", "critical_end": "01-31", "start_year_offset": -1},
    "Mustard": {"season": "Rabi",   "critical_start": "12-15", "critical_end": "01-15", "start_year_offset": -1},
}

SUPPORTED_CROPS = list(CROP_CALENDARS.keys())


def get_clipped_climate_data(db: Session, district: str, year: int, crop: str):
    """
    Retrieves DailyClimate records for a district/year that fall within the
    critical flowering window for the given crop.

    For Gram and Mustard the window crosses the calendar year boundary:
      e.g. year=2026 → Dec 20 2025 to Jan 31 2026
    """
    if crop not in CROP_CALENDARS:
        raise ValueError(
            f"Crop '{crop}' not supported. Choose from: {SUPPORTED_CROPS}"
        )

    cal        = CROP_CALENDARS[crop]
    start_year = year + cal["start_year_offset"]

    start_date = datetime.strptime(f"{start_year}-{cal['critical_start']}", "%Y-%m-%d").date()
    end_date   = datetime.strptime(f"{year}-{cal['critical_end']}",         "%Y-%m-%d").date()

    return db.query(DailyClimate).filter(
        DailyClimate.district == district,
        DailyClimate.date     >= start_date,
        DailyClimate.date     <= end_date,
    ).all()


def get_season_months(crop: str) -> list:
    """
    Returns the calendar month numbers (1-12) that fall inside the critical
    window for the given crop.  Used by dynamic_baselines to filter
    DailyClimate records before computing season-aware percentiles.
    """
    cal         = CROP_CALENDARS[crop]
    start_month = int(cal["critical_start"].split("-")[0])
    end_month   = int(cal["critical_end"].split("-")[0])

    if start_month <= end_month:
        return list(range(start_month, end_month + 1))
    # Window crosses year boundary (Dec → Jan)
    return list(range(start_month, 13)) + list(range(1, end_month + 1))


if __name__ == "__main__":
    print("Phenology Clipper — crop calendar summary")
    for crop, cal in CROP_CALENDARS.items():
        months = get_season_months(crop)
        print(f"  {crop:8s} | {cal['season']:6s} | "
              f"{cal['critical_start']} → {cal['critical_end']} | "
              f"season months: {months}")
