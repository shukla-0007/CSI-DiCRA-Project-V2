"""
dicra_fetcher.py
================
Fetches Land Surface Temperature (LST), Soil Moisture (SOILM), and NDVI
from the DiCRA Live Sync API for all 52 districts of Madhya Pradesh.

DiCRA API endpoint used:
    POST https://dicra-api.centralindia.cloudapp.azure.com/api/v2/getppointtrend

Layer IDs (confirmed from the DiCRA API docs at /docs#):
    232  →  LST  (MODIS Land Surface Temperature, °C)
    268  →  SOILM (SMAP Root-zone Soil Moisture, m³/m³)
    214  →  NDVI  (MODIS Normalised Difference Vegetation Index)

Fallback chain:
    DiCRA API  →  NASA POWER (for LST/SM)  →  Physics-guided simulation
"""

import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from src.database.db import SessionLocal, DailyClimate, log_api_status

# ── District list — all 52 of Madhya Pradesh ──────────────────────────────────
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

# ── DiCRA API configuration ────────────────────────────────────────────────────
DICRA_BASE_URL = "https://dicra-api.centralindia.cloudapp.azure.com"
DICRA_ENDPOINT = f"{DICRA_BASE_URL}/api/v2/getppointtrend"

DICRA_LAYER_LST   = 232   # MODIS LST (°C)
DICRA_LAYER_SOILM = 268   # SMAP Root-zone Soil Moisture (m³/m³)
DICRA_LAYER_NDVI  = 214   # MODIS NDVI

# ── NASA POWER fallback configuration ─────────────────────────────────────────
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


# ══════════════════════════════════════════════════════════════════════════════
#  COORDINATE LOOKUP
# ══════════════════════════════════════════════════════════════════════════════

def get_district_coordinates(district_name: str):
    """
    Returns (lat, lon) for an MP district from mp_district_coordinates.csv.
    Handles name normalisation between the DB format (with spaces) and the
    CSV format (CamelCase, special aliases for Khandwa / Khargone / Narsinghpur).
    Niwari is the newest district (carved out of Tikamgarh in 2018) and is
    not present in the original CSV, so it is mapped manually.
    """
    if district_name == "Niwari":
        return 25.45, 78.80

    # Try both relative and absolute paths so the script works from any cwd
    for csv_path in [
        "data/mp_district_coordinates.csv",
        os.path.join(os.path.dirname(__file__), "../../data/mp_district_coordinates.csv"),
        "/Users/sigma-7/Documents/CSI-DiCRA-Project/CSI-DiCRA-Project/data/mp_district_coordinates.csv",
    ]:
        if os.path.exists(csv_path):
            break
    else:
        # Default to geographic centre of MP if CSV missing
        return 23.5, 77.5

    # Build a normalised key for the district name
    name_map = {
        "Khandwa":      "EastNimar",
        "Khargone":     "WestNimar",
        "Narsinghpur":  "Narsimhapur",
    }
    mapped_name = name_map.get(district_name, district_name.replace(" ", ""))

    df = pd.read_csv(csv_path)
    row = df[df["district_name"] == mapped_name]
    if not row.empty:
        return float(row.iloc[0]["lat"]), float(row.iloc[0]["lon"])

    # Partial-match fallback
    row = df[df["district_name"].str.lower() == mapped_name.lower()]
    if not row.empty:
        return float(row.iloc[0]["lat"]), float(row.iloc[0]["lon"])

    return 23.5, 77.5   # geographic centre of MP


# ══════════════════════════════════════════════════════════════════════════════
#  DICRA API HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _dicra_post(lat: float, lon: float, start_date: str, end_date: str, layer_id: int) -> list:
    """
    Calls /api/v2/getppointtrend and returns a list of (date_str, value) tuples.

    Confirmed response format (from live API probe):
        {
            "code": 200,
            "trend": [[1580428800000, 22.08], [1582416000000, 24.1], ...]
        }
    where each element is [unix_epoch_ms, float_value].

    Returns [] on any error or when the API returns code != 200.
    """
    payload = {
        "latitude":  str(round(lat, 6)),
        "longitude": str(round(lon, 6)),
        "startdate": start_date,
        "enddate":   end_date,
        "layer_id":  layer_id,
    }
    try:
        resp = requests.post(
            DICRA_ENDPOINT,
            json=payload,
            timeout=60,
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code != 200:
            return []

        data = resp.json()

        # Check for API-level error code (e.g. {"code": 404, "message": "..."})
        if isinstance(data, dict) and data.get("code", 200) != 200:
            return []

        trend_raw = None
        # Primary path: confirmed {"code": 200, "trend": [[epoch_ms, val], ...]}
        if "trend" in data:
            trend_raw = data["trend"]
        else:
            # Fallback: scan other common key names
            for key in ("data", "result", "results", "timeseries"):
                if key in data:
                    trend_raw = data[key]
                    break

        if trend_raw is None:
            if isinstance(data, list):
                trend_raw = data
            else:
                return []

        if not isinstance(trend_raw, list) or not trend_raw:
            return []

        pairs = []
        first = trend_raw[0]

        # ── Format A (confirmed): [[epoch_ms, value], ...] ────────────────────
        if isinstance(first, (list, tuple)) and len(first) == 2:
            for item in trend_raw:
                try:
                    epoch_ms, val = item
                    if val is None:
                        continue
                    date_str = pd.Timestamp(int(epoch_ms), unit="ms").strftime("%Y-%m-%d")
                    pairs.append((date_str, float(val)))
                except Exception:
                    continue
            return pairs

        # ── Format B: [{"date": "YYYY-MM-DD", "value": 28.3}, ...] ───────────
        if isinstance(first, dict):
            for item in trend_raw:
                date_val = item.get("date") or item.get("Date") or item.get("time")
                val = item.get("value") or item.get("Value") or item.get("val")
                if date_val is not None and val is not None:
                    pairs.append((str(date_val)[:10], float(val)))
            return pairs

        # ── Format C: {"dates": [...], "values": [...]} ───────────────────────
        if isinstance(trend_raw, dict):
            dates = trend_raw.get("dates") or trend_raw.get("date") or []
            values = trend_raw.get("values") or trend_raw.get("value") or []
            return [(str(d)[:10], float(v)) for d, v in zip(dates, values) if v is not None]

        return []

    except Exception:
        return []


def _pairs_to_series(pairs: list, start_date: str, end_date: str) -> pd.Series:
    """
    Convert (date_str, value) pairs to a pandas Series indexed by date.
    Deduplicates by date — MODIS can return two observations for overlapping
    16-day compositing periods; we keep the last (most recent) non-null value.
    """
    if not pairs:
        return pd.Series(dtype=float)
    dates, values = zip(*pairs)
    s = pd.Series(
        data=list(values),
        index=pd.to_datetime(list(dates)),
        dtype=float,
    )
    # Drop duplicate dates — keep last value (most recently composited)
    s = s[~s.index.duplicated(keep='last')]
    # Clip to the requested window
    s = s.sort_index()[start_date:end_date]
    return s


def _series_to_dict(s: pd.Series) -> dict:
    """Convert a deduplicated Series to a plain dict for O(1) safe scalar lookup."""
    return {d: v for d, v in s.items() if not (isinstance(v, float) and np.isnan(v))}


# ══════════════════════════════════════════════════════════════════════════════
#  NASA POWER FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def _nasa_power_fetch(lat: float, lon: float, start_date: str, end_date: str):
    """
    Fetches T2M_MAX and GWETROOT from NASA POWER as fallback for LST and
    Soil Moisture when DiCRA API is unavailable.

    Returns (tmax_series, sm_series) as pd.Series indexed by date, or
    (None, None) on failure.
    """
    start_str = start_date.replace("-", "")
    end_str   = end_date.replace("-", "")
    params = {
        "start":      start_str,
        "end":        end_str,
        "latitude":   lat,
        "longitude":  lon,
        "community":  "ag",
        "parameters": "T2M_MAX,GWETROOT",
        "format":     "JSON",
    }
    try:
        resp = requests.get(NASA_POWER_URL, params=params, timeout=60)
        if resp.status_code == 200:
            param_data = resp.json()["properties"]["parameter"]
            tmax_dict  = param_data.get("T2M_MAX", {})
            sm_dict    = param_data.get("GWETROOT", {})

            tmax_series = pd.Series(
                {pd.Timestamp(k): (v if v != -999.0 else np.nan)
                 for k, v in tmax_dict.items()},
                dtype=float,
            )
            sm_series = pd.Series(
                {pd.Timestamp(k): (v if v != -999.0 else np.nan)
                 for k, v in sm_dict.items()},
                dtype=float,
            )
            return tmax_series, sm_series
    except Exception:
        pass
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATION FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_climate(dates: pd.DatetimeIndex) -> tuple:
    """
    Physics-guided climate simulation used only when both DiCRA and NASA
    POWER are unreachable.  Produces seasonally plausible Tmax, SM, and NDVI
    for Madhya Pradesh.
    """
    n = len(dates)
    # Day-of-year to month mapping for seasonal patterns
    months = np.array([d.month for d in dates])

    # Tmax: peaks ~42°C in May, lowest ~20°C in January
    tmax = 30.0 + 12.0 * np.sin(np.pi * (months - 2) / 6) + np.random.normal(0, 1.5, n)
    tmax = np.clip(tmax, 18.0, 46.0)

    # Soil moisture: highest during monsoon (Jul-Sep), lowest in May
    sm = 0.18 + 0.12 * np.sin(np.pi * (months - 3) / 5) + np.random.normal(0, 0.02, n)
    sm = np.clip(sm, 0.04, 0.40)

    # NDVI: peaks during monsoon, low in summer
    ndvi = 0.35 + 0.25 * np.sin(np.pi * (months - 3) / 5) + np.random.normal(0, 0.03, n)
    ndvi = np.clip(ndvi, 0.05, 0.85)

    return tmax, sm, ndvi


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def fetch_dicra_data(district_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [date, district, tmax, soil_moisture, ndvi]
    for the given district across the requested date range.

    Data source priority:
        1. DiCRA Live Sync API (real satellite data)
        2. NASA POWER (daily reanalysis — for LST/SM only, NDVI still simulated)
        3. Physics-guided simulation (seasonal patterns)
    """
    lat, lon = get_district_coordinates(district_name)
    print(f"  📡 [{district_name}] DiCRA API query {start_date} → {end_date} ...")

    def _db_log(status: str, service: str, message: str):
        db = SessionLocal()
        try:
            log_api_status(db, service, district_name, status, message)
        except Exception as e:
            print(f"Logging error: {e}")
        finally:
            db.close()


    # ── Step 1: Attempt DiCRA Live Sync ────────────────────────────────────────
    lst_pairs   = _dicra_post(lat, lon, start_date, end_date, DICRA_LAYER_LST)
    soilm_pairs = _dicra_post(lat, lon, start_date, end_date, DICRA_LAYER_SOILM)
    ndvi_pairs  = _dicra_post(lat, lon, start_date, end_date, DICRA_LAYER_NDVI)

    dicra_ok = bool(lst_pairs or soilm_pairs or ndvi_pairs)

    if dicra_ok:
        print(f"  ✅ [{district_name}] DiCRA returned "
              f"LST={len(lst_pairs)} SOILM={len(soilm_pairs)} NDVI={len(ndvi_pairs)} records.")
        _db_log("SUCCESS", "DiCRA Live Sync API", f"LST={len(lst_pairs)}, SOILM={len(soilm_pairs)}, NDVI={len(ndvi_pairs)} records fetched.")
    else:
        _db_log("FALLBACK", "DiCRA Live Sync API", "Empty response from all layers.")

    lst_s   = _pairs_to_series(lst_pairs,   start_date, end_date)
    soilm_s = _pairs_to_series(soilm_pairs, start_date, end_date)
    ndvi_s  = _pairs_to_series(ndvi_pairs,  start_date, end_date)

    # Reindex and interpolate sparse satellite data to daily resolution
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    if not lst_s.empty:
        lst_s = lst_s.reindex(dates).interpolate(method="time").ffill().bfill()
    if not soilm_s.empty:
        soilm_s = soilm_s.reindex(dates).interpolate(method="time").ffill().bfill()
    if not ndvi_s.empty:
        ndvi_s = ndvi_s.reindex(dates).interpolate(method="time").ffill().bfill()

    # Convert to plain dicts for safe O(1) scalar lookup (avoids Series ambiguity)
    lst_d   = _series_to_dict(lst_s)
    soilm_d = _series_to_dict(soilm_s)
    ndvi_d  = _series_to_dict(ndvi_s)

    # ── Step 2: NASA POWER fallback for LST / SM if DiCRA returned nothing ─────
    nasa_tmax_d: dict = {}
    nasa_sm_d:   dict = {}
    if not lst_d or not soilm_d:
        print(f"  ⚠️  [{district_name}] DiCRA partial/empty — trying NASA POWER fallback...")
        _db_log("FALLBACK", "DiCRA Live Sync API", "Partial/empty LST or Soil Moisture data. Reverting to NASA POWER.")
        nasa_tmax_s, nasa_sm_s = _nasa_power_fetch(lat, lon, start_date, end_date)
        if nasa_tmax_s is not None:
            nasa_tmax_s = nasa_tmax_s.reindex(dates).interpolate(method="time").ffill().bfill()
            nasa_tmax_d = _series_to_dict(nasa_tmax_s)
            if nasa_sm_s is not None:
                nasa_sm_s = nasa_sm_s.reindex(dates).interpolate(method="time").ffill().bfill()
                nasa_sm_d   = _series_to_dict(nasa_sm_s)
            print(f"  ✅ [{district_name}] NASA POWER returned {len(nasa_tmax_d)} records.")
            _db_log("SUCCESS", "NASA POWER API Fallback", f"Successfully fetched {len(nasa_tmax_d)} fallback records.")
        else:
            print(f"  ⚠️  [{district_name}] NASA POWER also unavailable — using simulation.")
            _db_log("FALLBACK", "NASA POWER API Fallback", "NASA POWER also unavailable; falling back to simulation.")
            _db_log("FALLBACK", "DiCRA Weather Fallback", "Both DiCRA and NASA POWER failed. Using seasonal daily weather simulation.")

    # ── Step 3: Build the full date range and fill all columns ─────────────────
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    tmax_vals  = []
    sm_vals    = []
    ndvi_vals  = []

    # Pre-compute simulation arrays in case needed
    sim_tmax, sim_sm, sim_ndvi = _simulate_climate(dates)

    for i, d in enumerate(dates):
        # — TMAX — dict.get returns a scalar directly, no Series ambiguity
        tmax_val = lst_d.get(d) or nasa_tmax_d.get(d)
        tmax_vals.append(float(tmax_val) if tmax_val is not None else float(sim_tmax[i]))

        # — SOIL MOISTURE —
        sm_val = soilm_d.get(d) or nasa_sm_d.get(d)
        sm_vals.append(float(sm_val) if sm_val is not None else float(sim_sm[i]))

        # — NDVI —
        ndvi_val = ndvi_d.get(d)
        ndvi_vals.append(float(ndvi_val) if ndvi_val is not None else float(sim_ndvi[i]))

    df = pd.DataFrame({
        "date":          dates,
        "district":      district_name,
        "tmax":          tmax_vals,
        "soil_moisture": sm_vals,
        "ndvi":          ndvi_vals,
    })

    if not ndvi_d:
        _db_log("FALLBACK", "DiCRA NDVI Fallback", "NDVI layer empty. Falling back to daily NDVI simulation.")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE UPSERT
# ══════════════════════════════════════════════════════════════════════════════

def update_database(db: Session, df: pd.DataFrame):
    """Upsert climate rows into the DailyClimate table."""
    for _, row in df.iterrows():
        d = row["date"].date() if hasattr(row["date"], "date") else row["date"]
        existing = db.query(DailyClimate).filter(
            DailyClimate.date     == d,
            DailyClimate.district == row["district"],
        ).first()

        if existing:
            existing.tmax          = row["tmax"]
            existing.soil_moisture = row["soil_moisture"]
            existing.ndvi          = row["ndvi"]
        else:
            db.add(DailyClimate(
                date          = d,
                district      = row["district"],
                tmax          = row["tmax"],
                soil_moisture = row["soil_moisture"],
                ndvi          = row["ndvi"],
            ))

    db.commit()
    print(f"  💾 [{df['district'].iloc[0]}] {len(df)} records upserted.")


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE SYNC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def sync_dicra_live():
    """
    Syncs the last 30 days of DiCRA data for every MP district.
    Called by a scheduler or manually to keep the DB current.
    """
    db        = SessionLocal()
    end_date  = datetime.today()
    start_date = end_date - timedelta(days=30)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    print(f"🔄 DiCRA Live Sync — {start_str} to {end_str} for all {len(MP_DISTRICTS)} districts...")
    for district in MP_DISTRICTS:
        df = fetch_dicra_data(district, start_str, end_str)
        update_database(db, df)
        time.sleep(0.5)   # polite rate-limiting

    db.close()
    print("✅ DiCRA Live Sync complete.")


if __name__ == "__main__":
    sync_dicra_live()
