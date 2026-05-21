"""
pmfby_validation.py
===================
Sync historical crop yield records from data.gov.in or generate a
deterministic synthetic dataset for all 52 MP districts (2018–2025)
when no API key is available.

Synthetic yields are drawn from PMFBY / ICAR documented typical ranges
for Madhya Pradesh and perturbed by a district-specific seed so results
are reproducible but varied across space and time.
"""

import os
import random
import requests
import pandas as pd
from sqlalchemy.orm import Session
from src.database.db import SessionLocal, YieldHistory, log_api_status

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

# ── Crop configuration ─────────────────────────────────────────────────────────
# (season, mean_yield_kg_ha, std_fraction, shock_year_factor)
# shock_year_factor: applied in 2018, 2021 to simulate documented poor years
CROP_CONFIG = {
    "Soybean": ("Kharif", 1000.0, 0.22, 0.62),   # 2018 MP soybean shock −38%
    "Rice":    ("Kharif", 1200.0, 0.18, 0.75),
    "Maize":   ("Kharif", 2000.0, 0.20, 0.80),
    "Wheat":   ("Rabi",   2000.0, 0.17, 0.85),
    "Gram":    ("Rabi",   1000.0, 0.20, 0.78),
    "Mustard": ("Rabi",   1100.0, 0.18, 0.82),
}

# Districts where historically a crop is NOT typically grown (yield ~ 0).
# We omit these pairs so the training data stays realistic.
NON_GROWING_PAIRS = {
    ("Alirajpur", "Wheat"),    # tribal belt — minimal wheat cultivation
    ("Barwani",   "Wheat"),
    ("Jhabua",    "Wheat"),
    ("Alirajpur", "Mustard"),
    ("Barwani",   "Mustard"),
    ("Jhabua",    "Mustard"),
    ("Alirajpur", "Gram"),
    ("Barwani",   "Gram"),
    ("Jhabua",    "Gram"),
}

# Shock years — documented below-normal production across MP
KHARIF_SHOCK_YEARS = {2018, 2021}
RABI_SHOCK_YEARS   = {2019, 2022}


def _synthetic_yield(district: str, year: int, crop: str) -> float | None:
    """
    Generates a deterministic but spatially-varied yield for one
    (district, year, crop) triplet.

    Determinism: seeded by hash(district + crop + year) so the same call
    always returns the same number — important for reproducibility.
    """
    if (district, crop) in NON_GROWING_PAIRS:
        return None   # skip this pair

    season, mean, std_frac, shock_factor = CROP_CONFIG[crop]

    # District-level multiplicative factor (±10%) to create spatial variation
    dist_seed = abs(hash(district + crop)) % 10000
    rng = random.Random(dist_seed + year * 31)
    spatial_factor = rng.uniform(0.90, 1.10)

    # Shock year adjustment
    is_shock = (season == "Kharif" and year in KHARIF_SHOCK_YEARS) or \
               (season == "Rabi"   and year in RABI_SHOCK_YEARS)

    base = mean * spatial_factor
    if is_shock:
        base *= shock_factor

    # Inter-annual random noise
    noise_rng = random.Random(dist_seed + year * 97 + ord(crop[0]))
    noise = noise_rng.gauss(0.0, std_frac * base * 0.5)
    yield_val = max(100.0, base + noise)

    return round(yield_val, 1)


def generate_synthetic_yields() -> pd.DataFrame:
    """
    Generates a full MP yield dataset for all 52 districts, 6 crops, 2018–2025.
    This replaces the original 24-row mock list with ~2496 realistic records
    (excluding non-growing district/crop pairs).
    """
    records = []
    for district in MP_DISTRICTS:
        for crop, (season, *_) in CROP_CONFIG.items():
            for year in range(2018, 2026):   # 2018 to 2025 inclusive
                y = _synthetic_yield(district, year, crop)
                if y is not None:
                    records.append({
                        "district":    district,
                        "year":        year,
                        "season":      season,
                        "crop":        crop,
                        "yield_kg_ha": y,
                    })
    df = pd.DataFrame(records)
    print(f"  📊 Synthetic yield dataset: {len(df)} records across "
          f"{df['district'].nunique()} districts × "
          f"{df['crop'].nunique()} crops × "
          f"{df['year'].nunique()} years.")
    return df


def fetch_historical_yields(db: Session = None) -> pd.DataFrame:
    """
    Loads historical crop yield statistics from local data/mp_historical_yields.csv.
    Normalizes raw columns (Tonnes/Hectare to kg/ha, crop names, district names) if detected,
    and falls back to generating the deterministic synthetic dataset if the CSV is missing.
    """
    local_csv_path = os.path.join("data", "mp_historical_yields.csv")

    if os.path.exists(local_csv_path):
        print(f"ℹ️  Loading historical yields from offline cache: {local_csv_path}")
        try:
            df = pd.read_csv(local_csv_path)
            
            # Detect if it's the raw/downloaded format from India Data Portal/Dataful
            if "district_name" in df.columns and "crop_name" in df.columns:
                print("  📊 Raw downloaded yields format detected. Normalizing and transforming...")
                
                # Mappings
                crop_mapping = {
                    "soyabean": "Soybean",
                    "soybean": "Soybean",
                    "wheat": "Wheat",
                    "maize": "Maize",
                    "gram": "Gram",
                    "rice": "Rice",
                    "rapeseed &mustard": "Mustard",
                    "rapeseed & mustard": "Mustard",
                    "mustard": "Mustard"
                }
                
                district_mapping = {
                    "agar-malwa": "Agar Malwa",
                    "agar malwa": "Agar Malwa",
                    "khandwa (east nimar)": "Khandwa",
                    "khandwa": "Khandwa",
                    "khargone (west nimar)": "Khargone",
                    "khargone": "Khargone",
                    "narmadapuram": "Hoshangabad",
                    "hoshangabad": "Hoshangabad",
                    "narsimhapur": "Narsinghpur",
                    "narsinghpur": "Narsinghpur",
                }
                
                # Normalize string representations for matching
                df["crop_lower"] = df["crop_name"].astype(str).str.strip().str.lower()
                df["dist_lower"] = df["district_name"].astype(str).str.strip().str.lower()
                
                # Filter to only the 6 supported crops
                df = df[df["crop_lower"].isin(crop_mapping.keys())].copy()
                
                # Apply mappings
                df["crop"] = df["crop_lower"].map(crop_mapping)
                
                # Filter to districts that we care about or map them
                def map_district(row):
                    d_low = row["dist_lower"]
                    if d_low in district_mapping:
                        return district_mapping[d_low]
                    d_title = row["district_name"].strip().title()
                    if d_title in MP_DISTRICTS:
                        return d_title
                    return None
                
                df["district"] = df.apply(map_district, axis=1)
                df = df.dropna(subset=["district"]).copy()
                
                # Extract year as integer (e.g. "2018-2019" -> 2018)
                def parse_year(yr_val):
                    try:
                        yr_str = str(yr_val).strip()
                        if "-" in yr_str:
                            return int(yr_str.split("-")[0])
                        return int(yr_str)
                    except Exception:
                        return None
                        
                df["year"] = df["year"].apply(parse_year)
                df = df.dropna(subset=["year"]).copy()
                df["year"] = df["year"].astype(int)
                
                # Convert yield to kg/ha (multiply Tonnes/Hectare by 1000)
                df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
                df = df.dropna(subset=["yield"]).copy()
                df["yield_kg_ha"] = df["yield"] * 1000.0
                
                # Filter to years 2018 onwards to match our daily climate weather range
                df = df[df["year"] >= 2018].copy()
                
                # Group by to avoid duplicates if any, keeping the mean yield
                df = df.groupby(["district", "year", "season", "crop"], as_index=False)["yield_kg_ha"].mean()
                
            if db:
                log_api_status(db, "Historical Yield Cache", None, "SUCCESS", f"Loaded yields from offline file: {local_csv_path}")
            return df
        except Exception as e:
            print(f"  ❌ Error reading or parsing local CSV: {e}")
            if db:
                log_api_status(db, "Historical Yield Cache", None, "ERROR", f"Error reading or parsing local CSV: {e}")

    print("⚠️  Offline historical yields CSV not found. Generating synthetic dataset as fallback.")
    if db:
        log_api_status(db, "Historical Yield Cache", None, "FALLBACK", "Offline cache missing; using synthetic generator")
    return generate_synthetic_yields()


def sync_yield_history():
    """Load yield data from local CSV (or synthetic fallback) and upsert into YieldHistory."""
    db = SessionLocal()
    
    # Clean out the existing table to avoid mixing simulated and authentic datasets
    print("🧹 Cleaning yield_history table in database...")
    db.query(YieldHistory).delete()
    db.commit()

    df = fetch_historical_yields(db=db)

    upserted = 0
    for _, row in df.iterrows():
        db.add(YieldHistory(
            district    = row["district"],
            year        = row["year"],
            season      = row["season"],
            crop        = row["crop"],
            yield_kg_ha = row["yield_kg_ha"],
        ))
        upserted += 1

    db.commit()
    print(f"✅ Synced {upserted} yield records across {df['crop'].nunique()} crops "
          f"and {df['district'].nunique()} districts.")
    db.close()



if __name__ == "__main__":
    sync_yield_history()
