from sqlalchemy import create_engine, Column, Integer, String, Float, Date
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import date

DATABASE_URL = "sqlite:///dicra_csi.db"

engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


class DailyClimate(Base):
    """Daily climate variables per district (DiCRA / IMD / CHIRPS)."""
    __tablename__ = "daily_climate"

    id             = Column(Integer, primary_key=True, index=True)
    date           = Column(Date,    index=True,  nullable=False)
    district       = Column(String,  index=True,  nullable=False)
    tmax           = Column(Float,   nullable=True)   # DiCRA Tmax (°C)
    soil_moisture  = Column(Float,   nullable=True)   # DiCRA Root-zone SM
    precipitation  = Column(Float,   nullable=True)   # CHIRPS/IMD (mm)
    ndvi           = Column(Float,   nullable=True)   # Sentinel-10m NDVI


class DistrictBaselines(Base):
    """
    Season-aware historical baselines per (district, crop).
    95th-percentile Tmax and 5th-percentile Soil Moisture are computed ONLY
    from DailyClimate records that fall within each crop's critical window
    months, so Wheat thresholds reflect January temperatures, not July.
    """
    __tablename__ = "district_baselines"

    district      = Column(String, primary_key=True, index=True)
    crop          = Column(String, primary_key=True, index=True)  # NEW — composite PK
    tmax_95th     = Column(Float,  nullable=False)
    sm_5th        = Column(Float,  nullable=False)
    calculated_on = Column(Date,   default=date.today)


class YieldHistory(Base):
    """Historical PMFBY CCE yield records for training / validation."""
    __tablename__ = "yield_history"

    id          = Column(Integer, primary_key=True, index=True)
    district    = Column(String,  index=True, nullable=False)
    year        = Column(Integer, index=True, nullable=False)
    season      = Column(String,  nullable=False)   # 'Kharif' | 'Rabi'
    crop        = Column(String,  nullable=False)   # 'Soybean', 'Wheat', …
    yield_kg_ha = Column(Float,   nullable=False)


class RiskPredictions(Base):
    """
    Output CSI Risk Tiers — historical back-test and CMIP6 future projections.
    Now includes crop as a dimension, one row per (district, year, scenario, crop).
    """
    __tablename__ = "risk_predictions"

    id        = Column(Integer, primary_key=True, index=True)
    district  = Column(String,  index=True, nullable=False)
    year      = Column(Integer, index=True, nullable=False)
    scenario  = Column(String,  index=True, nullable=False)  # SSP2_4.5 | SSP5_8.5
    crop      = Column(String,  index=True, nullable=False)  # NEW
    csi_score = Column(Float,   nullable=False)              # 0.0 – 1.0
    risk_tier = Column(Integer, nullable=False)              # 1 – 5


def init_db():
    """Create all tables if they do not already exist."""
    print(f"Initializing database at {DATABASE_URL}...")
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully.")


def get_db():
    """FastAPI dependency — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
