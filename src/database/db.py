from sqlalchemy import create_engine, Column, Integer, String, Float, Date, MetaData
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import date
import os

# Create the SQLite database in the root project folder
DATABASE_URL = "sqlite:///dicra_csi.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class DailyClimate(Base):
    """Stores daily climate variables per district (from DiCRA/IMD/CHIRPS)"""
    __tablename__ = "daily_climate"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True, nullable=False)
    district = Column(String, index=True, nullable=False)
    tmax = Column(Float, nullable=True)          # DiCRA Tmax
    soil_moisture = Column(Float, nullable=True) # DiCRA Soil Moisture
    precipitation = Column(Float, nullable=True) # CHIRPS/IMD Precipitation
    ndvi = Column(Float, nullable=True)          # Sentinel 10m NDVI

class DistrictBaselines(Base):
    """Stores the dynamic historical baselines (rolling 15-year)"""
    __tablename__ = "district_baselines"

    district = Column(String, primary_key=True, index=True)
    tmax_95th = Column(Float, nullable=False)
    sm_5th = Column(Float, nullable=False)
    calculated_on = Column(Date, default=date.today)

class YieldHistory(Base):
    """Stores historical PMFBY CCE yield records for training/validation"""
    __tablename__ = "yield_history"

    id = Column(Integer, primary_key=True, index=True)
    district = Column(String, index=True, nullable=False)
    year = Column(Integer, index=True, nullable=False)
    season = Column(String, nullable=False) # e.g., 'Kharif', 'Rabi'
    crop = Column(String, nullable=False)   # e.g., 'Soybean', 'Wheat'
    yield_kg_ha = Column(Float, nullable=False)

class RiskPredictions(Base):
    """Stores the output CSI Risk Tiers (both historical back-test and CMIP6 future projections)"""
    __tablename__ = "risk_predictions"

    id = Column(Integer, primary_key=True, index=True)
    district = Column(String, index=True, nullable=False)
    year = Column(Integer, index=True, nullable=False)
    scenario = Column(String, index=True, nullable=False) # e.g., 'Historical', 'RCP_4.5', 'RCP_8.5'
    csi_score = Column(Float, nullable=False)             # Probabilistic score (0 to 1)
    risk_tier = Column(Integer, nullable=False)           # Actuarial Tier 1 to 5

def init_db():
    """Creates all tables in the database if they don't exist"""
    print(f"Initializing database at {DATABASE_URL}...")
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully.")

def get_db():
    """Dependency for getting a database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
