---
title: CSI DiCRA Dashboard
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# 🌾 CSI-DiCRA: Compound Stress Index for Climate Resilient Agriculture

A production-ready spatiotemporal machine-learning pipeline and parametric crop insurance advisory system for the 52 districts of Madhya Pradesh, India.

---

## 📌 Project Overview
CSI-DiCRA addresses the **"Basis Risk"** failure in India's Pradhan Mantri Fasal Bima Yojana (PMFBY) crop insurance scheme. Traditional area-average models fail to capture localized compound heat-and-drought shocks during critical crop growth phases.

**The Solution:**
A multi-crop, season-aware Compound Stress Index (CSI) that:
1. Filters daily temperature, soil moisture, and precipitation data to crop-specific critical phenological windows (flowering & grain-fill) using FAO-aligned crop calendars.
2. Computes dynamic season-aware baselines (95th percentile Max Temperature, 5th percentile Soil Moisture) for each district and crop.
3. Computes the Standardized Precipitation Index (SPI-3) using rolling 90-day Gamma-distributed precipitation anomalies.
4. Trains crop-specific XGBoost regressors on authentic historical yields (1997–2023) and climate features.
5. Projects crop-specific actuarial risk tiers (1 to 5) for 2026–2041 across CMIP6 future climate scenarios (SSP2-4.5 and SSP5-8.5).

---

## 🏗 Modular Architecture (`src/`)

The project is structured as a modular Python package:

*   **`src/database/db.py`**: SQLite database setup via SQLAlchemy ORM. Defines tables for `DailyClimate`, `DistrictBaselines`, `YieldHistory`, `RiskPredictions`, and `APILog` (for audit and fallback tracking).
*   **`src/features/`**:
    *   `phenology_clipper.py`: Manages FAO growth windows for 6 supported crops (*Soybean, Rice, Maize, Wheat, Gram, Mustard*), handling year-crossing Rabi crops.
    *   `dynamic_baselines.py`: Computes historical season-aware temperature and soil moisture percentiles from observations.
*   **`src/pipeline/`**:
    *   `dicra_fetcher.py`: Connects to NASA POWER / DiCRA API for daily temperature, soil moisture, and NDVI observations.
    *   `imd_spi_calculator.py`: Fetches precipitation and fits a Gamma distribution to calculate SPI-3.
    *   `pmfby_validation.py`: Ingests authentic yields from `data/mp_historical_yields.csv`.
    *   `initialize_data.py`: Manages the master pipeline run with a thread-safe lock to prevent concurrent database sync conflicts.
*   **`src/model/`**:
    *   `train_xgboost.py`: Retrains the 6 crop regressor models.
    *   `forecaster.py`: Generates future projections for SSP scenarios, utilizing local disk caching (`data/processed/`) to bypass rate-limited API calls.
*   **`src/main.py`**: FastAPI backend featuring:
    *   Global `X-API-KEY` validation.
    *   Configurable CORS middleware.
    *   A background daemon thread that triggers an incremental climate sync on startup and schedules periodic refreshes every 3 days.
    *   Manual sync background tasks triggered via `POST /api/v1/pipeline/sync`.
*   **`src/frontend.py`**: A premium, dark-themed Streamlit dashboard showing:
    *   Crop-specific metric cards and advisory guidelines.
    *   A Plotly 15-year risk trajectory chart.
    *   A sidebar system status panel displaying the latest API sync logs and a "Sync Now" manual trigger.

---

## 🔒 Security & Configurations

The application implements several security hardening features for production deployment:

1.  **API Key Protection (`X-API-KEY`)**: All backend routes are protected.
2.  **Zero-Configuration Setup**: If `CSI_API_KEY` is not provided in the environment, the server automatically generates a secure 64-character token on startup and writes it to `.csi_api_key` (git-ignored). The Streamlit frontend automatically reads this file to seamlessly authenticate server-to-server queries.
3.  **CORS Restriction**: Configurable via `CSI_ALLOWED_ORIGINS` (defaults to wildcard `*`).
4.  **Non-Root Docker Execution**: The Docker container executes under a dedicated `USER appuser` (UID 999). Directory permissions for `/app` are explicitly set to allow SQLite transaction journaling and WAL file writes under non-root access.
5.  **Dependency Pinning**: All packages in `requirements.txt` are pinned to their exact verified versions.

---

## 🚀 Running the Project

### 1. Setup Environment
Copy the example environment file and adjust variables:
```bash
cp .env.example .env
```

### 2. Running Locally
Run the FastAPI backend:
```bash
PYTHONPATH=. python -m uvicorn src.main:app --port 8000
```

In a separate terminal, launch the Streamlit dashboard:
```bash
PYTHONPATH=. streamlit run src/frontend.py --server.port 8501
```

Access the dashboard at `http://localhost:8501`.

### 3. Container Deployment (Docker)
Build the hardened Docker image:
```bash
docker build -t csi-dicra .
```

Run the container (exposing FastAPI on port 8000 and Streamlit on port 8501):
```bash
docker run -p 8000:8000 -p 8501:8501 --env-file .env csi-dicra
```