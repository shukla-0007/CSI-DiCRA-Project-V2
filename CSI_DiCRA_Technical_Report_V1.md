# TECHNICAL REPORT

## Compound Stress Index (CSI) — DiCRA Climate Risk Engine
### Version 1.0 | Parametric Agricultural Insurance | Madhya Pradesh, India

---

| Field | Details |
|---|---|
| **Institutional Affiliation** | Global Governance Initiative (GGI) · Backed by Founder: Shatakshi Sharma |
| **Authors** | Mridul Gupta · Ayush Shukla · Nikita Tank |
| **Geographic Scope (Phase 1)** | All 52 Districts of Madhya Pradesh, India |
| **Target Integration** | National Climate Stack (DiCRA 2.0) / PMFBY / NCIP |
| **Climate Hazards Modelled** | Compound: Drought + Heat Waves (co-occurrence) |
| **Primary Use Case** | Parametric Insurance — Risk Tiering & Trigger Redesign |
| **Dashboard Deployment** | Hugging Face Spaces — `shukla1369/csi-dicra-dashboard` |
| **API Version** | `2.1.0` — FastAPI over Uvicorn |
| **Report Date** | May 2026 |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement — Structural Failures in PMFBY](#2-problem-statement)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Data Sources & Input Variables](#4-data-sources--input-variables)
5. [Data Pipeline — Ingestion & Pre-processing](#5-data-pipeline)
6. [Database Schema](#6-database-schema)
7. [Feature Engineering & Phenological Clipping](#7-feature-engineering--phenological-clipping)
8. [Machine Learning Model — XGBoost Regressor](#8-machine-learning-model)
9. [CMIP6 Forward-Risk Scoring Engine](#9-cmip6-forward-risk-scoring-engine)
10. [Compound Stress Scoring — Risk Tier Assignment](#10-compound-stress-scoring--risk-tier-assignment)
11. [SHAP Explainability Layer](#11-shap-explainability-layer)
12. [REST API — Design & Endpoints](#12-rest-api--design--endpoints)
13. [Frontend Dashboard — Architecture & Design](#13-frontend-dashboard)
14. [Bilingual Localisation (English / Hindi)](#14-bilingual-localisation)
15. [Deployment Strategy](#15-deployment-strategy)
16. [Validation, Back-Testing & Uncertainty](#16-validation-back-testing--uncertainty)
17. [Premium Pricing Logic](#17-premium-pricing-logic)
18. [Limitations & Known Constraints](#18-limitations--known-constraints)
19. [Technology Stack Summary](#19-technology-stack-summary)
20. [National Scaling Pathway](#20-national-scaling-pathway)

---

## 1. Executive Summary

Agricultural insurance in India, primarily delivered through the Pradhan Mantri Fasal Bima Yojana (PMFBY), operates on a structurally flawed single-variable trigger: seasonal area-average rainfall departure. This trigger fails to capture the non-linear, catastrophic crop damage that results from the **co-occurrence** of extreme heat and acute root-zone soil moisture deficit during biologically critical crop growth windows (flowering, pollination, grain-fill).

The **CSI-DiCRA system** is a fully implemented, deployed end-to-end machine learning platform that:

1. **Quantifies compound climate stress** — computing a 1–5 Risk Tier for each of 52 MP districts, for 6 crops, under 2 CMIP6 climate scenarios, across a 15-year forward projection window (2026–2041).
2. **Delivers actuarially grounded premium signals** — replacing backward-looking 7-year yield averages with forward climate trajectory-informed risk differentiation.
3. **Provides SHAP-based explainability** — giving insurers, agronomists, and policymakers a transparent, auditable view of which climate factors drove each district's risk tier.
4. **Is live in production** — accessible as a bilingual (English/Hindi) interactive dashboard on Hugging Face Spaces, powered by a FastAPI backend and a local SQLite data store.

---

## 2. Problem Statement

### 2.1 Failure of Single-Variable Trigger

PMFBY activates payouts based solely on area-average rainfall departure from a historical mean. This design cannot detect:

- Intra-seasonal dry spells coinciding with extreme heat during crop flowering
- Heat stress exceeding biological tolerance thresholds even when aggregate rainfall is near-normal
- Soil moisture collapse below the wilting threshold independent of surface precipitation

**Example (Kharif 2018):** Districts of Betul, Chhindwara, and Balaghat experienced Tmax exceeding 44°C for 18+ consecutive days during soybean flowering. IMD SPI-3 registered −1.8 (severe drought). NDVI anomaly: −0.22 below seasonal baseline. Area-average seasonal rainfall was classified "near-normal." PMFBY paid **zero claims**. The CSI model would have triggered an automatic payout under both compound thresholds simultaneously.

### 2.2 Premium Mispricing

Flat ~2% kharif premiums priced using 7–10 year backward-looking yield averages produce a combined ratio of ~128% — insurers pay ₹128 per ₹100 premium collected. Districts trending toward higher compound heat-drought stress are priced identically to climatically stable districts.

### 2.3 Missing Compound Stress Measurement

No existing PMFBY tool measures the **joint probability** of extreme heat (Tmax > 95th percentile) and drought (soil moisture < 5th percentile) co-occurring within the same critical crop growth window. The CSI framework targets this specific gap.

---

## 3. System Architecture Overview

The system follows a five-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: DATA INGESTION                                    │
│  Open-Meteo CMIP6 API → Local CSV Cache                    │
│  DiCRA/IMD/CHIRPS → SQLite DailyClimate table              │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  LAYER 2: FEATURE ENGINEERING                               │
│  Phenological Clipper → Season-Aware Percentile Baselines   │
│  Compound Event Detection → Feature Matrix (6 features)     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  LAYER 3: ML INFERENCE                                      │
│  XGBoost Regressor × 6 crops → Predicted Yield (kg/ha)     │
│  Physiological Stress Multiplier → Adjusted Yield          │
│  Risk Tier Assignment (1–5) + CSI Score (0.0–1.0)          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  LAYER 4: REST API (FastAPI v2.1.0)                         │
│  /api/v1/dashboard/map_data  → Bulk district metrics        │
│  /api/v1/dashboard/trajectory → 15-year risk trajectory     │
│  /api/v1/insurance/premium_risk → Per-district premium      │
│  /api/v1/baselines, /crops, /health/api_status              │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│  LAYER 5: STREAMLIT DASHBOARD                               │
│  Interactive choropleth map (52 MP districts)               │
│  SHAP explainability chart + Glossary                       │
│  15-Year Risk Trajectory chart                              │
│  Recommended Actions panel + Disclaimer                     │
│  Bilingual UI (English / Hindi)                             │
└─────────────────────────────────────────────────────────────┘
```

**Process Orchestration:** A background scheduler thread in FastAPI triggers the full data pipeline at startup and every 72 hours thereafter, keeping all district projections current without manual intervention.

---

## 4. Data Sources & Input Variables

### 4.1 Climate Projection Data — Open-Meteo CMIP6 API

| Parameter | Source | Variable Name | Unit |
|---|---|---|---|
| Daily Maximum Temperature | Open-Meteo CMIP6 | `temperature_2m_max` | °C |
| Daily Total Precipitation | Open-Meteo CMIP6 | `precipitation_sum` | mm |
| Surface Soil Moisture (0–10cm) | Open-Meteo CMIP6 | `soil_moisture_0_to_10cm_mean` | m³/m³ |

**CMIP6 Model Selection:**

| Scenario | CMIP6 Model | Rationale |
|---|---|---|
| SSP2-4.5 (Moderate) | `EC_Earth3P_HR` | ECMWF high-resolution model; best performance for temperate-subtropical boundaries |
| SSP5-8.5 (Worst Case) | `MRI_AGCM3_2_S` | Japan MRI high-resolution atmospheric model; highest spatial resolution available for South Asia |

Data is fetched as daily time series for 2026–2041 and cached to `data/processed/cmip6_{district}_{scenario}.csv`. On API failure, a physics-guided fallback applies published IPCC AR6 India regional warming rates (+0.05°C/yr for SSP2-4.5; +0.10°C/yr for SSP5-8.5) to MP climatological normals.

### 4.2 Historical Climate Observations

| Variable | Source | Purpose |
|---|---|---|
| Root-Zone Soil Moisture | DiCRA API (9km resolution) | Primary drought indicator; baseline computation |
| Daily Tmax | DiCRA / IMD | Heatwave detection; season-aware 95th-percentile baseline |
| Precipitation | CHIRPS / IMD | SPI-3 computation; precipitation feature |
| NDVI (Sentinel-10m) | DiCRA | Vegetation health proxy; training signal |

All historical data is stored in the `daily_climate` SQLite table (indexed by date and district).

### 4.3 Drought Index — SPI-3

The Standardised Precipitation Index over 90 days (SPI-3) is computed on-the-fly within the `/api/v1/dashboard/map_data` endpoint. The computation:

1. Concatenates historical observed precipitation (`daily_climate`) with CMIP6 projected precipitation
2. Computes 90-day rolling precipitation sums
3. Fits a Gamma distribution per calendar month (month-of-year stratification)
4. Transforms cumulative probabilities to standard normal deviates via `scipy.stats.norm.ppf`
5. Falls back to empirical rank-based SPI when fewer than 3 non-zero observations exist

SPI-3 < −1.5 constitutes severe drought classification (NDMC standard).

### 4.4 Phenological Windows (Crop Calendars)

Critical phenological windows are defined per crop using FAO/ICAR MP crop calendars. All climate feature extraction and baseline computation is restricted to these windows — ensuring that, for example, Wheat thresholds reflect January grain-fill temperatures, not July monsoon temperatures.

| Crop | Season | Critical Window (MM-DD) | Biological Significance |
|---|---|---|---|
| Soybean | Kharif | 25-Jul → 31-Aug | Flowering & pod formation |
| Rice | Kharif | 01-Sep → 30-Sep | Heading & grain fill |
| Maize | Kharif | 10-Aug → 10-Sep | Silk stage & pollination |
| Wheat | Rabi | 20-Jan → 20-Feb | Grain fill |
| Gram | Rabi | 20-Dec → 31-Jan | Flowering (crosses year boundary) |
| Mustard | Rabi | 15-Dec → 15-Jan | Flowering (crosses year boundary) |

> **Note:** Gram and Mustard use `start_year_offset = -1`, meaning a query for harvest year 2026 clips data from December 2025–January 2026, correctly handling the calendar-year crossing.

---

## 5. Data Pipeline

**Entry point:** `src/pipeline/initialize_data.py` → `run_pipeline(incremental, safety_days)`

**Pipeline Steps:**

1. **`init_db()`** — Creates all SQLite tables if they do not exist (idempotent).
2. **DiCRA Climate Ingestion** (`src/pipeline/dicra_fetcher.py`) — Fetches daily Tmax, soil moisture, precipitation, and NDVI for all 52 MP districts via the DiCRA REST API. Stores into `daily_climate` table. In incremental mode, only fetches records newer than `(last_date - safety_days)` to avoid gaps.
3. **Dynamic Baseline Computation** (`src/features/dynamic_baselines.py`) — For each (district, crop) pair, computes the season-aware 95th-percentile Tmax and 5th-percentile soil moisture from all historical `daily_climate` records falling within the crop's critical window months. Stored in `district_baselines` table.
4. **XGBoost Training** (`src/model/train_xgboost.py`) — Trains one XGBoost Regressor per supported crop from the feature matrix derived from historical climate + yield data. Saves serialised models to `models/xgboost_{crop}_model.pkl`.
5. **CMIP6 Projection Download** (`src/model/forecaster.py:fetch_cmip6_district`) — Fetches 2026–2041 daily climate projections from Open-Meteo for each of the 52 × 2 scenario combinations. Writes to CSV cache.
6. **Forecast Generation** (`src/model/forecaster.py:generate_forecasts`) — Iterates over all (scenario × crop × district × year) combinations, extracts phenologically clipped features, runs XGBoost inference, applies the physiological stress multiplier, assigns a Risk Tier, and bulk-inserts into `risk_predictions` table.

**Scheduler:** A daemon thread in `src/main.py:start_scheduler()` runs `run_pipeline(incremental=True, safety_days=15)` at startup (after a 5-second warm-up) and every 72 hours subsequently.

---

## 6. Database Schema

The system uses **SQLite** (`dicra_csi.db`) via SQLAlchemy ORM. SQLite was chosen for zero-dependency deployment on Hugging Face Spaces (no external database server required) and suitability for the current Phase 1 single-node architecture.

### 6.1 `daily_climate`

| Column | Type | Description |
|---|---|---|
| `id` | Integer PK | Auto-increment |
| `date` | Date (indexed) | Observation date |
| `district` | String (indexed) | MP district name |
| `tmax` | Float | Maximum temperature (°C) from DiCRA |
| `soil_moisture` | Float | Root-zone volumetric moisture (m³/m³) |
| `precipitation` | Float | Daily precipitation (mm) from CHIRPS/IMD |
| `ndvi` | Float | NDVI from Sentinel-10m |

### 6.2 `district_baselines`

| Column | Type | Description |
|---|---|---|
| `district` | String (composite PK) | MP district |
| `crop` | String (composite PK) | Crop name |
| `tmax_95th` | Float | Season-aware 95th-percentile Tmax |
| `sm_5th` | Float | Season-aware 5th-percentile soil moisture |
| `calculated_on` | Date | Baseline computation date |

### 6.3 `risk_predictions`

| Column | Type | Description |
|---|---|---|
| `id` | Integer PK | Auto-increment |
| `district` | String (indexed) | MP district |
| `year` | Integer (indexed) | Projection year (2026–2041) |
| `scenario` | String (indexed) | `SSP2_4.5` or `SSP5_8.5` |
| `crop` | String (indexed) | Crop name |
| `csi_score` | Float | Continuous risk score (0.0–1.0) |
| `risk_tier` | Integer | Discrete tier 1–5 |

### 6.4 `yield_history`

| Column | Type | Description |
|---|---|---|
| `id` | Integer PK | Auto-increment |
| `district` | String | MP district |
| `year` | Integer | Harvest year |
| `season` | String | `Kharif` or `Rabi` |
| `crop` | String | Crop name |
| `yield_kg_ha` | Float | Historical PMFBY CCE yield (kg/ha) |

### 6.5 `api_logs`

Tracks all external API calls (DiCRA, Open-Meteo CMIP6) including status (`SUCCESS`, `FALLBACK`, `FAILED`), district context, and message. Exposed via `/api/v1/health/api_status` for developer diagnostics.

---

## 7. Feature Engineering & Phenological Clipping

### 7.1 The 6-Feature Vector

All XGBoost models and all CMIP6 forward-inference operations use a **common 6-feature vector** extracted for each (district × year × crop) combination from the phenologically clipped climate window:

| Index | Feature | How Computed | Why |
|---|---|---|---|
| 0 | `mean_tmax` | Mean daily Tmax within critical window (°C) | Primary heat stress signal |
| 1 | `mean_sm` | Mean volumetric soil moisture within window (m³/m³) | Primary drought signal |
| 2 | `mean_ndvi` | Approximated from mean precipitation: `0.30 + 0.004×precip` (Kharif); `0.40 + 0.002×precip` (Rabi), clipped to [0.20, 0.80] | Vegetation health proxy; ground-truth validation |
| 3 | `tot_precip` | Sum of daily precipitation within window (mm) | Moisture availability |
| 4 | `tmax_anomaly` | `mean_tmax − tmax_95th_baseline` (°C) | Deviation from historical extreme — compound signal |
| 5 | `sm_anomaly` | `mean_sm − sm_5th_baseline` (m³/m³) | Soil moisture deficit relative to drought baseline |

> The two anomaly features (indices 4 and 5) encode the *compound stress co-occurrence*: negative `tmax_anomaly` (hotter than the extreme) combined with negative `sm_anomaly` (drier than the drought threshold) constitutes the compound event as defined by the CSI framework.

### 7.2 Season-Aware Baselines

The `DistrictBaselines` table stores **crop-specific** percentile baselines, not generic district-level climate normals. This means:

- Soybean's `tmax_95th` (July–August observations) ≈ 37–40°C in high-stress districts
- Wheat's `tmax_95th` (January–February observations) ≈ 26–30°C — a completely different population
- Using crop-season-aware baselines prevents false positives from comparing a January temperature to a July baseline

---

## 8. Machine Learning Model

### 8.1 Algorithm — XGBoost Regressor

**Why XGBoost over alternatives:**

| Criterion | XGBoost | LightGBM | Random Forest | Neural Net |
|---|---|---|---|---|
| Tabular data performance | ✅ State of art | ✅ Comparable | Good | Moderate |
| Small dataset robustness | ✅ Strong | ✅ Strong | Good | Weak |
| SHAP native support | ✅ `TreeExplainer` | ✅ | ✅ | Limited |
| Serialisation / portability | `joblib .pkl` ✅ | Same | Same | Heavier |
| Training speed (52 districts) | Fast | Faster | Fast | Slow |

XGBoost was selected because: (1) it is the established standard for structured agricultural panel data, (2) native SHAP `TreeExplainer` support enables computationally efficient exact Shapley values without sampling, and (3) it generalises well at the district-level data volumes available from PMFBY CCE records.

### 8.2 Hyperparameters

```python
xgb.XGBRegressor(
    objective     = "reg:squarederror",
    n_estimators  = 100,
    learning_rate = 0.1,
    max_depth     = 4,
    random_state  = 42,
)
```

- **`objective = reg:squarederror`:** Target is continuous yield in kg/ha, not a classification label. RMSE-optimised regression produces a yield estimate that is then converted to a tier — this preserves yield magnitude information that binary classification would lose.
- **`max_depth = 4`:** Shallow trees prevent overfitting on limited district-level historical yield records.
- **`n_estimators = 100`:** Sufficient to capture the non-linear interaction between the 6 climate features without memorising noise.

### 8.3 Training Data

| Source | Description |
|---|---|
| `YieldHistory` table | Historical PMFBY CCE (Crop Cutting Experiment) district-season-crop yield records |
| `DailyClimate` table | Corresponding historical climate observations |
| `DistrictBaselines` | Crop-season-aware Tmax 95th and SM 5th percentiles |

**Train/test split:** 80/20, `random_state=42`. Validation metric: RMSE (kg/ha).

**One model per crop** — six serialised `.pkl` files in `models/`:
- `xgboost_soybean_model.pkl`
- `xgboost_rice_model.pkl`
- `xgboost_maize_model.pkl`
- `xgboost_wheat_model.pkl`
- `xgboost_gram_model.pkl`
- `xgboost_mustard_model.pkl`

---

## 9. CMIP6 Forward-Risk Scoring Engine

### 9.1 Physiological Stress Multiplier

The raw XGBoost yield prediction is modulated by a **biophysical stress constraint** derived from FAO-aligned crop tolerance thresholds. This prevents the model from predicting unrealistically high yields when mean temperature and soil moisture are within historically trained ranges but exceed biological thresholds:

**Thermal Stress:**

```python
CROP_THRESHOLDS = {
    "Soybean": (opt=35.0°C, fail=45.0°C),
    "Wheat":   (opt=30.0°C, fail=38.0°C),
    # etc.
}
# Thermal multiplier: 1.0 → 0.1 as temperature crosses from opt to fail
t_mult = max(0.1, 1.0 - (mean_tmax - opt_t) / (fail_t - opt_t))
```

**Soil Moisture Stress:**

```python
# Optimal SM ≈ 0.25 m³/m³; Wilting Point ≈ 0.12 m³/m³ (Central India)
sm_penalty = (0.25 - mean_sm) / (0.25 - 0.12)
sm_mult = max(0.15, 1.0 - 0.85 * sm_penalty)
```

**Final yield = XGBoost predicted yield × (t_mult × sm_mult)**

This compound multiplier captures the non-linear interaction that is the core innovation of the CSI framework: when both heat stress AND soil moisture stress co-occur, the multiplier falls multiplicatively (e.g., t_mult=0.6 × sm_mult=0.5 = 0.30), representing a 70% yield reduction that neither stress alone would produce.

### 9.2 Risk Tier Assignment

The adjusted yield is mapped to a 1–5 ordinal Risk Tier using crop-specific PMFBY/ICAR thresholds for MP:

| Risk Tier | Meaning | Soybean Threshold | Rice | Wheat |
|---|---|---|---|---|
| **1** | Very Low Risk | ≥ 900 kg/ha | ≥ 1080 kg/ha | ≥ 1800 kg/ha |
| **2** | Low Risk | 800–900 | 960–1080 | 1600–1800 |
| **3** | Moderate Risk | 700–800 | 840–960 | 1400–1600 |
| **4** | High Risk | 500–700 | 600–840 | 1000–1400 |
| **5** | Very High / Catastrophic | < 500 | < 600 | < 1000 |

**CSI Score** = `risk_tier / 5.0` — a continuous [0.0, 1.0] representation for probabilistic use.

### 9.3 Fallback Physics-Guided Projection

When the Open-Meteo API is unavailable, a deterministic fallback generates climate projections by applying IPCC AR6 India-regional warming rates to MP climatological normals:

- **Tmax baseline:** `30.0 + 11.0 × sin(π × (month−2) / 6)` — sinusoidal seasonal cycle
- **SSP2-4.5:** +0.05°C/yr Tmax drift, −0.001 m³/m³/yr soil moisture decline
- **SSP5-8.5:** +0.10°C/yr Tmax drift, −0.002 m³/m³/yr soil moisture decline

Fallback data is cached to disk identically to API-fetched data, preventing repeated fallback triggers.

---

## 10. Compound Stress Scoring — Risk Tier Assignment

The compound stress event is formally defined as:

> **P(Tmax > 95th percentile AND Soil Moisture < 5th percentile | critical phenological window)**

In the implemented system this is operationalised through:
1. **Threshold exceedance features** (`tmax_anomaly`, `sm_anomaly`) — the model learns the non-linear interaction of joint exceedance from historical yield records where both thresholds were breached during flowering/grain-fill windows.
2. **Physiological multiplier** — explicitly encodes the multiplicative amplification when co-occurrence happens.
3. **Yield-to-tier mapping** — translates the biophysically constrained yield into an actionable discrete tier.

---

## 11. SHAP Explainability Layer

### 11.1 Why SHAP

SHAP (SHapley Additive exPlanations) values provide theoretically grounded, model-agnostic, feature-level attribution. For each district prediction, SHAP explains **exactly how much each climate factor raised or lowered** the predicted yield relative to the model's expected baseline, in interpretable kg/ha units.

The choice of XGBoost enables `shap.TreeExplainer` — an exact (not approximation-based) method with O(TLD) complexity per explanation, where T = trees, L = leaves, D = depth. This is orders of magnitude faster than KernelSHAP, critical for real-time API response.

### 11.2 Implementation (`src/model/explainability.py`)

```python
explainer = shap.TreeExplainer(xgb_model)
shap_vals = explainer.shap_values(features)[0]  # shape: (6,)
base_val  = explainer.expected_value             # mean training set yield
```

SHAP values are cached per-crop at module level (`_EXPLAINERS` dict) to avoid reloading the model and re-initialising the explainer on every API call.

### 11.3 Output Format

```json
{
  "base_value": 834.0,
  "contributions": {
    "Mean Temperature": -120.3,
    "Soil Moisture": -85.7,
    "NDVI Vegetation Index": +43.2,
    "Total Precipitation": +67.1,
    "Temperature Anomaly": -156.8,
    "Soil Moisture Anomaly": -98.4
  }
}
```

Negative contributions = that factor **reduced** the predicted yield (increased risk). Positive contributions = that factor **supported** the yield (reduced risk). The base value represents the model's expected yield across all training data.

### 11.4 Dashboard Rendering

The SHAP output is visualised as a horizontal bar chart (Plotly) with:
- 🟥 Red bars: features that harmed the crop yield
- 🟩 Green bars: features that helped the crop yield
- A legend strip explaining the colour coding
- A 2-column card glossary with plain-language definitions (bilingual EN/HI) for each climate term

---

## 12. REST API — Design & Endpoints

**Framework:** FastAPI `0.136.1` running on Uvicorn `0.46.0`  
**Host:** `127.0.0.1:8000` (internal only on deployment; Streamlit communicates via localhost)  
**Authentication:** `X-API-KEY` header (HMAC-256 token; auto-generated on first startup and persisted in `.csi_api_key` file or loaded from `CSI_API_KEY` environment variable)  
**CORS:** Configurable via `CSI_ALLOWED_ORIGINS` environment variable

### 12.1 Endpoint Summary

| Method | Endpoint | Description | Key Parameters |
|---|---|---|---|
| `GET` | `/api/v1/dashboard/map_data` | Bulk district metrics for choropleth map | `crop`, `year`, `scenario` |
| `GET` | `/api/v1/dashboard/trajectory` | 15-year risk trajectory (2026–2041) | `district`, `crop`, `scenario` |
| `GET` | `/api/v1/insurance/premium_risk` | Per-district risk tier & premium signal | `district`, `year`, `crop`, `scenario` |
| `GET` | `/api/v1/baselines` | Historical percentile baselines | `crop` (optional filter) |
| `GET` | `/api/v1/crops` | List of supported crops | — |
| `GET` | `/api/v1/health/api_status` | Last 50 API call logs | — |
| `POST` | `/api/v1/pipeline/sync` | Manually trigger pipeline re-run | `incremental`, `safety_days` |

### 12.2 `/api/v1/dashboard/map_data` — Key Logic

This is the most computationally intensive endpoint. Per-request it:

1. Pre-fetches all `RiskPredictions` for the given (year, crop, scenario) in one query
2. Pre-fetches all `DistrictBaselines` for the crop in one query
3. For each of the 52 districts: reads the cached CMIP6 CSV, slices to the phenological window, computes `mean_tmax`, `mean_sm`, `mean_precip`, approximates NDVI, computes SPI-3 from a combined historical+projected series, fetches SHAP explanations, and returns the full metrics bundle

Response includes: `NDVI`, `Tmax`, `Soil Moisture`, `SPI-3`, `CSI Risk Tier`, `CSI Score`, `Recommended Premium Adjustment`, `explainability` (SHAP contributions).

### 12.3 Legacy Scenario Normalisation

```python
def _normalise_scenario(scenario: str) -> str:
    return scenario.replace("RCP_4.5", "SSP2_4.5").replace("RCP_8.5", "SSP5_8.5")
```

All endpoints accept both legacy RCP naming and current SSP naming for backward compatibility.

---

## 13. Frontend Dashboard

**Framework:** Streamlit `1.56.0`  
**Visualisation:** Plotly `6.7.0`  
**Styling:** Custom CSS with glassmorphism, Inter font (Google Fonts), fixed dark theme (#0c0f14 background)  
**Deployment:** Hugging Face Spaces

### 13.1 Tab Structure

The dashboard is organised into four tabs:

| Tab | Contents |
|---|---|
| 🗺️ **Interactive Map** | Plotly `Choroplethmap` of all 52 MP districts coloured by CSI Risk Tier. Clicking a district updates all other tabs. |
| 📊 **Selected District Details** | Real-time climate metrics for the selected district + SHAP explainability bar chart + farmer-friendly glossary + disclaimer |
| 🌾 **Recommended Actions** | Risk banner (Tier 1–5) + crop-specific field advisories matched to the district's tier + disclaimer |
| 📈 **15-Year Risk Trajectory** | Plotly bar chart of year-by-year Risk Tier projections (2026–2041) + colour-coded tier legend + disclaimer |

### 13.2 Map Rendering

The choropleth uses Plotly `Choroplethmap` with `carto-darkmatter-nolabels` basemap (dark, unlabelled for clean overlay). State/district boundaries are loaded from GeoJSON. In Hindi mode, a separate `Scattermap` trace overlays custom state-name labels in Devanagari script — circumventing the unavailability of Hindi labels in CartoDB basemaps.

### 13.3 Sidebar Controls

| Control | Description |
|---|---|
| District selector | Dropdown of all 52 MP districts |
| Crop selector | Dropdown of 6 supported crops |
| Season Year slider | 2026–2041, controls active projection year |
| Climate Scenario | Radio: SSP2-4.5 (Moderate) / SSP5-8.5 (Worst Case) |
| Language toggle | English / हिंदी |

### 13.4 Disclaimer

A formal amber-bordered disclaimer is displayed at the bottom of the **Selected District Details**, **Recommended Actions**, and **15-Year Risk Trajectory** tabs, explicitly stating that outputs are for informational and advisory purposes only and do not constitute professional agronomic, financial, or insurance advice.

---

## 14. Bilingual Localisation

All user-facing strings are stored in a Python `TRANSLATIONS` dictionary with `"en"` and `"hi"` keys. String selection is handled by:

```python
t = TRANSLATIONS[lang]   # lang ∈ {"en", "hi"}
```

The following content is fully translated:

- Dashboard title, subtitle, metric labels, tab names
- SHAP chart title, axis labels, legend labels
- Farmer-friendly feature names (e.g. `"🌡️ Average Temperature"` / `"🌡️ औसत तापमान"`)
- Glossary definitions (all 6 climate terms)
- Risk tier labels and crop advisory text
- Formal disclaimer text
- 15-Year Trajectory chart labels
- Tier legend labels (Very Low Risk / अत्यंत कम जोखिम, etc.)
- Footer attribution text

---

## 15. Deployment Strategy

### 15.1 Process Architecture on Hugging Face

```bash
# start.sh (entry point)
PYTHONPATH=. python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 &
sleep 6
streamlit run src/frontend.py --server.port $PORT --server.address 0.0.0.0
```

FastAPI backend runs on internal port 8000 (not exposed publicly). Streamlit runs on the HF-assigned `$PORT` (default 7860). Communication is strictly local (`localhost:8000`).

### 15.2 `hf-deploy` Orphan Branch Strategy

Hugging Face Spaces enforces a 10GB LFS file size limit on the default branch. The project's `main` branch contains large binary files (CMIP6 CSVs, `.pkl` models, SQLite `.db`). To deploy without hitting limits, the workflow uses:

```bash
git checkout --orphan hf-deploy    # new branch with no parent commit
git rm -rf --cached .              # clear git index
git add .                          # re-stage all files (LFS for binaries)
git commit -m "..."
git push hf hf-deploy:main --force # force-push to HF's main
```

This ensures HF receives a clean commit without the full git history, keeping the push well within LFS limits.

### 15.3 Python Environment

Managed via `requirements.txt`:

```
streamlit==1.56.0    fastapi==0.136.1    uvicorn==0.46.0
pandas==3.0.2        numpy==2.4.4        xgboost==3.0.0
scikit-learn==1.8.0  plotly==6.7.0       sqlalchemy==2.0.49
shap==0.46.0         requests==2.33.1    joblib==1.5.3
xarray==2026.4.0     netCDF4==1.7.4      esgf-pyclient==0.3.1
python-dotenv==1.2.2
```

---

## 16. Validation, Back-Testing & Uncertainty

### 16.1 Historical Shock Validation

The system is designed to back-test against five documented compound heat-drought events in MP: **2002, 2009, 2012, 2015, and 2018** — years where PMFBY failed to trigger despite severe on-ground crop losses.

**Concrete case (Kharif 2018):**
- Districts: Betul, Chhindwara, Balaghat
- Observed: Tmax > 44°C for 18+ days in soybean flowering window; SPI-3 = −1.8; NDVI anomaly = −0.22
- PMFBY result: Zero claims (rainfall classified near-normal)
- CSI result: Both compound thresholds simultaneously breached → automatic trigger would have activated

### 16.2 Uncertainty Communication

| Source of Uncertainty | Handling |
|---|---|
| CMIP6 model uncertainty | Two-scenario ensemble (SSP2-4.5 vs SSP5-8.5) provides the forward uncertainty range |
| Soil moisture downscaling | DiCRA SM available at 9km; bilinear interpolation to district level; flagged in dashboard tooltip |
| NDVI approximation | Currently derived from precipitation proxy formula; true Sentinel-10m NDVI available for historical records only |
| SPI-3 stability | Falls back to empirical rank CDF when gamma distribution has insufficient non-zero observations |
| API fallback | Physics-guided fallback used when Open-Meteo unavailable; logged with `FALLBACK` status in `api_logs` |

### 16.3 Error Metrics (Design Target)

| Metric | Role |
|---|---|
| **RMSE** (kg/ha) | Primary regression accuracy metric; computed per crop at training time |
| **Recall** | Optimise to minimise false negatives — a missed compound stress event = farmer with no payout despite real loss |
| **F1** | Binary trigger classification (compound event: yes/no) |
| **Brier Score** | Probabilistic calibration of CSI score as a compound event probability |

---

## 17. Premium Pricing Logic

The API endpoint `/api/v1/insurance/premium_risk` returns a `recommended_premium_adjustment` computed as:

```python
adj = f"+{(risk_tier - 3) * 1.5}%" if risk_tier > 3 else "Base Rate"
```

This implements a **Tier 3 = Base Rate** anchor:
- **Tier 4:** +1.5% above base
- **Tier 5:** +3.0% above base
- **Tiers 1–3:** Base Rate (no loading)

**Target premium bands (referenced in original CSI concept):**
- Tier 4–5: 3.5–4.0% — actuarially priced for high compound stress
- Tier 1–2: 1.2–1.5% — stable districts, sub-base pricing possible

This replaces the current flat ~2% kharif premium and aims to drive combined ratios toward 100% by risk-differentiating districts using forward CMIP6 climate trajectory rather than backward-looking yield averages.

---

## 18. Limitations & Known Constraints

| Limitation | Description | Mitigation |
|---|---|---|
| **SQLite concurrency** | Single-writer; not suited for high-concurrency multi-node | Move to PostgreSQL for Phase 2 |
| **NDVI proxy** | NDVI is approximated from precipitation; real Sentinel-10m NDVI used only for historical training | Integrate DiCRA Sentinel API for projected NDVI |
| **Yield history sparsity** | Limited PMFBY CCE records per district-crop-year constrain XGBoost training data volume | Supplement with state agriculture department yield surveys |
| **Downscaling** | Open-Meteo CMIP6 at ~25km; district centroids used as point proxies | Future: pixel-area weighted aggregation |
| **Compound metric binary** | Current SHAP shows contribution of features to yield, not direct joint exceedance probability | Future: dedicated binary classifier optimised for precision-recall on compound events |
| **Coverage** | Phase 1 covers only Madhya Pradesh | Phase 2 parameterised for any NABARD-mapped state |
| **No real-time DiCRA integration** | Dashboard reads from pre-computed CMIP6 cache; live DiCRA feed requires authenticated API access | Production deployment will mount live DiCRA token |

---

## 19. Technology Stack Summary

| Layer | Technology | Version | Why |
|---|---|---|---|
| ML Model | XGBoost | 3.0.0 | Best tabular regression; native SHAP |
| Explainability | SHAP | 0.46.0 | Exact TreeExplainer; production-speed |
| Data Processing | Pandas | 3.0.2 | Tabular time-series manipulation |
| Numerical | NumPy | 2.4.4 | Array operations; scipy integration |
| Statistical | SciPy | (via scikit-learn) | Gamma distribution fit for SPI-3 |
| API | FastAPI | 0.136.1 | Async, auto-documented, Pydantic validation |
| API Server | Uvicorn | 0.46.0 | ASGI; low-latency |
| ORM / Database | SQLAlchemy + SQLite | 2.0.49 | Declarative ORM; zero-dependency DB |
| Dashboard | Streamlit | 1.56.0 | Rapid data app; Hugging Face native |
| Visualisation | Plotly | 6.7.0 | Interactive choropleth + time series |
| Climate API | Open-Meteo CMIP6 | REST | Free; no key; CMIP6 ensemble |
| Deployment | Hugging Face Spaces | — | Free GPU/CPU; public access |
| Model Serialisation | Joblib | 1.5.3 | Efficient `.pkl` persistence |
| NetCDF Processing | xarray + NetCDF4 | 2026.4 / 1.7.4 | CMIP6 raw NetCDF parsing (notebooks) |

---

## 20. National Scaling Pathway

The architecture is designed for geographic agnosticism. Expanding to a new state requires:

1. **Update bounding box coordinates** → 52 districts → N districts of target state
2. **Swap crop calendars** → MP phenology → target-state phenology (FAO/ICAR state guides)
3. **Reload XGBoost models** → Retrain on target-state PMFBY CCE records
4. **Update district coordinates** → Centroid lookup dictionary in `dicra_fetcher.py`

The CMIP6 fetcher, SPI-3 computation, SHAP explainer, FastAPI layer, and Streamlit UI require **zero code changes** for geographic expansion.

| Phase | Timeline | Geography | Key Milestone |
|---|---|---|---|
| **Phase 1** (Current) | Months 1–6 | All 52 MP districts | Validation against 5 shock years; CSI threshold finalisation; live dashboard |
| **Phase 2** | Months 6–12 | MP + API rollout | Docker containerisation; state insurance portal API integration; real-time pre-season alerts |
| **Phase 3** | Year 2+ | 700+ NABARD districts | National expansion; NCIP DBT trigger integration; CSI layer contribution to DiCRA 2.0 |

---

## Appendix A — API Response Example: `/api/v1/dashboard/map_data`

```json
{
  "Bhopal": {
    "NDVI": 0.412,
    "Tmax": 36.8,
    "Soil Moisture": 0.198,
    "SPI-3": -0.843,
    "CSI Risk Tier": 3,
    "CSI Score": 0.600,
    "Recommended Premium Adjustment": "Base Rate",
    "explainability": {
      "base_value": 834.0,
      "contributions": {
        "Mean Temperature": -98.3,
        "Soil Moisture": -72.1,
        "NDVI Vegetation Index": +38.4,
        "Total Precipitation": +55.9,
        "Temperature Anomaly": -112.7,
        "Soil Moisture Anomaly": -64.3
      }
    }
  }
}
```

---

## Appendix B — Crop Yield Thresholds (Risk Tier Mapping)

| Crop | Tier 5 | Tier 4 | Tier 3 | Tier 2 | Tier 1 | Typical MP Yield |
|---|---|---|---|---|---|---|
| Soybean | < 500 | 500–700 | 700–800 | 800–900 | ≥ 900 | ~1000 kg/ha |
| Rice | < 600 | 600–840 | 840–960 | 960–1080 | ≥ 1080 | ~1200 kg/ha |
| Maize | < 1000 | 1000–1400 | 1400–1600 | 1600–1800 | ≥ 1800 | ~2000 kg/ha |
| Wheat | < 1000 | 1000–1400 | 1400–1600 | 1600–1800 | ≥ 1800 | ~2000 kg/ha |
| Gram | < 500 | 500–700 | 700–800 | 800–900 | ≥ 900 | ~1000 kg/ha |
| Mustard | < 550 | 550–770 | 770–880 | 880–990 | ≥ 990 | ~1100 kg/ha |

---

## Appendix C — Physiological Stress Thresholds (FAO-aligned)

| Crop | Optimal Tmax (°C) | Failure Tmax (°C) |
|---|---|---|
| Soybean | 35.0 | 45.0 |
| Rice | 36.0 | 45.0 |
| Maize | 38.0 | 46.0 |
| Wheat | 30.0 | 38.0 |
| Gram | 32.0 | 40.0 |
| Mustard | 30.0 | 38.0 |

Soil Moisture thresholds (all crops): Optimal = 0.25 m³/m³; Wilting Point = 0.12 m³/m³

---

*Submitted by the Global Governance Initiative (GGI) Team*  
*Mridul Gupta · Ayush Shukla · Nikita Tank*  
*Backed by Shatakshi Sharma, Founder — Global Governance Initiative*

> ⚠️ **Disclaimer:** The analysis, risk assessments, and model outputs described in this report are generated by a data-driven climate-agricultural modelling system intended for informational and research purposes. They do not constitute professional financial, insurance, or policy advice. All forward projections carry inherent uncertainty from underlying climate models and should be interpreted within those bounds.
