from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from src.database.db import get_db, RiskPredictions, DistrictBaselines, APILog
from src.features.phenology_clipper import SUPPORTED_CROPS
import uvicorn

app = FastAPI(
    title       = "CSI-DiCRA Compound Stress Index API",
    description = "Actuarial climate risk tiers for agricultural insurance — Madhya Pradesh.",
    version     = "2.1.0",
)


def _normalise_scenario(scenario: str) -> str:
    """Accept legacy RCP names alongside current SSP names."""
    return scenario.replace("RCP_4.5", "SSP2_4.5").replace("RCP_8.5", "SSP5_8.5")


@app.get("/api/v1/insurance/premium_risk")
def get_premium_risk(
    district: str,
    year:     int,
    crop:     str = "Soybean",
    scenario: str = "SSP5_8.5",
    db: Session   = Depends(get_db),
):
    """
    For insurance companies: returns the CSI risk tier and recommended
    premium adjustment for a given district / year / crop / scenario.
    """
    if crop not in SUPPORTED_CROPS:
        raise HTTPException(
            status_code=400,
            detail=f"Crop '{crop}' not supported. Choose from: {SUPPORTED_CROPS}",
        )
    scenario   = _normalise_scenario(scenario)
    prediction = db.query(RiskPredictions).filter(
        RiskPredictions.district == district,
        RiskPredictions.year     == year,
        RiskPredictions.scenario == scenario,
        RiskPredictions.crop     == crop,
    ).first()

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail=f"No projection found for {district} / {year} / {scenario} / {crop}.",
        )

    adj = (
        f"+{(prediction.risk_tier - 3) * 1.5}%"
        if prediction.risk_tier > 3
        else "Base Rate"
    )
    return {
        "district":                    prediction.district,
        "year":                        prediction.year,
        "scenario":                    prediction.scenario,
        "crop":                        prediction.crop,
        "csi_risk_tier":               prediction.risk_tier,
        "probabilistic_score":         round(prediction.csi_score, 3),
        "recommended_premium_adjustment": adj,
    }


@app.get("/api/v1/dashboard/trajectory")
def get_risk_trajectory(
    district: str,
    crop:     str = "Soybean",
    scenario: str = "SSP5_8.5",
    db: Session   = Depends(get_db),
):
    """
    For the Streamlit dashboard: returns the 15-year risk trajectory
    (2026-2041) for a given district / crop / scenario.
    """
    if crop not in SUPPORTED_CROPS:
        raise HTTPException(
            status_code=400,
            detail=f"Crop '{crop}' not supported. Choose from: {SUPPORTED_CROPS}",
        )
    scenario    = _normalise_scenario(scenario)
    predictions = db.query(RiskPredictions).filter(
        RiskPredictions.district == district,
        RiskPredictions.scenario == scenario,
        RiskPredictions.crop     == crop,
    ).order_by(RiskPredictions.year).all()

    return [
        {"year": p.year, "risk_tier": p.risk_tier, "score": p.csi_score}
        for p in predictions
    ]


@app.get("/api/v1/baselines")
def get_baselines(
    crop: str | None = None,
    db:   Session    = Depends(get_db),
):
    """
    Returns season-aware baselines for all districts.
    Optionally filter by crop (e.g. ?crop=Wheat).
    """
    query = db.query(DistrictBaselines)
    if crop:
        if crop not in SUPPORTED_CROPS:
            raise HTTPException(
                status_code=400,
                detail=f"Crop '{crop}' not supported. Choose from: {SUPPORTED_CROPS}",
            )
        query = query.filter(DistrictBaselines.crop == crop)

    return [
        {
            "district":  b.district,
            "crop":      b.crop,
            "tmax_95th": b.tmax_95th,
            "sm_5th":    b.sm_5th,
        }
        for b in query.all()
    ]


@app.get("/api/v1/crops")
def list_supported_crops():
    """Returns the list of crops the system supports."""
    return {"supported_crops": SUPPORTED_CROPS}


@app.get("/api/v1/health/api_status")
def get_api_status(db: Session = Depends(get_db)):
    """
    Returns the latest 50 API logs from the system for developer diagnostics,
    including successful calls and fallback events.
    """
    logs = db.query(APILog).order_by(APILog.timestamp.desc()).limit(50).all()
    return [
        {
            "id":        log.id,
            "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S") if log.timestamp else None,
            "service":   log.service,
            "district":  log.district,
            "status":    log.status,
            "message":   log.message,
        }
        for log in logs
    ]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)