from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from src.database.db import get_db, RiskPredictions, DistrictBaselines
import uvicorn

app = FastAPI(
    title="CSI-DiCRA Compound Stress Index API",
    description="API for accessing actuarial climate risk tiers for agricultural insurance.",
    version="2.0.0"
)

@app.get("/api/v1/insurance/premium_risk")
def get_premium_risk(district: str, year: int, scenario: str = "SSP5_8.5", db: Session = Depends(get_db)):
    """
    Endpoint strictly for Insurance Companies to query risk tiers and adjust premiums.
    """
    # Normalise legacy RCP names to SSP names for backward compatibility
    scenario = scenario.replace("RCP_4.5", "SSP2_4.5").replace("RCP_8.5", "SSP5_8.5")
    prediction = db.query(RiskPredictions).filter(
        RiskPredictions.district == district,
        RiskPredictions.year == year,
        RiskPredictions.scenario == scenario
    ).first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="No projection found for this district/year/scenario.")
        
    return {
        "district": prediction.district,
        "year": prediction.year,
        "scenario": prediction.scenario,
        "csi_risk_tier": prediction.risk_tier,
        "probabilistic_score": round(prediction.csi_score, 3),
        "recommended_premium_adjustment": f"+{(prediction.risk_tier - 3) * 1.5}%" if prediction.risk_tier > 3 else "Base Rate"
    }

@app.get("/api/v1/dashboard/trajectory")
def get_risk_trajectory(district: str, scenario: str = "SSP5_8.5", db: Session = Depends(get_db)):
    """
    Endpoint for the Streamlit dashboard to plot 15-year trajectories.
    """
    # Normalise legacy RCP names
    scenario = scenario.replace("RCP_4.5", "SSP2_4.5").replace("RCP_8.5", "SSP5_8.5")
    predictions = db.query(RiskPredictions).filter(
        RiskPredictions.district == district,
        RiskPredictions.scenario == scenario
    ).order_by(RiskPredictions.year).all()
    
    return [
        {
            "year": p.year,
            "risk_tier": p.risk_tier,
            "score": p.csi_score
        } for p in predictions
    ]

@app.get("/api/v1/baselines")
def get_baselines(db: Session = Depends(get_db)):
    """
    Returns the dynamic 95th/5th baselines for all districts.
    """
    baselines = db.query(DistrictBaselines).all()
    return [{"district": b.district, "tmax_95th": b.tmax_95th, "sm_5th": b.sm_5th} for b in baselines]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)