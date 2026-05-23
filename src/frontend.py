import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import numpy as np
import json
import copy
import os

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000/api/v1"

def load_api_key() -> str:
    """Load the API key from environment variable or fall back to reading from .csi_api_key."""
    key = os.environ.get("CSI_API_KEY", "").strip()
    if key:
        return key

    key_file = ".csi_api_key"
    if os.path.exists(key_file):
        try:
            with open(key_file, "r") as f:
                val = f.read().strip()
                if val:
                    return val
        except Exception:
            pass
    return ""

CSI_API_KEY = load_api_key()
HEADERS = {"X-API-KEY": CSI_API_KEY} if CSI_API_KEY else {}

def wait_for_backend(timeout_seconds: int = 30) -> bool:
    """Wait for the FastAPI backend to become responsive."""
    import time
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            resp = requests.get(f"{API_URL}/crops", headers=HEADERS, timeout=2)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


st.set_page_config(
    page_title="CSI-DiCRA | Farmer Advisory",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Translation Dictionary ──────────────────────────────────────────────────────
TRANSLATIONS = {
    "en": {
        "title": "Compound Stress Index",
        "sub_header": "Data in Climate Resilient Agriculture",
        "district_lbl": "District",
        "crop_lbl": "Crop",
        "season_year_lbl": "Season Year",
        "scenario_lbl": "Climate Scenario",
        "search_btn": "Search 🔍",
        "data_sources": "Data Sources",
        "dicra_api": "DiCRA API (Tmax, NDVI, SM)",
        "chirps_drought": "CHIRPS (SPI-3 Drought)",
        "pmfby_cce": "PMFBY CCE Records",
        "cmip6": "CMIP6 ({scenario})",
        "fao_calendars": "FAO Crop Calendars",
        "mp_badge": "MADHYA PRADESH",
        "sub_badge": "District Advisory",
        "tab_map": "🗺️ Interactive Map",
        "tab_details": "📊 Selected District Details",
        "tab_actions": "🌾 Recommended Actions",
        "tab_trajectory": "📈 15-Year Risk Trajectory",
        "sec_map": "Interactive District Map",
        "sec_details": "Selected District Details",
        "sec_actions": "Risk Assessment & Advisories",
        "sec_trajectory": "15-Year Risk Trajectory",
        "layer_control": "Map Layer Select",
        "metric_risk_tier": "Risk Tier",
        "metric_csi_score": "CSI Score",
        "metric_csi_range": "0.0 = Min · 1.0 = Max",
        "metric_sm": "Soil Moisture",
        "metric_sm_sub": "Window Avg m³/m³",
        "metric_spi3": "SPI-3 Index",
        "metric_spi3_sub": "Drought Metric",
        "metric_ndvi": "NDVI",
        "metric_ndvi_sub": "Vegetation Index",
        "metric_tmax": "Tmax",
        "metric_tmax_sub": "Window Max Temp",
        "insurance_signal": "Insurance Premium Signal",
        "actuarial_premium": "Actuarial Risk Premium Adjustment",
        "premium_base": "Base Rate",
        "shap_title": "🔍 Actuarial Explainability: Yield Deficit/Surplus (Base: {base_value:,.0f} kg/ha)",
        "shap_xaxis": "Yield Deviation Contribution (kg/ha)",
        "traj_title": "📈 15-Year Risk Trajectory (2026 – 2041)",
        "traj_xaxis": "Year",
        "traj_yaxis": "Risk Tier",
        "traj_selected": "Selected ({year})",
        "traj_selected_hover": "Selected Year: {year}",
        "traj_tier_hover": "Tier: {tier}",
        "footer_text1": "CSI-DiCRA Actuarial Climate Risk Engine  ·  Data: DiCRA · CHIRPS · PMFBY · CMIP6",
        "footer_text2": "NABARD  ·  UNDP  ·  CSI-DiCRA-Project-V2",
        "no_trajectory": "No trajectory data available.",
        "api_error": "⚠️ Cannot connect to the FastAPI backend or load map data. Ensure it is running on port 8000.",
        "actions_title": "🌾 Recommended Actions — {crop} {year}",
        
        # Crop names
        "Soybean": "Soybean",
        "Rice": "Rice",
        "Maize": "Maize",
        "Wheat": "Wheat",
        "Gram": "Gram",
        "Mustard": "Mustard",
        
        # Seasons
        "Kharif": "Kharif",
        "Rabi": "Rabi",
        
        # SHAP Features
        "shap_feat_temp": "Mean Temperature",
        "shap_feat_sm": "Soil Moisture",
        "shap_feat_ndvi": "NDVI Vegetation Index",
        "shap_feat_precip": "Total Precipitation",
        "shap_feat_temp_anom": "Temperature Anomaly",
        "shap_feat_sm_anom": "Soil Moisture Anomaly",
        
        # Scenarios
        "ssp2_display": "SSP2-4.5 — Moderate Action",
        "ssp5_display": "SSP5-8.5 — Worst Case",
    },
    "hi": {
        "title": "संयुक्त तनाव सूचकांक",
        "sub_header": "जलवायु-अनुकूल कृषि में डेटा",
        "district_lbl": "जिला",
        "crop_lbl": "फसल",
        "season_year_lbl": "मौसम वर्ष",
        "scenario_lbl": "जलवायु परिदृश्य",
        "search_btn": "खोजें 🔍",
        "data_sources": "डेटा स्रोत",
        "dicra_api": "डीकरा एपीआई (Tmax, NDVI, SM)",
        "chirps_drought": "चिरप्स (SPI-3 सूखा)",
        "pmfby_cce": "पीएमएफबीवाई सीसीई रिकॉर्ड",
        "cmip6": "सीएमआईपी6 ({scenario})",
        "fao_calendars": "एफएओ फसल कैलेंडर",
        "mp_badge": "मध्य प्रदेश",
        "sub_badge": "जिला सलाह",
        "tab_map": "🗺️ इंटरैक्टिव मानचित्र",
        "tab_details": "📊 चयनित जिले का विवरण",
        "tab_actions": "🌾 अनुशंसित कार्रवाई",
        "tab_trajectory": "📈 15-वर्षीय जोखिम प्रक्षेपवक्र",
        "sec_map": "इंटरैक्टिव जिला मानचित्र",
        "sec_details": "चयनित जिले का विवरण",
        "sec_actions": "जोखिम मूल्यांकन और सलाह",
        "sec_trajectory": "15-वर्षीय जोखिम प्रक्षेपवक्र",
        "layer_control": "मानचित्र परत चयन",
        "metric_risk_tier": "जोखिम श्रेणी",
        "metric_csi_score": "सीएसआई स्कोर",
        "metric_csi_range": "0.0 = न्यूनतम · 1.0 = अधिकतम",
        "metric_sm": "मिट्टी की नमी",
        "metric_sm_sub": "अवधि औसत (m³/m³)",
        "metric_spi3": "एसपीआई-3 सूचकांक",
        "metric_spi3_sub": "सूखा संकेतक",
        "metric_ndvi": "एनडीवीआई (NDVI)",
        "metric_ndvi_sub": "वनस्पति सूचकांक",
        "metric_tmax": "अधिकतम तापमान (Tmax)",
        "metric_tmax_sub": "अवधि अधिकतम तापमान",
        "insurance_signal": "बीमा प्रीमियम संकेत",
        "actuarial_premium": "एक्चुरियल जोखिम प्रीमियम समायोजन",
        "premium_base": "मूल दर",
        "shap_title": "🔍 एक्चुरियल विश्लेषण: उपज घाटा/अधिशेष (आधार: {base_value:,.0f} किग्रा/हेक्टेयर)",
        "shap_xaxis": "उपज विचलन योगदान (किग्रा/हेक्टेयर)",
        "traj_title": "📈 15-वर्षीय जोखिम प्रक्षेपवक्र (2026 – 2041)",
        "traj_xaxis": "वर्ष",
        "traj_yaxis": "जोखिम श्रेणी",
        "traj_selected": "चयनित ({year})",
        "traj_selected_hover": "चयनित वर्ष: {year}",
        "traj_tier_hover": "श्रेणी: {tier}",
        "footer_text1": "सीएसआई-डीकरा एक्चुरियल जलवायु जोखिम इंजन  ·  डेटा: डीकरा · चिरप्स · पीएमएफबीवाई · सीएमआईपी6",
        "footer_text2": "नाबार्ड  ·  यूएनडीपी  ·  सीएसआई-डीकरा-प्रोजेक्ट-वी2",
        "no_trajectory": "कोई प्रक्षेपवक्र डेटा उपलब्ध नहीं है।",
        "api_error": "⚠️ FastAPI बैकएंड से कनेक्ट या मानचित्र डेटा लोड नहीं किया जा सका। सुनिश्चित करें कि यह पोर्ट 8000 पर चल रहा है।",
        "actions_title": "🌾 अनुशंसित कार्रवाई — {crop} {year}",
        
        # Crop names
        "Soybean": "सोयाबीन",
        "Rice": "धान (Rice)",
        "Maize": "मक्का (Maize)",
        "Wheat": "गेहूं (Wheat)",
        "Gram": "चना (Gram)",
        "Mustard": "सरसों (Mustard)",
        
        # Seasons
        "Kharif": "खरीफ",
        "Rabi": "रबी",
        
        # SHAP Features
        "shap_feat_temp": "औसत तापमान (Mean Temp)",
        "shap_feat_sm": "मिट्टी की नमी (Soil Moisture)",
        "shap_feat_ndvi": "एनडीवीआई सूचकांक (NDVI)",
        "shap_feat_precip": "कुल वर्षा (Precipitation)",
        "shap_feat_temp_anom": "तापमान विसंगति (Temp Anomaly)",
        "shap_feat_sm_anom": "मिट्टी नमी विसंगति (SM Anomaly)",
        
        # Scenarios
        "ssp2_display": "SSP2-4.5 — मध्यम प्रयास",
        "ssp5_display": "SSP5-8.5 — चरम स्थिति",
    }
}

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0c0f14; }

    /* Header with Glassmorphism */
    .dash-header {
        background: linear-gradient(135deg, rgba(26, 42, 26, 0.45) 0%, rgba(13, 31, 13, 0.45) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(45, 74, 45, 0.3);
        border-radius: 16px;
        padding: 20px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    .dash-title {
        font-size: 26px; font-weight: 700;
        color: #ffffff; margin: 0; letter-spacing: -0.5px;
    }
    .dash-subtitle {
        font-size: 13px; color: #82b082; margin-top: 4px; font-weight: 400;
    }
    .badge {
        display: inline-block;
        background: rgba(45, 74, 45, 0.6); color: #7aff7a;
        font-size: 11px; font-weight: 600; letter-spacing: 1px;
        padding: 3px 10px; border-radius: 20px;
        margin-left: 10px; vertical-align: middle;
        border: 1px solid rgba(74, 117, 74, 0.3);
    }

    /* Metric Cards with Glassmorphism */
    .metric-card {
        background: rgba(22, 27, 34, 0.55);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 18px;
        text-align: center;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .metric-card:hover {
        border-color: rgba(255, 255, 255, 0.18);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px 0 rgba(0, 0, 0, 0.35);
        background: rgba(22, 27, 34, 0.7);
    }
    .metric-label {
        font-size: 10px; font-weight: 600; letter-spacing: 1.5px;
        text-transform: uppercase; color: #8b949e; margin-bottom: 6px;
    }
    .metric-value {
        font-size: 28px; font-weight: 700; color: #ffffff; line-height: 1.1;
    }
    .metric-sub { font-size: 11px; color: #8b949e; margin-top: 6px; }

    /* Risk Banner with Glassmorphism */
    .risk-banner {
        border-radius: 12px;
        padding: 18px 24px;
        margin: 8px 0;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 5px solid;
        box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.2);
    }
    .risk-1 { background: rgba(13, 43, 13, 0.45); border-color: #3fb950; color: #a2fda2; }
    .risk-2 { background: rgba(30, 42, 13, 0.45); border-color: #a8cc3a; color: #e1fda2; }
    .risk-3 { background: rgba(42, 30, 13, 0.45); border-color: #d29922; color: #fde8a2; }
    .risk-4 { background: rgba(42, 17, 13, 0.45); border-color: #f85149; color: #fda2a2; }
    .risk-5 { background: rgba(42, 13, 13, 0.45); border-color: #ff0000; color: #ffbcbc; }

    .risk-title { font-size: 17px; font-weight: 700; margin-bottom: 6px; }
    .risk-body  { font-size: 13.5px; opacity: 0.9; line-height: 1.5; }

    /* Advisory Box with Glassmorphism */
    .advisory-box {
        background: rgba(22, 27, 34, 0.55);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 8px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
    }
    .advisory-heading {
        font-size: 12px; font-weight: 600; letter-spacing: 1px;
        color: #8b949e; text-transform: uppercase; margin-bottom: 12px;
    }
    .advisory-item {
        font-size: 13.5px; color: #c9d1d9;
        padding: 8px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        display: flex; align-items: flex-start; gap: 10px;
    }
    .advisory-item:last-child { border-bottom: none; }

    /* Section label */
    .section-label {
        font-size: 11px; font-weight: 600; letter-spacing: 1.5px;
        text-transform: uppercase; color: #8b949e;
        margin: 24px 0 12px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0b0d12; border-right: 1px solid rgba(255, 255, 255, 0.07); }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label { color: #8b949e !important; font-size: 12px !important; }

    /* Style Streamlit containers with borders as glassmorphic cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(22, 27, 34, 0.45) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25) !important;
        margin-bottom: 15px !important;
    }

    /* Hide default Streamlit chrome but keep sidebar toggle */
    #MainMenu, footer { visibility: hidden; }
    header { visibility: visible; background: transparent !important; }
    .block-container { padding-top: 24px; }

    /* Mobile and Tablet Responsiveness */
    @media (max-width: 768px) {
        div[data-testid="stPlotlyChart"] {
            height: 400px !important;
        }
        .dash-header {
            padding: 14px 18px !important;
            margin-bottom: 16px !important;
        }
        .dash-title {
            font-size: 19px !important;
        }
        .dash-subtitle {
            font-size: 11px !important;
        }
        .metric-card {
            padding: 12px 10px !important;
        }
        .metric-value {
            font-size: 20px !important;
        }
        .metric-label {
            font-size: 8.5px !important;
            letter-spacing: 1px !important;
        }
        .risk-banner {
            padding: 12px 16px !important;
        }
        .risk-title {
            font-size: 14px !important;
        }
        .risk-body {
            font-size: 11.5px !important;
        }
        .advisory-box {
            padding: 14px 16px !important;
        }
        .advisory-item {
            font-size: 11.5px !important;
            padding: 6px 0 !important;
        }
        button[data-baseweb="tab"] {
            font-size: 11.5px !important;
            padding: 8px 10px !important;
        }
    }

    /* Sidebar toggle buttons (collapsedControl and stBaseButton-headerNoPadding) styling for high visibility and permanent display */
    button[data-testid="collapsedControl"],
    button[data-testid="stBaseButton-headerNoPadding"] {
        background-color: rgba(26, 42, 26, 0.8) !important;
        border: 1px solid rgba(45, 74, 45, 0.4) !important;
        border-radius: 8px !important;
        color: #7aff7a !important;
        padding: 5px !important;
        z-index: 999999 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    button[data-testid="collapsedControl"] svg,
    button[data-testid="stBaseButton-headerNoPadding"] svg {
        color: #7aff7a !important;
        fill: #7aff7a !important;
        opacity: 1 !important;
    }
    /* Disable default hover-only hidden state of the expanded sidebar close button */
    [data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"] {
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Glassmorphic Tabs Styling */
    div[data-baseweb="tab-list"] {
        gap: 6px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        margin-bottom: 20px !important;
        background: transparent !important;
    }
    button[data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: #8b949e !important;
        background: rgba(22, 27, 34, 0.35) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-bottom: none !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 18px !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background: rgba(22, 27, 34, 0.65) !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #7aff7a !important;
        background: rgba(45, 74, 45, 0.2) !important;
        border-color: rgba(74, 117, 74, 0.4) !important;
        border-bottom: 2px solid #7aff7a !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 -4px 12px rgba(74, 117, 74, 0.15) !important;
    }
    /* Hide the default purple underline highlight from baseweb */
    div[data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Crop display metadata ───────────────────────────────────────────────────────
CROP_DISPLAY = {
    "Soybean": {"season": "Kharif", "window": "Jul 25 – Aug 31", "emoji": "🫘"},
    "Rice":    {"season": "Kharif", "window": "Sep 1 – Sep 30",  "emoji": "🌾"},
    "Maize":   {"season": "Kharif", "window": "Aug 10 – Sep 10", "emoji": "🌽"},
    "Wheat":   {"season": "Rabi",   "window": "Jan 20 – Feb 20", "emoji": "🌿"},
    "Gram":    {"season": "Rabi",   "window": "Dec 20 – Jan 31", "emoji": "🟤"},
    "Mustard": {"season": "Rabi",   "window": "Dec 15 – Jan 15", "emoji": "🌼"},
}

# ── Crop-aware Tier Config (6 crops × 5 tiers) ─────────────────────────────────
CROP_TIER_CONFIG = {
    "Soybean": {
        1: {
            "label": {"en": "Normal Season", "hi": "सामान्य मौसम"}, "emoji": "🟢", "color": "#3fb950", "css": "risk-1",
            "headline": {
                "en": "Conditions are within normal range for Soybean. Standard Kharif practices recommended.",
                "hi": "सोयाबीन के लिए परिस्थितियां सामान्य सीमा के भीतर हैं। सामान्य खरीफ प्रथाओं की सिफारिश की जाती है।"
            },
            "actions": {
                "en": [
                    "🌱 Proceed with standard sowing (June–July) as per local KVK norms.",
                    "💧 Maintain regular irrigation — no acute drought stress anticipated at flowering.",
                    "📋 No special variety changes required; use your district's recommended cultivar.",
                    "📞 Consult KVK agronomist for optimal fertiliser timing at pod-fill stage.",
                ],
                "hi": [
                    "🌱 स्थानीय केवीके (KVK) मानदंडों के अनुसार मानक बुवाई (जून-जुलाई) के साथ आगे बढ़ें।",
                    "💧 नियमित सिंचाई बनाए रखें — पुष्पन (flowering) के समय गंभीर सूखे के तनाव की आशंका नहीं है।",
                    "📋 किसी विशेष किस्म में बदलाव की आवश्यकता नहीं है; अपने जिले की अनुशंसित किस्म का उपयोग करें।",
                    "📞 फली भरने की अवस्था में इष्टतम उर्वरक समय के लिए केवीके कृषि वैज्ञानिक से परामर्श करें।"
                ]
            }
        },
        2: {
            "label": {"en": "Mild Stress", "hi": "हल्का तनाव"}, "emoji": "🟡", "color": "#a8cc3a", "css": "risk-2",
            "headline": {
                "en": "Mild compound heat + drought stress detected during the Soybean flowering window.",
                "hi": "सोयाबीन के पुष्पन (flowering) समय के दौरान हल्का संयुक्त गर्मी और सूखा तनाव देखा गया।"
            },
            "actions": {
                "en": [
                    "🌡️ Monitor temperature forecasts closely during July 25 – August 31.",
                    "💧 Ensure supplemental irrigation is available at pod initiation (R1-R3 stages).",
                    "🌾 Apply moisture-retentive mulching between rows to reduce soil water loss.",
                    "📋 Document any stress observations for KVK and PMFBY reporting.",
                ],
                "hi": [
                    "🌡️ जुलाई 25 - अगस्त 31 के दौरान तापमान पूर्वानुमानों की बारीकी से निगरानी करें।",
                    "💧 फली बनने (R1-R3 चरणों) के समय पूरक सिंचाई सुनिश्चित करें।",
                    "🌾 नमी बनाए रखने के लिए पंक्तियों के बीच मल्चिंग का उपयोग करें ताकि मिट्टी से जल हानि कम हो।",
                    "📋 केवीके और पीएमएफबीवाई रिपोर्टिंग के लिए फसल तनाव के अवलोकनों को दर्ज करें।"
                ]
            }
        },
        3: {
            "label": {"en": "Moderate Risk", "hi": "मध्यम जोखिम"}, "emoji": "🟠", "color": "#d29922", "css": "risk-3",
            "headline": {
                "en": "Moderate compound heat + drought projected during Soybean critical window. Proactive action needed.",
                "hi": "सोयाबीन की संवेदनशील अवधि के दौरान मध्यम संयुक्त गर्मी और सूखा अनुमानित है। सक्रिय कार्रवाई की आवश्यकता है।"
            },
            "actions": {
                "en": [
                    "⏰ Advance sowing by 7–10 days to shift flowering away from peak stress period.",
                    "🌿 Switch to semi-drought-tolerant variety: JS-9560 or NRC-86 are proven for MP.",
                    "💧 Install drip or sprinkler irrigation — flood irrigation is inefficient under heat stress.",
                    "📞 Enroll in your district's PMFBY scheme before sowing deadline.",
                    "🌡️ Follow IMD AgroMet bulletins weekly during Kharif for real-time adjustments.",
                ],
                "hi": [
                    "⏰ बुवाई को 7-10 दिन पहले करें ताकि पुष्पन चरम तनाव अवधि से बच सके।",
                    "🌿 अर्ध-सूखा प्रतिरोधी किस्मों (JS-9560 या NRC-86) पर स्विच करें जो मध्य प्रदेश के लिए प्रमाणित हैं।",
                    "💧 ड्रिप या स्प्रिंकलर सिंचाई स्थापित करें - गर्मी के तनाव में बाढ़ सिंचाई (flood irrigation) अप्रभावी है।",
                    "📞 बुवाई की समय सीमा से पहले अपने जिले की पीएमएफबीवाई योजना में नामांकन करें।",
                    "🌡️ साप्ताहिक रूप से आईएमडी (IMD) कृषि-मौसम बुलेटिन का पालन करें।"
                ]
            }
        },
        4: {
            "label": {"en": "Severe Risk", "hi": "गंभीर जोखिम"}, "emoji": "🔴", "color": "#f85149", "css": "risk-4",
            "headline": {
                "en": "Severe compound stress projected. Significant agronomic action required for Soybean.",
                "hi": "गंभीर संयुक्त तनाव अनुमानित है। सोयाबीन के लिए महत्वपूर्ण कृषि संबंधी कार्रवाई की आवश्यकता है।"
            },
            "actions": {
                "en": [
                    "🚨 Shift to a shorter-duration, drought-tolerant Soybean variety immediately.",
                    "⏰ Advance sowing by 14–21 days to move flowering window before peak heat.",
                    "🌾 Intercrop Soybean with drought-resistant Arhar (Pigeonpea) for risk diversification.",
                    "💧 Prioritize critical irrigation at R1 (flowering) and R3 (pod fill) growth stages.",
                    "📋 Ensure PMFBY enrollment is complete; document field conditions with date-stamped photos.",
                    "📞 Contact your KVK for district-specific emergency advisories.",
                ],
                "hi": [
                    "🚨 तुरंत कम अवधि की, सूखा-सहिष्णु सोयाबीन किस्म पर स्विच करें।",
                    "⏰ चरम गर्मी से बचने के लिए बुवाई को 14-21 दिन पहले करें।",
                    "🌾 जोखिम विविधीकरण के लिए सोयाबीन के साथ सूखा-प्रतिरोधी अरहर की अंतर-फसली खेती करें।",
                    "💧 पुष्पन (R1) और फली भरने (R3) के विकास चरणों में सिंचाई को प्राथमिकता दें।",
                    "📋 पीएमएफबीवाई नामांकन पूरा होना सुनिश्चित करें; दिनांक-सहित फोटो के साथ खेत की स्थितियों को दर्ज करें।",
                    "📞 जिला-विशिष्ट आपातकालीन सलाह के लिए अपने केवीके से संपर्क करें।"
                ]
            }
        },
        5: {
            "label": {"en": "Catastrophic", "hi": "चरम स्थिति"}, "emoji": "🔴", "color": "#ff6b6b", "css": "risk-5",
            "headline": {
                "en": "Catastrophic compound stress projected. Emergency measures required for Soybean.",
                "hi": "विनाशकारी संयुक्त तनाव अनुमानित है। सोयाबीन के लिए आपातकालीन उपायों की आवश्यकता है।"
            },
            "actions": {
                "en": [
                    "⛔ Kharif Soybean in this district carries extreme financial risk this year.",
                    "🔄 Strongly consider crop substitution: Arhar, Maize, or Bajra tolerate heat+drought far better.",
                    "💰 Ensure PMFBY insurance enrollment — trigger probability is very high this season.",
                    "🌱 If continuing Soybean, use only NRCS-approved extreme-stress varieties.",
                    "💧 Deficit irrigation: target only R1 and R3 stages; do not flood-irrigate.",
                    "📞 Escalate to District Agriculture Officer for emergency seed and input support.",
                    "🏦 Explore Kisan Credit Card (KCC) crop-loss advance against expected PMFBY claim.",
                ],
                "hi": [
                    "⛔ इस जिले में खरीफ सोयाबीन में अत्यधिक वित्तीय जोखिम है।",
                    "🔄 फसल प्रतिस्थापन पर गंभीरता से विचार करें: अरहर, मक्का या बाजरा गर्मी और सूखा बहुत बेहतर सहन करते हैं।",
                    "💰 पीएमएफबीवाई बीमा नामांकन सुनिश्चित करें - इस सीजन में दावा भुगतान की संभावना बहुत अधिक है।",
                    "🌱 यदि सोयाबीन जारी रखते हैं, तो केवल स्वीकृत अत्यधिक-तनाव प्रतिरोधी किस्मों का उपयोग करें।",
                    "💧 सीमित सिंचाई: केवल R1 और R3 चरणों को लक्षित करें; बाढ़ सिंचाई न करें।",
                    "📞 आपातकालीन बीज और इनपुट सहायता के लिए जिला कृषि अधिकारी से संपर्क करें।",
                    "🏦 संभावित पीएमएफबीवाई दावे के विरुद्ध किसान क्रेडिट कार्ड (KCC) फसल-हानि ऋण सहायता प्राप्त करें।"
                ]
            }
        },
    },

    "Rice": {
        1: {
            "label": {"en": "Normal Season", "hi": "सामान्य मौसम"}, "emoji": "🟢", "color": "#3fb950", "css": "risk-1",
            "headline": {
                "en": "Conditions are within normal range for Rice. Standard Kharif transplanting recommended.",
                "hi": "परिस्थितियां धान के लिए सामान्य सीमा के भीतर हैं। सामान्य खरीफ रोपाई की सिफारिश की जाती है।"
            },
            "actions": {
                "en": [
                    "🌱 Proceed with standard transplanting schedule (June–July nursery, July–Aug transplanting).",
                    "💧 Maintain adequate standing water (5–7 cm) during panicle initiation and heading.",
                    "📋 No special variety changes required for this season.",
                    "📞 Consult KVK for optimal top-dress urea timing at panicle initiation.",
                ],
                "hi": [
                    "🌱 मानक रोपाई कार्यक्रम (जून-जुलाई नर्सरी, जुलाई-अगस्त रोपाई) के साथ आगे बढ़ें।",
                    "💧 बाली बनने (panicle initiation) और पुष्प गुच्छ निकलने के दौरान पर्याप्त खड़ा पानी (5-7 सेमी) बनाए रखें।",
                    "📋 इस सीजन के लिए किसी विशेष किस्म में बदलाव की आवश्यकता नहीं है।",
                    "📞 बाली बनने के समय यूरिया देने के इष्टतम समय के लिए केवीके से परामर्श करें।"
                ]
            }
        },
        2: {
            "label": {"en": "Mild Stress", "hi": "हल्का तनाव"}, "emoji": "🟡", "color": "#a8cc3a", "css": "risk-2",
            "headline": {
                "en": "Mild heat + drought stress detected during Rice heading/grain-fill window (Sep).",
                "hi": "धान के बाली निकलने/दाना भरने की अवधि (सितंबर) के दौरान हल्का गर्मी और सूखा तनाव देखा गया।"
            },
            "actions": {
                "en": [
                    "🌡️ Monitor temperatures closely during September — heat at panicle emergence causes sterility.",
                    "💧 Maintain standing water during heading; do not allow fields to dry out in September.",
                    "🌾 Apply potassium (MOP) at panicle initiation to improve heat tolerance.",
                    "📋 Document stress observations and unusual flowering dates for KVK reporting.",
                ],
                "hi": [
                    "🌡️ सितंबर के दौरान तापमान की बारीकी से निगरानी करें - बाली निकलने के समय गर्मी से बांझपन (sterility) होता है।",
                    "💧 बाली निकलने के दौरान खड़ा पानी बनाए रखें; सितंबर में खेतों को सूखने न दें।",
                    "🌾 गर्मी सहनशीलता में सुधार के लिए बाली बनने की शुरुआत में पोटेशियम (MOP) का छिड़काव करें।",
                    "📋 केवीके रिपोर्टिंग के लिए तनाव के अवलोकनों और असामान्य पुष्पन तिथियों को दर्ज करें।"
                ]
            }
        },
        3: {
            "label": {"en": "Moderate Risk", "hi": "मध्यम जोखिम"}, "emoji": "🟠", "color": "#d29922", "css": "risk-3",
            "headline": {
                "en": "Moderate compound stress projected during Rice heading. Proactive water management needed.",
                "hi": "धान के बाली निकलने के दौरान मध्यम संयुक्त तनाव अनुमानित है। सक्रिय जल प्रबंधन की आवश्यकता है।"
            },
            "actions": {
                "en": [
                    "⏰ Advance nursery sowing by 10–15 days to shift heading before peak September heat.",
                    "🌿 Switch to shorter-duration, heat-tolerant variety: Sahbhagi Dhan or MTU-1010.",
                    "💧 Critical irrigation at panicle initiation and heading stages — do not allow stress at these points.",
                    "📞 Enroll in PMFBY before sowing deadline.",
                    "🌡️ Use IMD AgroMet bulletins to time final nitrogen application away from heat peaks.",
                ],
                "hi": [
                    "⏰ नर्सरी की बुवाई 10-15 दिन पहले करें ताकि बाली निकलना सितंबर की चरम गर्मी से पहले हो सके।",
                    "🌿 कम अवधि की, गर्मी-सहिष्णु किस्म पर स्विच करें: सहभागी धान या MTU-1010।",
                    "💧 बाली बनने की शुरुआत और निकलने के चरणों में महत्वपूर्ण सिंचाई - इन बिंदुओं पर तनाव न होने दें।",
                    "📞 बुवाई की समय सीमा से पहले पीएमएफबीवाई में नामांकन करें।",
                    "🌡️ गर्मी के चरम से बचकर अंतिम नाइट्रोजन अनुप्रयोग का समय निर्धारित करने के लिए आईएमडी कृषि-मौसम बुलेटिन का उपयोग करें।"
                ]
            }
        },
        4: {
            "label": {"en": "Severe Risk", "hi": "गंभीर जोखिम"}, "emoji": "🔴", "color": "#f85149", "css": "risk-4",
            "headline": {
                "en": "Severe compound stress projected during Rice grain-fill. Major action required.",
                "hi": "धान के दाना भरने के दौरान गंभीर संयुक्त तनाव अनुमानित है। प्रमुख कार्रवाई की आवश्यकता है।"
            },
            "actions": {
                "en": [
                    "🚨 Use only heat and drought-tolerant varieties: Sahbhagi Dhan, DRR Dhan 42, or Swarna Sub1.",
                    "⏰ Advance transplanting by 2–3 weeks to shift the critical window before peak heat.",
                    "💧 Alternate Wetting and Drying (AWD) technique — conserves water while maintaining yield.",
                    "🌾 Reduce seedling age at transplanting (25 days vs 35) for faster establishment.",
                    "📋 Ensure PMFBY enrollment; photograph field conditions at heading stage.",
                    "📞 Contact KVK for emergency advisory on late-season irrigation scheduling.",
                ],
                "hi": [
                    "🚨 केवल गर्मी और सूखा-सहिष्णु किस्मों का उपयोग करें: सहभागी धान, डीआरआर धान 42, या स्वर्ण सब1।",
                    "⏰ चरम गर्मी से पहले संवेदनशील अवधि को स्थानांतरित करने के लिए रोपाई को 2-3 सप्ताह पहले करें।",
                    "💧 वैकल्पिक गीला और सुखाने (AWD) तकनीक का उपयोग करें - यह उपज बनाए रखते हुए पानी बचाता है।",
                    "🌾 तेजी से स्थापना के लिए रोपाई के समय पौधे की उम्र कम (35 दिन के बजाय 25 दिन) रखें।",
                    "📋 पीएमएफबीवाई नामांकन सुनिश्चित करें; बाली आने के चरण में खेत की स्थितियों की फोटो लें।",
                    "📞 देर से मौसम में सिंचाई समय निर्धारण पर आपातकालीन सलाह के लिए केवीके से संपर्क करें।"
                ]
            }
        },
        5: {
            "label": {"en": "Catastrophic", "hi": "चरम स्थिति"}, "emoji": "🔴", "color": "#ff6b6b", "css": "risk-5",
            "headline": {
                "en": "Catastrophic stress projected during Rice heading. Emergency measures required.",
                "hi": "धान के बाली निकलने के दौरान विनाशकारी तनाव अनुमानित है। आपातकालीन उपायों की आवश्यकता है।"
            },
            "actions": {
                "en": [
                    "⛔ Kharif Rice in this district faces extreme risk of spikelet sterility and total yield loss.",
                    "🔄 Consider crop substitution: Maize or Soybean are more resilient under these conditions.",
                    "💰 Ensure PMFBY enrollment is complete — trigger probability is very high.",
                    "🌱 If continuing Rice, use only extreme stress-tolerant varieties (Sahbhagi Dhan, Nagina-22).",
                    "💧 Maintain standing water at all costs during September — even one day of drying at heading causes sterility.",
                    "📞 Escalate to District Agriculture Officer for emergency water allocation and input support.",
                    "🏦 Explore KCC crop-loss advance given the very high trigger probability this season.",
                ],
                "hi": [
                    "⛔ इस जिले में खरीफ धान को पुष्पगुच्छ बांझपन (spikelet sterility) और कुल उपज हानि का अत्यधिक जोखिम है।",
                    "🔄 फसल प्रतिस्थापन पर विचार करें: इन परिस्थितियों में मक्का या सोयाबीन अधिक लचीले हैं।",
                    "💰 पीएमएफबीवाई नामांकन पूरा होना सुनिश्चित करें - दावा भुगतान की संभावना बहुत अधिक है।",
                    "🌱 यदि धान जारी रखते हैं, तो केवल चरम तनाव-सहिष्णु किस्मों (सहभागी धान, नगीना-22) का उपयोग करें।",
                    "💧 सितंबर के दौरान हर कीमत पर पानी खड़ा रखें - बाली निकलने पर एक दिन भी सूखने से बांझपन हो सकता है।",
                    "📞 आपातकालीन जल आवंटन और इनपुट सहायता के लिए जिला कृषि अधिकारी से संपर्क करें।",
                    "🏦 इस सीजन में बहुत अधिक फसल हानि की संभावना को देखते हुए केसीसी अग्रिम सहायता का पता लगाएं।"
                ]
            }
        },
    },

    "Maize": {
        1: {
            "label": {"en": "Normal Season", "hi": "सामान्य मौसम"}, "emoji": "🟢", "color": "#3fb950", "css": "risk-1",
            "headline": {
                "en": "Normal conditions for Maize. Standard Kharif practices apply.",
                "hi": "मक्का के लिए सामान्य परिस्थितियां। मानक खरीफ कृषि पद्धतियां लागू होती हैं।"
            },
            "actions": {
                "en": [
                    "🌱 Proceed with standard sowing (June–July) as per local recommendations.",
                    "💧 Ensure one irrigation at tasseling (V14–VT stage) for maximum pollination.",
                    "📋 Standard fertiliser schedule — apply top-dress urea at knee-height (V6) stage.",
                    "📞 Consult KVK for hybrid selection suited to your district's soil type.",
                ],
                "hi": [
                    "🌱 स्थानीय अनुशंसाओं के अनुसार मानक बुवाई (जून-जुलाई) के साथ आगे बढ़ें।",
                    "💧 अधिकतम परागण के लिए नर मंजरी निकलने (tasseling - V14-VT चरण) के समय एक सिंचाई सुनिश्चित करें।",
                    "📋 मानक उर्वरक कार्यक्रम - घुटने की ऊंचाई (V6) के चरण में यूरिया का छिड़काव करें।",
                    "📞 अपने जिले की मिट्टी के प्रकार के अनुकूल हाइब्रिड चयन के लिए केवीके से परामर्श करें।"
                ]
            }
        },
        2: {
            "label": {"en": "Mild Stress", "hi": "हल्का तनाव"}, "emoji": "🟡", "color": "#a8cc3a", "css": "risk-2",
            "headline": {
                "en": "Mild heat + drought stress detected during Maize tasseling/silking window.",
                "hi": "मक्का के मंजरी (tasseling) / भुट्टा सिल्क (silking) निकलने की अवधि के दौरान हल्का गर्मी और सूखा तनाव देखा गया।"
            },
            "actions": {
                "en": [
                    "🌡️ Monitor temperatures during August 10 – September 10 — heat above 35°C at tasseling causes pollen sterility.",
                    "💧 Ensure one critical irrigation at tasseling (V14-VT) and one at silking (R1).",
                    "🌾 Apply foliar potassium spray (1% KCl) during tasseling to improve heat tolerance.",
                    "📋 Document stress symptoms (tassel blast, poor kernel set) for KVK records.",
                ],
                "hi": [
                    "🌡️ 10 अगस्त - 10 सितंबर के दौरान तापमान की निगरानी करें - मंजरी निकलने पर 35°C से अधिक गर्मी से पराग बांझपन होता है।",
                    "💧 मंजरी निकलने (V14-VT) पर एक और सिल्क निकलने (R1) पर एक सिंचाई सुनिश्चित करें।",
                    "🌾 गर्मी सहनशीलता में सुधार के लिए मंजरी निकलने के दौरान पोटेशियम (1% KCl) का पर्णीय छिड़काव करें।",
                    "📋 केवीके रिकॉर्ड के लिए तनाव के लक्षणों (मंजरी झुलसना, खराब दाना बैठना) को दर्ज करें।"
                ]
            }
        },
        3: {
            "label": {"en": "Moderate Risk", "hi": "मध्यम जोखिम"}, "emoji": "🟠", "color": "#d29922", "css": "risk-3",
            "headline": {
                "en": "Moderate compound stress projected during Maize critical tasseling window.",
                "hi": "मक्का की संवेदनशील मंजरी निकलने की अवधि के दौरान मध्यम संयुक्त तनाव अनुमानित है।"
            },
            "actions": {
                "en": [
                    "⏰ Advance sowing by 7–10 days to shift tasseling before peak August heat.",
                    "🌿 Use drought-tolerant hybrids: Vivek QPM-9, DKC-9108, or locally recommended stress-tolerant varieties.",
                    "💧 Two critical irrigations: at tasseling (V14-VT) and at silking (R1) — non-negotiable.",
                    "📞 Register with PMFBY before sowing deadline.",
                    "🌾 Avoid high nitrogen rates if drought stress is expected — increases lodging risk.",
                ],
                "hi": [
                    "⏰ बुवाई को 7-10 दिन पहले करें ताकि मंजरी निकलना अगस्त की चरम गर्मी से पहले हो सके।",
                    "🌿 सूखा-सहिष्णु हाइब्रिड का उपयोग करें: विवेक QPM-9, DKC-9108, या स्थानीय स्तर पर अनुशंसित किस्म।",
                    "💧 दो महत्वपूर्ण सिंचाइयां: मंजरी निकलने (V14-VT) और सिल्क निकलने (R1) पर - अनिवार्य है।",
                    "📞 बुवाई की समय सीमा से पहले पीएमएफबीवाई में पंजीकरण करें।",
                    "🌾 यदि सूखा तनाव की संभावना हो तो उच्च नाइट्रोजन दरों से बचें - इससे पौधों के गिरने का खतरा बढ़ जाता है।"
                ]
            }
        },
        4: {
            "label": {"en": "Severe Risk", "hi": "गंभीर जोखिम"}, "emoji": "🔴", "color": "#f85149", "css": "risk-4",
            "headline": {
                "en": "Severe stress projected at Maize tasseling. Significant action required.",
                "hi": "मक्का की मंजरी निकलने के चरण में गंभीर तनाव अनुमानित है। महत्वपूर्ण कार्रवाई की आवश्यकता है।"
            },
            "actions": {
                "en": [
                    "🚨 Shift to drought-tolerant hybrid varieties only — standard hybrids will fail at tasseling.",
                    "⏰ Advance sowing by 14–21 days to move tasseling away from the peak heat window.",
                    "💧 Drip or sprinkler irrigation at tasseling and grain fill is critical — flood irrigation is insufficient.",
                    "🌾 Reduce plant population by 10–15% to lower water demand per plant.",
                    "📋 PMFBY enrollment mandatory — ensure it is complete before sowing.",
                    "📞 Contact KVK for emergency seed substitution if standard seed lots are already purchased.",
                ],
                "hi": [
                    "🚨 केवल सूखा-सहिष्णु हाइब्रिड किस्मों पर स्विच करें - मानक हाइब्रिड मंजरी निकलने पर विफल हो जाएंगे।",
                    "⏰ चरम गर्मी से बचने के लिए बुवाई को 14-21 दिन पहले करें।",
                    "💧 मंजरी निकलने और दाना भरने पर ड्रिप या स्प्रिंकलर सिंचाई महत्वपूर्ण है - बाढ़ सिंचाई अपर्याप्त है।",
                    "🌾 प्रति पौधा पानी की मांग को कम करने के लिए पौधों की संख्या 10-15% कम करें।",
                    "📋 पीएमएफबीवाई नामांकन अनिवार्य है - बुवाई से पहले पूरा होना सुनिश्चित करें।",
                    "📞 यदि मानक बीज पहले ही खरीदे जा चुके हैं तो आपातकालीन बीज प्रतिस्थापन के लिए केवीके से संपर्क करें।"
                ]
            }
        },
        5: {
            "label": {"en": "Catastrophic", "hi": "चरम स्थिति"}, "emoji": "🔴", "color": "#ff6b6b", "css": "risk-5",
            "headline": {
                "en": "Catastrophic compound stress projected at Maize tasseling. Emergency action required.",
                "hi": "मक्का की मंजरी निकलने पर विनाशकारी संयुक्त तनाव अनुमानित है। आपातकालीन कार्रवाई की आवश्यकता है।"
            },
            "actions": {
                "en": [
                    "⛔ Kharif Maize in this district faces very high risk of total pollen failure and yield collapse.",
                    "🔄 Consider crop substitution: Arhar (Pigeonpea) or Bajra are far more heat+drought tolerant.",
                    "💰 PMFBY enrollment is critical — trigger probability is very high this season.",
                    "🌱 If continuing Maize, use only ICAR-approved extreme heat-tolerant hybrids.",
                    "💧 Deficit irrigation: prioritise tasseling and R1 stages exclusively.",
                    "📞 Escalate to District Agriculture Officer for emergency seed and input support.",
                    "🏦 Explore KCC crop-loss advance against expected PMFBY payout.",
                ],
                "hi": [
                    "⛔ इस जिले में खरीफ मक्का को पूर्ण पराग विफलता और उपज समाप्त होने का बहुत अधिक जोखिम है।",
                    "🔄 फसल प्रतिस्थापन पर विचार करें: अरहर या बाजरा गर्मी और सूखा सहन करने में बहुत अधिक सक्षम हैं।",
                    "💰 पीएमएफबीवाई नामांकन महत्वपूर्ण है - इस सीजन में दावा भुगतान की संभावना बहुत अधिक है।",
                    "🌱 यदि मक्का जारी रखते हैं, तो केवल आईसीएआर-अनुमोदित चरम गर्मी-सहिष्णु हाइब्रिड का उपयोग करें।",
                    "💧 सीमित सिंचाई: विशेष रूप से मंजरी निकलने और R1 चरणों को प्राथमिकता दें।",
                    "📞 आपातकालीन बीज और इनपुट सहायता के लिए जिला कृषि अधिकारी से संपर्क करें।",
                    "🏦 संभावित पीएमएफबीवाई भुगतान के विरुद्ध केसीसी फसल-हानि अग्रिम सहायता का पता लगाएं।"
                ]
            }
        },
    },

    "Wheat": {
        1: {
            "label": {"en": "Normal Season", "hi": "सामान्य मौसम"}, "emoji": "🌿", "color": "#3fb950", "css": "risk-1",
            "headline": {
                "en": "Normal Rabi conditions for Wheat. Standard sowing and irrigation schedule applies.",
                "hi": "गेहूं के लिए सामान्य रबी परिस्थितियां। मानक बुवाई और सिंचाई कार्यक्रम लागू होता है।"
            },
            "actions": {
                "en": [
                    "🌱 Proceed with standard sowing (November–December) as per district norms.",
                    "💧 Maintain the standard 5–6 irrigation schedule: CRI, tillering, jointing, heading, grain fill.",
                    "📋 Standard fertiliser schedule — basal DAP + top-dress urea at CRI and tillering.",
                    "📞 Consult KVK for variety selection suited to your district's soil type.",
                ],
                "hi": [
                    "🌱 जिला मानदंडों के अनुसार मानक बुवाई (नवंबर-दिसंबर) के साथ आगे बढ़ें।",
                    "💧 मानक 5-6 सिंचाई कार्यक्रम बनाए रखें: ताजमूल (CRI), कल्ले फूटना, गांठ बनना, बाली आना, दाना भरना।",
                    "📋 मानक उर्वरक कार्यक्रम - आधार रूप में DAP + ताजमूल (CRI) और कल्ले फूटने पर यूरिया का छिड़काव।",
                    "📞 अपने जिले की मिट्टी के अनुकूल किस्म चयन के लिए केवीके से परामर्श करें।"
                ]
            }
        },
        2: {
            "label": {"en": "Mild Stress", "hi": "हल्का तनाव"}, "emoji": "🟡", "color": "#a8cc3a", "css": "risk-2",
            "headline": {
                "en": "Mild unseasonal warming detected during the Wheat heading window (Jan–Feb).",
                "hi": "गेहूं की बाली आने की अवधि (जनवरी-फरवरी) के दौरान हल्की बेमौसम गर्मी देखी गई।"
            },
            "actions": {
                "en": [
                    "🌡️ Monitor temperature forecasts in January–February — even 3–4°C above normal at heading causes sterility.",
                    "💧 Ensure irrigation at heading (Zadoks 49–55) — do not allow moisture stress at this stage.",
                    "🌾 Apply micronutrient spray (Zinc 0.5%) at booting stage to improve stress resilience.",
                    "📋 Document any premature heading or bleaching symptoms for KVK records.",
                ],
                "hi": [
                    "🌡️ जनवरी-फरवरी में तापमान पूर्वानुमानों की निगरानी करें - बाली आने पर सामान्य से 3-4°C अधिक तापमान भी बांझपन का कारण बनता है।",
                    "💧 बाली आने पर सिंचाई सुनिश्चित करें - इस चरण में नमी का तनाव न होने दें।",
                    "🌾 तनाव सहनशीलता में सुधार के लिए गभोट (booting) चरण में सूक्ष्म पोषक तत्व छिड़काव (जिंक 0.5%) करें।",
                    "📋 केवीके रिकॉर्ड के लिए समय से पहले बाली आने या सूखने के लक्षणों को दर्ज करें।"
                ]
            }
        },
        3: {
            "label": {"en": "Moderate Risk", "hi": "मध्यम जोखिम"}, "emoji": "🟠", "color": "#d29922", "css": "risk-3",
            "headline": {
                "en": "Moderate terminal heat stress projected during Wheat heading. Proactive action needed.",
                "hi": "गेहूं की बाली आने के दौरान मध्यम अंतिम गर्मी (terminal heat) तनाव अनुमानित है। सक्रिय कार्रवाई आवश्यक है।"
            },
            "actions": {
                "en": [
                    "⏰ Advance sowing by 7–10 days (early November) to complete grain fill before February heat.",
                    "🌿 Use heat-tolerant varieties: HD-2781, HI-8498, or GW-496 for this district.",
                    "💧 Critical irrigation at booting and heading stages — missing either one under heat stress causes 30-40% yield loss.",
                    "📞 Enroll in PMFBY before the Rabi sowing deadline.",
                    "🌡️ Follow IMD's weekly AgroMet bulletin for precise irrigation timing.",
                ],
                "hi": [
                    "⏰ बुवाई को 7-10 दिन पहले (नवंबर की शुरुआत में) करें ताकि फरवरी की गर्मी से पहले दाना भरना पूरा हो सके।",
                    "🌿 गर्मी-सहिष्णु किस्मों का उपयोग करें: इस जिले के लिए HD-2781, HI-8498, या GW-496।",
                    "💧 गभोट और बाली आने के चरणों में सिंचाई - गर्मी के तनाव में इनमें से किसी भी चरण को छोड़ने से 30-40% उपज हानि होती है।",
                    "📞 रबी बुवाई की समय सीमा से पहले पीएमएफबीवाई में नामांकन करें।",
                    "🌡️ सटीक सिंचाई समय निर्धारण के लिए आईएमडी के साप्ताहिक कृषि-मौसम बुलेटिन का पालन करें।"
                ]
            }
        },
        4: {
            "label": {"en": "Severe Risk", "hi": "गंभीर जोखिम"}, "emoji": "🔴", "color": "#f85149", "css": "risk-4",
            "headline": {
                "en": "Severe terminal heat stress projected for Wheat. Major agronomic action required.",
                "hi": "गेहूं के लिए गंभीर अंतिम गर्मी (terminal heat) तनाव अनुमानित है। प्रमुख कृषि संबंधी कार्रवाई की आवश्यकता है।"
            },
            "actions": {
                "en": [
                    "🚨 Sow only heat-tolerant varieties: HD-2781, HI-8498, K-0307 — standard HD-2967 will underperform severely.",
                    "⏰ Sow in first fortnight of November — every 10 days of delayed sowing increases terminal heat exposure.",
                    "💧 Irrigation at booting, heading AND grain-fill stages is mandatory under heat stress conditions.",
                    "🌾 Apply 'Stay Green' foliar micronutrients (Zinc + Boron) at heading to extend grain-fill duration.",
                    "📋 PMFBY enrollment is mandatory — complete before deadline.",
                    "📞 Contact KVK for emergency variety and input advisory for late-sown fields.",
                ],
                "hi": [
                    "🚨 केवल गर्मी-सहिष्णु किस्में बोएं: HD-2781, HI-8498, K-0307 - मानक HD-2967 बहुत खराब प्रदर्शन करेगी।",
                    "⏰ नवंबर के पहले पखवाड़े में बोएं - बुवाई में 10 दिन की देरी भी गर्मी का जोखिम बहुत बढ़ा देती है।",
                    "💧 गर्मी के तनाव में गभोट, बाली आने और दाना भरने के चरणों में सिंचाई करना अनिवार्य है।",
                    "🌾 दाना भरने की अवधि बढ़ाने के लिए बाली आने पर 'स्टे ग्रीन' सूक्ष्म पोषक तत्वों (जिंक + बोरॉन) का छिड़काव करें।",
                    "📋 पीएमएफबीवाई नामांकन अनिवार्य है - समय सीमा से पहले पूरा करें।",
                    "📞 देर से बोए गए खेतों के लिए आपातकालीन किस्म और इनपुट सलाह के लिए केवीके से संपर्क करें।"
                ]
            }
        },
        5: {
            "label": {"en": "Catastrophic", "hi": "चरम स्थिति"}, "emoji": "🔴", "color": "#ff6b6b", "css": "risk-5",
            "headline": {
                "en": "Catastrophic terminal heat projected for Wheat. Emergency measures required.",
                "hi": "गेहूं के लिए विनाशकारी अंतिम गर्मी तनाव अनुमानित है। आपातकालीन उपायों की आवश्यकता है।"
            },
            "actions": {
                "en": [
                    "⛔ Rabi Wheat in this district faces extreme risk of grain shrivelling and near-total yield failure.",
                    "🔄 Strongly consider substitution with Gram or Mustard — both tolerate Rabi heat far better.",
                    "💰 PMFBY enrollment is critical — trigger probability is very high this season.",
                    "🌱 If continuing Wheat, use only ICAR-approved extreme heat-tolerant varieties for MP.",
                    "💧 Maximum irrigation support at booting, heading, and grain-fill — 3 critical irrigations.",
                    "📞 Escalate to District Agriculture Officer for emergency seed access and input subsidies.",
                    "🏦 Explore KCC crop-loss advance against expected PMFBY claim.",
                ],
                "hi": [
                    "⛔ इस जिले में रबी गेहूं को दाने सिकुड़ने और लगभग पूर्ण उपज विफलता का अत्यधिक जोखिम है।",
                    "🔄 चना या सरसों के साथ प्रतिस्थापन पर गंभीरता से विचार करें - दोनों रबी गर्मी को बहुत बेहतर सहन करते हैं।",
                    "💰 पीएमएफबीवाई नामांकन महत्वपूर्ण है - इस सीजन में दावा भुगतान की संभावना बहुत अधिक है।",
                    "🌱 यदि गेहूं जारी रखते हैं, तो केवल मध्य प्रदेश के लिए स्वीकृत अत्यधिक गर्मी-सहिष्णु किस्मों का उपयोग करें।",
                    "💧 गभोट, बाली आने और दाना भरने के दौरान अधिकतम सिंचाई सहायता - 3 महत्वपूर्ण सिंचाइयां।",
                    "📞 आपातकालीन बीज और इनपुट सब्सिडी के लिए जिला कृषि अधिकारी से संपर्क करें।",
                    "🏦 संभावित पीएमएफबीवाई दावे के विरुद्ध केसीसी फसल-हानि अग्रिम सहायता का पता लगाएं।"
                ]
            }
        },
    },

    "Gram": {
        1: {
            "label": {"en": "Normal Season", "hi": "सामान्य मौसम"}, "emoji": "🟤", "color": "#3fb950", "css": "risk-1",
            "headline": {
                "en": "Normal Rabi conditions for Gram (Chickpea). Standard practices apply.",
                "hi": "चना के लिए सामान्य रबी परिस्थितियां। मानक पद्धतियां लागू होती हैं।"
            },
            "actions": {
                "en": [
                    "🌱 Proceed with standard sowing (October–November) under rainfed or light irrigation.",
                    "💧 Gram is largely rainfed — avoid excess irrigation at flowering (causes flower drop).",
                    "📋 Standard fertiliser: basal P and K only — Gram fixes its own nitrogen.",
                    "📞 Consult KVK for wilt-resistant variety suitable to your district's soil.",
                ],
                "hi": [
                    "🌱 वर्षा आधारित या हल्की सिंचाई के तहत मानक बुवाई (अक्टूबर-नवंबर) के साथ आगे बढ़ें।",
                    "💧 चना काफी हद तक वर्षा आधारित है - पुष्पन के समय अतिरिक्त सिंचाई से बचें (फूल गिरने का कारण बनता है)।",
                    "📋 मानक उर्वरक: केवल आधार फास्फोरस (P) और पोटेशियम (K) - चना अपना नाइट्रोजन खुद स्थिर करता है।",
                    "📞 अपने जिले की मिट्टी के अनुकूल उकठा-प्रतिरोधी (wilt-resistant) किस्म के लिए केवीके से परामर्श करें।"
                ]
            }
        },
        2: {
            "label": {"en": "Mild Stress", "hi": "हल्का तनाव"}, "emoji": "🟡", "color": "#a8cc3a", "css": "risk-2",
            "headline": {
                "en": "Mild heat + drought stress detected during Gram flowering window (Dec–Jan).",
                "hi": "चना के पुष्पन समय (दिसंबर-जनवरी) के दौरान हल्का गर्मी और सूखा तनाव देखा गया।"
            },
            "actions": {
                "en": [
                    "🌡️ Monitor temperatures in December–January — temperatures above 30°C at flowering cause pod abortion.",
                    "💧 One light pre-sowing irrigation and one at pod-fill is sufficient — avoid over-irrigation.",
                    "🌾 Apply sulfur (20 kg/ha) at sowing — improves drought resilience in Gram.",
                    "📋 Document any premature flower drop or wilting symptoms for KVK records.",
                ],
                "hi": [
                    "🌡️ दिसंबर-जनवरी में तापमान की निगरानी करें - पुष्पन पर 30°C से अधिक तापमान से फली का विकास रुकता है।",
                    "💧 बुवाई से पहले एक हल्की सिंचाई और फली भरने पर एक सिंचाई पर्याप्त है - अत्यधिक सिंचाई से बचें।",
                    "🌾 बुवाई के समय सल्फर (20 किग्रा/हेक्टेयर) डालें - यह चने में सूखा सहनशीलता में सुधार करता है।",
                    "📋 समय से पहले फूल गिरने या मुरझाने के किसी भी लक्षण को केवीके रिकॉर्ड के लिए दर्ज करें।"
                ]
            }
        },
        3: {
            "label": {"en": "Moderate Risk", "hi": "मध्यम जोखिम"}, "emoji": "🟠", "color": "#d29922", "css": "risk-3",
            "headline": {
                "en": "Moderate compound stress projected during Gram critical flowering window. Proactive action needed.",
                "hi": "चना की संवेदनशील पुष्पन अवधि के दौरान मध्यम संयुक्त तनाव अनुमानित है। सक्रिय कार्रवाई की आवश्यकता है।"
            },
            "actions": {
                "en": [
                    "⏰ Advance sowing by 7–10 days (early October) to complete flowering before January heat.",
                    "🌿 Use heat-tolerant variety: JAKI-9218 or JG-14 are recommended for MP conditions.",
                    "💧 One critical irrigation at pod initiation (avoid at full flowering — causes flower drop).",
                    "📞 Enroll in PMFBY before Rabi sowing deadline.",
                    "🌾 Reduce plant population by 10% to lower water demand per plant under stress.",
                ],
                "hi": [
                    "⏰ बुवाई को 7-10 दिन पहले (अक्टूबर की शुरुआत में) करें ताकि पुष्पन जनवरी की गर्मी से पहले पूरा हो सके।",
                    "🌿 गर्मी-सहिष्णु किस्मों का उपयोग करें: मध्य प्रदेश के लिए JAKI-9218 या JG-14 अनुशंसित हैं।",
                    "💧 फली बनने की शुरुआत में एक महत्वपूर्ण सिंचाई (पूर्ण पुष्पन के समय बचें - फूल गिरते हैं)।",
                    "📞 रबी बुवाई की समय सीमा से पहले पीएमएफबीवाई में नामांकन करें।",
                    "🌾 तनाव में प्रति पौधा पानी की मांग को कम करने के लिए पौधों की संख्या 10% कम करें।"
                ]
            }
        },
        4: {
            "label": {"en": "Severe Risk", "hi": "गंभीर जोखिम"}, "emoji": "🔴", "color": "#f85149", "css": "risk-4",
            "headline": {
                "en": "Severe stress projected during Gram flowering. Significant action required.",
                "hi": "चना पुष्पन के दौरान गंभीर तनाव अनुमानित है। महत्वपूर्ण कार्रवाई की आवश्यकता है।"
            },
            "actions": {
                "en": [
                    "🚨 Shift to heat-tolerant short-duration varieties: JAKI-9218 or Radhey (JG-74).",
                    "⏰ Advance sowing to first fortnight of October to avoid January heat at flowering.",
                    "💧 One targeted irrigation at pod initiation ONLY — avoid irrigating during full bloom.",
                    "🌾 Intercrop with Mustard (2:1 ratio) — Mustard provides partial shade and diversifies income.",
                    "📋 PMFBY enrollment is mandatory — complete before deadline.",
                    "📞 Contact KVK for emergency variety and input advisory.",
                ],
                "hi": [
                    "🚨 गर्मी-सहिष्णु लघु अवधि किस्मों पर स्विच करें: JAKI-9218 या राधे (JG-74)।",
                    "⏰ पुष्पन के समय जनवरी की गर्मी से बचने के लिए बुवाई अक्टूबर के पहले पखवाड़े में करें।",
                    "💧 केवल फली बनने की शुरुआत में एक लक्षित सिंचाई - पूर्ण पुष्पन के दौरान सिंचाई से बचें।",
                    "🌾 सरसों के साथ अंतर-फसली (2:1 अनुपात) खेती करें - सरसों आंशिक छाया देती है और आय को संतुलित करती है।",
                    "📋 पीएमएफबीवाई नामांकन अनिवार्य है - समय सीमा से पहले पूरा करें।",
                    "📞 आपातकालीन किस्म और इनपुट सलाह के लिए केवीके से संपर्क करें।"
                ]
            }
        },
        5: {
            "label": {"en": "Catastrophic", "hi": "चरम स्थिति"}, "emoji": "🔴", "color": "#ff6b6b", "css": "risk-5",
            "headline": {
                "en": "Catastrophic compound stress projected for Gram. Emergency action required.",
                "hi": "चना के लिए विनाशकारी संयुक्त तनाव अनुमानित है। आपातकालीन कार्रवाई की आवश्यकता है।"
            },
            "actions": {
                "en": [
                    "⛔ Gram in this district faces extreme risk of total pod abortion and yield failure.",
                    "🔄 Strongly consider substituting with Mustard or Lentil — both tolerate heat better than Gram.",
                    "💰 PMFBY enrollment is critical — trigger probability is very high this season.",
                    "🌱 If continuing Gram, use only ICAR extreme heat-tolerant varieties.",
                    "💧 One targeted irrigation at pod initiation — any additional irrigation will cause more flower drop.",
                    "📞 Escalate to District Agriculture Officer for emergency input support.",
                    "🏦 Explore KCC crop-loss advance against expected PMFBY claim.",
                ],
                "hi": [
                    "⛔ इस जिले में चना को कुल फली नष्ट होने और उपज विफलता का अत्यधिक जोखिम है।",
                    "🔄 सरसों या मसूर के साथ प्रतिस्थापित करने पर विचार करें - दोनों चने की तुलना में गर्मी बेहतर सहन करते हैं।",
                    "💰 पीएमएफबीवाई नामांकन महत्वपूर्ण है - इस सीजन में दावा भुगतान की संभावना बहुत अधिक है।",
                    "🌱 यदि चना जारी रखते हैं, तो केवल आईसीएआर की चरम गर्मी-सहिष्णु किस्मों का उपयोग करें।",
                    "💧 केवल फली बनने की शुरुआत में एक लक्षित सिंचाई - कोई भी अतिरिक्त सिंचाई फूल गिरा देगी।",
                    "📞 आपातकालीन इनपुट सहायता के लिए जिला कृषि अधिकारी से संपर्क करें।",
                    "🏦 संभावित पीएमएफबीवाई दावे के विरुद्ध केसीसी फसल-हानि अग्रिम ऋण का पता लगाएं।"
                ]
            }
        },
    },

    "Mustard": {
        1: {
            "label": {"en": "Normal Season", "hi": "सामान्य मौसम"}, "emoji": "🌼", "color": "#3fb950", "css": "risk-1",
            "headline": {
                "en": "Normal Rabi conditions for Mustard. Standard practices recommended.",
                "hi": "सरसों के लिए सामान्य रबी परिस्थितियां। मानक पद्धतियों की सिफारिश की जाती है।"
            },
            "actions": {
                "en": [
                    "🌱 Proceed with standard sowing (October–November) for your district.",
                    "💧 One irrigation at branching and one at siliqua development is standard.",
                    "📋 Standard fertiliser: basal DAP + top-dress nitrogen at flowering.",
                    "📞 Consult KVK for variety selection (PM-28, RH-749, or Varuna) for your district.",
                ],
                "hi": [
                    "🌱 अपने जिले के लिए मानक बुवाई (अक्टूबर-नवंबर) के साथ आगे बढ़ें।",
                    "💧 शाखाएं बनते समय एक सिंचाई और फली (siliqua) विकास पर एक सिंचाई मानक है।",
                    "📋 मानक उर्वरक: आधार रूप में DAP + पुष्पन पर नाइट्रोजन का छिड़काव।",
                    "📞 अपने जिले के अनुकूल किस्मों (PM-28, RH-749, या वरुणा) के लिए केवीके से परामर्श करें।"
                ]
            }
        },
        2: {
            "label": {"en": "Mild Stress", "hi": "हल्का तनाव"}, "emoji": "🟡", "color": "#a8cc3a", "css": "risk-2",
            "headline": {
                "en": "Mild heat + drought stress detected during the Mustard flowering window (Dec–Jan).",
                "hi": "सरसों के पुष्पन समय (दिसंबर-जनवरी) के दौरान हल्का गर्मी और सूखा तनाव देखा गया।"
            },
            "actions": {
                "en": [
                    "🌡️ Monitor temperatures in December–January — heat above 28°C at full bloom causes pod shatter.",
                    "💧 Ensure irrigation at flowering stage — do not allow moisture stress during bloom.",
                    "🌾 Watch for aphid infestation — warm Rabi conditions accelerate aphid population cycles in Mustard.",
                    "📋 Document any premature siliqua drop or aphid damage for KVK reporting.",
                ],
                "hi": [
                    "🌡️ दिसंबर-जनवरी में तापमान की निगरानी करें - पूर्ण पुष्पन पर 28°C से अधिक गर्मी से फलियां टूट (shatter) सकती हैं।",
                    "💧 पुष्पन के चरण में सिंचाई सुनिश्चित करें - इस समय नमी का तनाव न होने दें।",
                    "🌾 माहू (aphid) के प्रकोप पर नजर रखें - गर्म रबी परिस्थितियां सरसों में माहू चक्र को तेज करती हैं।",
                    "📋 केवीके रिपोर्टिंग के लिए समय से पहले फली गिरने या माहू के नुकसान को दर्ज करें।"
                ]
            }
        },
        3: {
            "label": {"en": "Moderate Risk", "hi": "मध्यम जोखिम"}, "emoji": "🟠", "color": "#d29922", "css": "risk-3",
            "headline": {
                "en": "Moderate compound stress projected during Mustard critical flowering window.",
                "hi": "सरसों की संवेदनशील पुष्पन अवधि के दौरान मध्यम संयुक्त तनाव अनुमानित है।"
            },
            "actions": {
                "en": [
                    "⏰ Advance sowing by 7–10 days (early October) to complete flowering before January heat.",
                    "🌿 Use heat-tolerant variety: PM-28 or Pusa Mustard-27 are suitable for MP conditions.",
                    "💧 Two irrigations: one at flowering and one at siliqua development — both are critical.",
                    "📞 Enroll in PMFBY before Rabi sowing deadline.",
                    "🌾 Apply thiourea spray (500 ppm) at budding stage — proven to reduce heat stress in Mustard.",
                ],
                "hi": [
                    "⏰ बुवाई को 7-10 दिन पहले (अक्टूबर की शुरुआत में) करें ताकि पुष्पन जनवरी की गर्मी से पहले हो सके।",
                    "🌿 गर्मी-सहिष्णु किस्मों का उपयोग करें: मध्य प्रदेश के लिए PM-28 या पूसा सरसों-27 उपयुक्त हैं।",
                    "💧 दो सिंचाइयां: एक पुष्पन पर और एक फली विकास पर - दोनों महत्वपूर्ण हैं।",
                    "📞 रबी बुवाई की समय सीमा से पहले पीएमएफबीवाई में नामांकन करें।",
                    "🌾 कली आने की अवस्था में थायोरिया छिड़काव (500 ppm) करें - जो सरसों में गर्मी के तनाव को कम करता है।"
                ]
            }
        },
        4: {
            "label": {"en": "Severe Risk", "hi": "गंभीर जोखिम"}, "emoji": "🔴", "color": "#f85149", "css": "risk-4",
            "headline": {
                "en": "Severe heat stress projected at Mustard flowering. Major action required.",
                "hi": "सरसों के पुष्पन के समय गंभीर गर्मी तनाव अनुमानित है। प्रमुख कार्रवाई आवश्यक है।"
            },
            "actions": {
                "en": [
                    "🚨 Shift to early-maturing heat-tolerant varieties only: PM-28 or Pusa Mustard-27.",
                    "⏰ Advance sowing to first fortnight of October — every 10-day delay increases heat exposure at bloom.",
                    "💧 Two critical irrigations: at branching and at pod development — do not skip.",
                    "🌾 Apply thiourea foliar spray (500 ppm) at full flowering to reduce pollen sterility.",
                    "📋 PMFBY enrollment is mandatory — complete before deadline.",
                    "📞 Contact KVK for emergency variety access if standard seed lots are already purchased.",
                ],
                "hi": [
                    "🚨 केवल शीघ्र पकने वाली गर्मी-सहिष्णु किस्मों पर स्विच करें: PM-28 या पूसा सरसों-27।",
                    "⏰ अक्टूबर के पहले पखवाड़े में बुवाई करें - 10 दिन की देरी भी पुष्पन पर गर्मी का प्रभाव बढ़ाती है।",
                    "💧 दो महत्वपूर्ण सिंचाइयां: शाखाएं बनते समय और फली विकास के चरणों में - न छोड़ें।",
                    "🌾 पराग बांझपन कम करने के लिए पूर्ण पुष्पन पर थायोरिया छिड़काव (500 ppm) करें।",
                    "📋 पीएमएफबीवाई नामांकन अनिवार्य है - समय सीमा से पहले पूरा करें।",
                    "📞 यदि सामान्य बीज खरीदे जा चुके हैं तो आपातकालीन किस्म विस्थापन के लिए केवीके से संपर्क करें।"
                ]
            }
        },
        5: {
            "label": {"en": "Catastrophic", "hi": "चरम स्थिति"}, "emoji": "🔴", "color": "#ff6b6b", "css": "risk-5",
            "headline": {
                "en": "Catastrophic heat stress projected for Mustard. Emergency measures required.",
                "hi": "सरसों के लिए विनाशकारी गर्मी तनाव अनुमानित है। आपातकालीन उपायों की आवश्यकता है।"
            },
            "actions": {
                "en": [
                    "⛔ Rabi Mustard in this district faces extreme risk of mass pod shatter and near-total yield loss.",
                    "🔄 Consider substituting with Gram or Lentil — both tolerate heat better than Mustard.",
                    "💰 PMFBY enrollment is critical — trigger probability is very high this season.",
                    "🌱 If continuing Mustard, use only ICAR-approved extreme heat-tolerant varieties.",
                    "💧 Two critical irrigations at branching and pod-fill — do not reduce under any cost pressure.",
                    "📞 Escalate to District Agriculture Officer for emergency seed and input support.",
                    "🏦 Explore KCC crop-loss advance against expected PMFBY claim this season.",
                ],
                "hi": [
                    "⛔ इस जिले में रबी सरसों को भारी फली टूटने और लगभग पूर्ण उपज हानि का अत्यधिक जोखिम है।",
                    "🔄 चना या मसूर के साथ प्रतिस्थापित करने पर विचार करें - दोनों रबी गर्मी को बेहतर सहन करते हैं।",
                    "💰 पीएमएफबीवाई नामांकन महत्वपूर्ण है - इस सीजन में दावा भुगतान की संभावना बहुत अधिक है।",
                    "🌱 यदि सरसों जारी रखते हैं, तो केवल स्वीकृत अत्यधिक गर्मी-सहिष्णु किस्मों का उपयोग करें।",
                    "💧 शाखाएं बनते समय और फली भरने पर दो महत्वपूर्ण सिंचाइयां - किसी भी इनपुट दबाव में न घटाएं।",
                    "📞 आपातकालीन बीज और इनपुट सहायता के लिए जिला कृषि अधिकारी से संपर्क करें।",
                    "🏦 इस सीजन में संभावित पीएमएफबीवाई दावे के विरुद्ध केसीसी अग्रिम फसल-हानि का पता लगाएं।"
                ]
            }
        },
    },
}

# Initialize session state for selected district, applied filters, and language
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
if "selected_district" not in st.session_state:
    st.session_state["selected_district"] = "Bhopal"
if "applied_crop" not in st.session_state:
    st.session_state["applied_crop"] = "Soybean"
if "applied_year" not in st.session_state:
    st.session_state["applied_year"] = 2026
if "applied_scenario" not in st.session_state:
    st.session_state["applied_scenario"] = "SSP2_4.5"

# Language toggle at the top of the sidebar
selected_lang = st.sidebar.selectbox(
    "Language / भाषा",
    ["English", "हिन्दी (Hindi)"],
    index=0 if st.session_state["lang"] == "en" else 1,
    key="lang_selector_widget"
)
st.session_state["lang"] = "hi" if "हिन्दी" in selected_lang else "en"
lang = st.session_state["lang"]
t = TRANSLATIONS[lang]

# Sync temporary filters from applied/session state if not set
if "temp_district_select" not in st.session_state:
    st.session_state["temp_district_select"] = st.session_state["selected_district"]
if "temp_crop_select" not in st.session_state:
    st.session_state["temp_crop_select"] = st.session_state["applied_crop"]
if "temp_year_slider" not in st.session_state:
    st.session_state["temp_year_slider"] = st.session_state["applied_year"]

# Map scenario text dynamically based on current language selectbox value
ssp2_label = t["ssp2_display"]
ssp5_label = t["ssp5_display"]

if "temp_scenario_radio_widget" not in st.session_state:
    st.session_state["temp_scenario_radio_widget"] = st.session_state["applied_scenario"]

# Derived local variables for data queries and UI components
district = st.session_state["selected_district"]
crop = st.session_state["applied_crop"]
year = st.session_state["applied_year"]
scenario_param = st.session_state["applied_scenario"]
scenario_display = "SSP2-4.5" if scenario_param == "SSP2_4.5" else "SSP5-8.5"
crop_meta = CROP_DISPLAY[crop]

# ── GeoJSON & API Fetching with Caching ─────────────────────────────────────────
@st.cache_data
def load_geojson():
    with open("data/mp_boundaries.json", "r") as f:
        geojson = json.load(f)
    rename_map = {
        "AgarMalwa": "Agar Malwa",
        "EastNimar": "Khandwa",
        "WestNimar": "Khargone",
        "Narsimhapur": "Narsinghpur"
    }
    tikamgarh_feature = None
    for feature in geojson["features"]:
        name = feature["properties"]["district_name"]
        if name in rename_map:
            feature["properties"]["district_name"] = rename_map[name]
        if name == "Tikamgarh":
            tikamgarh_feature = feature
            
    # Add Niwari by duplicating Tikamgarh if not present
    has_niwari = any(f["properties"]["district_name"] == "Niwari" for f in geojson["features"])
    if not has_niwari and tikamgarh_feature:
        niwari_feature = copy.deepcopy(tikamgarh_feature)
        niwari_feature["properties"]["district_name"] = "Niwari"
        geojson["features"].append(niwari_feature)
    return geojson

@st.cache_data(ttl=300)
def fetch_bulk_map_data(crop: str, year: int, scenario: str):
    try:
        resp = requests.get(
            f"{API_URL}/dashboard/map_data",
            params={"crop": crop, "year": year, "scenario": scenario},
            headers=HEADERS,
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        st.error(f"Error fetching map data: {e}")
    return {}

@st.cache_data(ttl=300)
def fetch_trajectory_data(district: str, crop: str, scenario: str):
    try:
        resp = requests.get(
            f"{API_URL}/dashboard/trajectory",
            params={"district": district, "crop": crop, "scenario": scenario},
            headers=HEADERS,
            timeout=10,
        )
        if resp.status_code == 200:
            return pd.DataFrame(resp.json())
    except Exception:
        pass
    return pd.DataFrame()

# ── Sidebar Filters ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding: 8px 0 8px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 20px;'>
        <div style='font-size:18px; font-weight:700; color:#ffffff;'>🌾 {t['title']}</div>
        <div style='font-size:11px; color:#7aab7a; margin-top:2px;'>{t['sub_header']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:11px;color:#8b949e;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;'>{t['district_lbl']}</div>", unsafe_allow_html=True)
    mp_districts = [
        "Agar Malwa", "Alirajpur", "Anuppur", "Ashoknagar", "Balaghat", "Barwani", "Betul", "Bhind",
        "Bhopal", "Burhanpur", "Chhatarpur", "Chhindwara", "Damoh", "Datia", "Dewas", "Dhar",
        "Dindori", "Guna", "Gwalior", "Harda", "Hoshangabad", "Indore", "Jabalpur", "Jhabua",
        "Katni", "Khandwa", "Khargone", "Mandla", "Mandsaur", "Morena", "Narsinghpur", "Neemuch",
        "Niwari", "Panna", "Raisen", "Rajgarh", "Ratlam", "Rewa", "Sagar", "Satna", "Sehore",
        "Seoni", "Shahdol", "Shajapur", "Sheopur", "Shivpuri", "Sidhi", "Singrauli", "Tikamgarh",
        "Ujjain", "Umaria", "Vidisha"
    ]
    
    st.selectbox(
        t['district_lbl'], 
        mp_districts, 
        label_visibility="collapsed",
        key="temp_district_select"
    )

    st.markdown(f"<div style='font-size:11px;color:#8b949e;letter-spacing:1px;text-transform:uppercase;margin:20px 0 6px;'>{t['crop_lbl']}</div>", unsafe_allow_html=True)
    crop_options = list(CROP_DISPLAY.keys())

    st.selectbox(
        t['crop_lbl'],
        crop_options,
        format_func=lambda x: f"{CROP_DISPLAY[x]['emoji']} {t[x]}",
        label_visibility="collapsed",
        key="temp_crop_select",
    )

    st.markdown(f"<div style='font-size:11px;color:#8b949e;letter-spacing:1px;text-transform:uppercase;margin:20px 0 6px;'>{t['season_year_lbl']}</div>", unsafe_allow_html=True)
    st.slider(
        t['season_year_lbl'], 
        min_value=2026, 
        max_value=2041, 
        label_visibility="collapsed",
        key="temp_year_slider"
    )

    st.markdown(f"<div style='font-size:11px;color:#8b949e;letter-spacing:1px;text-transform:uppercase;margin:20px 0 6px;'>{t['scenario_lbl']}</div>", unsafe_allow_html=True)
    
    # Render scenario selection with dynamic labels while maintaining static internal keys
    st.radio(
        t['scenario_lbl'],
        ["SSP2_4.5", "SSP5_8.5"],
        format_func=lambda x: ssp2_label if x == "SSP2_4.5" else ssp5_label,
        index=0 if st.session_state["applied_scenario"] == "SSP2_4.5" else 1,
        label_visibility="collapsed",
        key="temp_scenario_radio_widget"
    )

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    search_clicked = st.button(t["search_btn"], use_container_width=True, type="primary")
    if search_clicked:
        st.session_state["selected_district"] = st.session_state["temp_district_select"]
        st.session_state["applied_crop"] = st.session_state["temp_crop_select"]
        st.session_state["applied_year"] = st.session_state["temp_year_slider"]
        st.session_state["applied_scenario"] = st.session_state["temp_scenario_radio_widget"]
        st.rerun()

    temp_scenario_display = "SSP2-4.5" if st.session_state["temp_scenario_radio_widget"] == "SSP2_4.5" else "SSP5-8.5"

    st.markdown(f"""
    <div style='margin-top:32px;padding:12px;background:rgba(22, 27, 34, 0.5);border:1px solid rgba(255,255,255,0.08);border-radius:8px;'>
        <div style='font-size:11px;color:#8b949e;'>
            <b style='color:#c9d1d9;'>{t['data_sources']}</b><br><br>
            🛰️ {t['dicra_api']}<br>
            🌧️ {t['chirps_drought']}<br>
            📊 {t['pmfby_cce']}<br>
            🌍 {t['cmip6'].format(scenario=temp_scenario_display)}<br>
            🌱 {t['fao_calendars']}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="dash-header">
    <div class="dash-title">
        {t['title']}
        <span class="badge">{t['mp_badge']}</span>
    </div>
    <div class="dash-subtitle">
        {t['sub_header']} &nbsp;·&nbsp; {t['sub_badge']} &nbsp;·&nbsp;
        {st.session_state["selected_district"]} &nbsp;·&nbsp; {crop_meta['emoji']} {t[crop]} &nbsp;·&nbsp;
        {t[crop_meta['season']]} {year} &nbsp;·&nbsp; {scenario_display}
    </div>
</div>
""", unsafe_allow_html=True)

# Fetch bulk map data
if not wait_for_backend(timeout_seconds=30):
    st.error(t["api_error"])
    st.code("cd CSI-DiCRA-Project && PYTHONPATH=. python -m uvicorn src.main:app --reload --port 8000")
    st.stop()

map_data = fetch_bulk_map_data(crop, year, scenario_param)

if not map_data:
    st.error(t["api_error"])
    st.code("cd CSI-DiCRA-Project && PYTHONPATH=. python -m uvicorn src.main:app --reload --port 8000")
    st.stop()


# ── Layout ──────────────────────────────────────────────────────────────────────
# Define the tabs below the CSI header
tab_map, tab_details, tab_actions, tab_trajectory = st.tabs([
    t["tab_map"],
    t["tab_details"],
    t["tab_actions"],
    t["tab_trajectory"]
])

with tab_map:
    st.markdown(f"<div class='section-label'>{t['sec_map']}</div>", unsafe_allow_html=True)
    
    # Layer selector control translated dynamically
    layer_options_mapped = [t["metric_risk_tier"], t["metric_spi3"], t["metric_sm"], t["metric_tmax"], t["metric_ndvi"]]
    selected_layer_translated = st.segmented_control(
        t["layer_control"],
        options=layer_options_mapped,
        default=t["metric_risk_tier"],
        label_visibility="collapsed",
        key="map_layer_control"
    )
    if not selected_layer_translated:
        selected_layer_translated = t["metric_risk_tier"]
        
    layer_mapping = {
        t["metric_risk_tier"]: "CSI Risk Tier",
        t["metric_spi3"]: "SPI-3",
        t["metric_sm"]: "Soil Moisture",
        t["metric_tmax"]: "Tmax",
        t["metric_ndvi"]: "NDVI"
    }
    selected_layer = layer_mapping.get(selected_layer_translated, "CSI Risk Tier")
    geojson = load_geojson()
    
    # Build Map DataFrame
    map_records = []
    for d_name in mp_districts:
        d_vals = map_data.get(d_name, {})
        rec = {
            "district": d_name,
            "NDVI": d_vals.get("NDVI", 0.0),
            "Tmax": d_vals.get("Tmax", 0.0),
            "Soil Moisture": d_vals.get("Soil Moisture", 0.0),
            "SPI-3": d_vals.get("SPI-3", 0.0),
            "CSI Risk Tier": d_vals.get("CSI Risk Tier", 1),
            "CSI Score": d_vals.get("CSI Score", 0.0),
            "Recommended Premium Adjustment": d_vals.get("Recommended Premium Adjustment", "Base Rate")
        }
        map_records.append(rec)
        
    map_df = pd.DataFrame(map_records)
    map_df["Recommended Premium Adjustment Translated"] = map_df["Recommended Premium Adjustment"].apply(
        lambda x: t["premium_base"] if x == "Base Rate" else x
    )
    
    # Layer configuration dictionary
    LAYER_CONFIGS = {
        "CSI Risk Tier": {
            "colorscale": [
                [0.0, "#3fb950"],
                [0.2, "#3fb950"],
                [0.2, "#a8cc3a"],
                [0.4, "#a8cc3a"],
                [0.4, "#d29922"],
                [0.6, "#d29922"],
                [0.6, "#f85149"],
                [0.8, "#f85149"],
                [0.8, "#ff6b6b"],
                [1.0, "#ff6b6b"]
            ],
            "zmin": 1,
            "zmax": 5,
            "colorbar_title": t["metric_risk_tier"],
        },
        "SPI-3": {
            "colorscale": "RdYlBu",
            "zmin": -2.0,
            "zmax": 2.0,
            "colorbar_title": t["metric_spi3"],
        },
        "Soil Moisture": {
            "colorscale": "YlGnBu",
            "zmin": 0.1,
            "zmax": 0.5,
            "colorbar_title": t["metric_sm"],
        },
        "Tmax": {
            "colorscale": "YlOrRd",
            "zmin": 25.0,
            "zmax": 40.0,
            "colorbar_title": t["metric_tmax"],
        },
        "NDVI": {
            "colorscale": "YlGn",
            "zmin": 0.2,
            "zmax": 0.8,
            "colorbar_title": t["metric_ndvi"],
        }
    }
    
    l_cfg = LAYER_CONFIGS[selected_layer]
    
    # Stack data for custom hovers
    customdata = np.stack((
        map_df["NDVI"],
        map_df["Tmax"],
        map_df["Soil Moisture"],
        map_df["SPI-3"],
        map_df["CSI Risk Tier"],
        map_df["CSI Score"],
        map_df["Recommended Premium Adjustment Translated"]
    ), axis=-1)
    
    fig = go.Figure()
    
    # Add base choropleth layer
    fig.add_trace(go.Choroplethmap(
        geojson=geojson,
        locations=map_df["district"],
        z=map_df[selected_layer],
        featureidkey="properties.district_name",
        colorscale=l_cfg["colorscale"],
        zmin=l_cfg["zmin"],
        zmax=l_cfg["zmax"],
        marker_opacity=0.6,
        marker_line_width=0.8,
        marker_line_color="rgba(255,255,255,0.15)",
        customdata=customdata,
        hovertemplate=(
            f"<b>{t['district_lbl']}: %{{location}}</b><br><br>"
            f"🔴 {t['metric_risk_tier']}: <b>%{{customdata[4]}} / 5</b><br>"
            f"📊 {t['metric_csi_score']}: <b>%{{customdata[5]:.2f}}</b><br>"
            f"🌱 {t['metric_ndvi']}: <b>%{{customdata[0]:.3f}}</b><br>"
            f"🌡️ {t['metric_tmax']}: <b>%{{customdata[1]:.1f}}°C</b><br>"
            f"💧 {t['metric_sm']}: <b>%{{customdata[2]:.3f}}</b><br>"
            f"🌧️ {t['metric_spi3']}: <b>%{{customdata[3]:.3f}}</b><br>"
            f"💼 {t['insurance_signal']}: <b>%{{customdata[6]}}</b><br>"
            "<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text=l_cfg["colorbar_title"], font=dict(color="#8b949e", size=11)),
            tickfont=dict(color="#8b949e", size=10),
            bgcolor="rgba(0,0,0,0)",
            thickness=12,
            len=0.6,
            yanchor="middle",
            y=0.5,
            xpad=10
        )
    ))
    
    # Add highlight layer for selected district
    selected_dist = st.session_state["selected_district"]
    selected_df = map_df[map_df["district"] == selected_dist]
    if not selected_df.empty:
        fig.add_trace(go.Choroplethmap(
            geojson=geojson,
            locations=selected_df["district"],
            z=selected_df[selected_layer],
            featureidkey="properties.district_name",
            colorscale=l_cfg["colorscale"],
            zmin=l_cfg["zmin"],
            zmax=l_cfg["zmax"],
            marker_opacity=0.9,
            marker_line_width=2.5,
            marker_line_color="#ffffff",
            showscale=False,
            hoverinfo="skip"
        ))
        
    fig.update_layout(
        map=dict(
            style="carto-darkmatter",
            center=dict(lat=23.3, lon=78.2),
            zoom=5.3,
            uirevision=True
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=620
    )
    
    event = st.plotly_chart(
        fig, 
        on_select="rerun", 
        selection_mode="points", 
        key="map_plot", 
        use_container_width=True
    )
    
    # Trigger state update on map click
    if event and "selection" in event and event["selection"]["points"]:
        clicked_district = event["selection"]["points"][0].get("location")
        if clicked_district and clicked_district in mp_districts:
            if clicked_district != st.session_state["selected_district"]:
                st.session_state["selected_district"] = clicked_district
                st.session_state["temp_district_select"] = clicked_district
                st.rerun()

with tab_details:
    st.markdown(f"<div class='section-label'>{t['sec_details']}</div>", unsafe_allow_html=True)
    
    # Load selected district stats
    dist_stats = map_data.get(st.session_state["selected_district"], {})
    tier = dist_stats.get("CSI Risk Tier", 1)
    csi_score = dist_stats.get("CSI Score", 0.0)
    premium_adj = dist_stats.get("Recommended Premium Adjustment", "Base Rate")
    premium_adj_translated = t["premium_base"] if premium_adj == "Base Rate" else premium_adj
    ndvi_val = dist_stats.get("NDVI", 0.0)
    tmax_val = dist_stats.get("Tmax", 0.0)
    sm_val = dist_stats.get("Soil Moisture", 0.0)
    spi3_val = dist_stats.get("SPI-3", 0.0)
    
    cfg = CROP_TIER_CONFIG[crop][tier]
    
    # 3-Column Grid of Metrics
    mc1, mc2, mc3 = st.columns(3, gap="medium")
    with mc1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{t['metric_risk_tier']}</div>
            <div class="metric-value" style="color:{cfg['color']};">{tier} / 5</div>
            <div class="metric-sub">{cfg['emoji']} {cfg['label'][lang]}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{t['metric_csi_score']}</div>
            <div class="metric-value">{csi_score:.2f}</div>
            <div class="metric-sub">{t['metric_csi_range']}</div>
        </div>""", unsafe_allow_html=True)
        
    with mc2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{t['metric_sm']}</div>
            <div class="metric-value" style="color:#7fb5ff;">{sm_val:.3f}</div>
            <div class="metric-sub">{t['metric_sm_sub']}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{t['metric_spi3']}</div>
            <div class="metric-value" style="color:#a8cc3a;">{spi3_val:.2f}</div>
            <div class="metric-sub">{t['metric_spi3_sub']}</div>
        </div>""", unsafe_allow_html=True)
        
    with mc3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{t['metric_ndvi']}</div>
            <div class="metric-value" style="color:#5cdb95;">{ndvi_val:.3f}</div>
            <div class="metric-sub">{t['metric_ndvi_sub']}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{t['metric_tmax']}</div>
            <div class="metric-value" style="color:#ff6b6b;">{tmax_val:.1f}°C</div>
            <div class="metric-sub">{t['metric_tmax_sub']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)
    
    # Premium Adjust Card
    st.markdown(f"""
    <div class="metric-card" style="padding: 14px 20px;">
        <div class="metric-label">{t['insurance_signal']}</div>
        <div class="metric-value" style="font-size:20px; color:#ffb703;">{premium_adj_translated}</div>
        <div class="metric-sub">{t['actuarial_premium']}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── SHAP Explainability Plot ──────────────────────────────────────────────────
    explainability = dist_stats.get("explainability", {})
    contributions = explainability.get("contributions", {})
    base_value = explainability.get("base_value", 0.0)
    
    if contributions:
        # Sort contributions for a clean top-to-bottom or bottom-to-top order
        sorted_contribs = sorted(contributions.items(), key=lambda x: x[1])
        features_list = [x[0] for x in sorted_contribs]
        values_list = [x[1] for x in sorted_contribs]
        
        # Colors: Green for yield-increasing factors, Red for yield-decreasing factors
        colors = ["rgba(239, 83, 80, 0.75)" if v < 0 else "rgba(38, 166, 154, 0.75)" for v in values_list]
        borders = ["#ef5350" if v < 0 else "#26a69a" for v in values_list]
        
        # Feature name translations
        feature_translations = {
            "Mean Temperature": t["shap_feat_temp"],
            "Soil Moisture": t["shap_feat_sm"],
            "NDVI Vegetation Index": t["shap_feat_ndvi"],
            "Total Precipitation": t["shap_feat_precip"],
            "Temperature Anomaly": t["shap_feat_temp_anom"],
            "Soil Moisture Anomaly": t["shap_feat_sm_anom"]
        }
        features_list_translated = [feature_translations.get(f, f) for f in features_list]
        
        fig_shap = go.Figure()
        fig_shap.add_trace(go.Bar(
            y=features_list_translated,
            x=values_list,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color=borders, width=1.2)
            ),
            hovertemplate="Factor: %{y}<br>Impact: %{x:+,d} kg/ha<extra></extra>"
        ))
        
        fig_shap.update_layout(
            title=dict(
                text=t["shap_title"].format(base_value=base_value),
                font=dict(color="#e4e6eb", size=13),
                x=0.0,
                y=0.98
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title=dict(
                    text=t["shap_xaxis"],
                    font=dict(color="#8b949e", size=10)
                ),
                tickfont=dict(color="#8b949e", size=9),
                gridcolor="rgba(255,255,255,0.08)",
                zerolinecolor="rgba(255,255,255,0.2)"
            ),
            yaxis=dict(
                tickfont=dict(color="#c9d1d9", size=11),
                gridcolor="rgba(255,255,255,0.0)"
            ),
            margin=dict(l=10, r=10, t=35, b=10),
            height=260,
        )
        
        st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.plotly_chart(fig_shap, use_container_width=True)


with tab_actions:
    st.markdown(f"<div class='section-label'>{t['sec_actions']}</div>", unsafe_allow_html=True)
    
    dist_stats = map_data.get(st.session_state["selected_district"], {})
    tier = dist_stats.get("CSI Risk Tier", 1)
    cfg = CROP_TIER_CONFIG[crop][tier]
    
    # Risk Assessment Banner
    st.markdown(f"""
    <div class="risk-banner {cfg['css']}">
        <div class="risk-title">{cfg['emoji']} {t['metric_risk_tier']} {tier} — {cfg['label'][lang]}</div>
        <div class="risk-body">{cfg['headline'][lang]}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)

    # Field Advisory Panel
    with st.container(border=True):
        st.markdown(f"<div class='advisory-heading'>{t['actions_title'].format(crop=t[crop], year=year)}</div>", unsafe_allow_html=True)
        for a in cfg["actions"][lang]:
            st.markdown(f"<div class='advisory-item'><span>{a}</span></div>", unsafe_allow_html=True)

with tab_trajectory:
    st.markdown(f"<div class='section-label'>{t['sec_trajectory']}</div>", unsafe_allow_html=True)
    
    # 15-Year Trajectory Panel
    with st.container(border=True):
        st.markdown(f"<div class='advisory-heading'>{t['traj_title']}</div>", unsafe_allow_html=True)
        
        trajectory_df = fetch_trajectory_data(st.session_state["selected_district"], crop, scenario_param)
        
        if not trajectory_df.empty:
            tier_colors = {1: "#3fb950", 2: "#a8cc3a", 3: "#d29922", 4: "#f85149", 5: "#ff6b6b"}
            bar_colors  = [tier_colors.get(int(t_idx), "#8b949e") for t_idx in trajectory_df["risk_tier"]]

            fig_traj = go.Figure()
            fig_traj.add_trace(go.Bar(
                x=trajectory_df["year"],
                y=trajectory_df["risk_tier"],
                marker_color=bar_colors,
                marker_line_color="rgba(0,0,0,0)",
                opacity=0.7,
                name=t["metric_risk_tier"],
                hovertemplate=f"<b>%{{x}}</b><br>{t['metric_risk_tier']}: %{{y}}<br><extra></extra>",
            ))

            selected_row = trajectory_df[trajectory_df["year"] == year]
            if not selected_row.empty:
                selected_risk_tier = selected_row.iloc[0]["risk_tier"]
                fig_traj.add_trace(go.Scatter(
                    x=selected_row["year"],
                    y=selected_row["risk_tier"],
                    mode="markers",
                    marker=dict(size=12, color="#ffffff", line=dict(color=tier_colors.get(int(selected_risk_tier), "#ffffff"), width=3)),
                    name=t["traj_selected"].format(year=year),
                    hovertemplate=f"<b>{t['traj_selected_hover'].format(year=year)}</b><br>{t['traj_tier_hover'].format(tier=tier)}<extra></extra>",
                ))

            fig_traj.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#8b949e", size=11),
                xaxis=dict(
                    title=t["traj_xaxis"], title_font=dict(color="#8b949e", size=10),
                    tickfont=dict(color="#8b949e"), gridcolor="rgba(255,255,255,0.06)",
                    tickmode="linear", dtick=3,
                ),
                yaxis=dict(
                    title=t["traj_yaxis"], title_font=dict(color="#8b949e", size=10),
                    tickfont=dict(color="#8b949e"), gridcolor="rgba(255,255,255,0.06)",
                    range=[0, 5.5], tickvals=[1, 2, 3, 4, 5],
                    ticktext=["1", "2", "3", "4", "5"],
                ),
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                height=220,
                bargap=0.25,
            )
            for t_val, t_color in tier_colors.items():
                fig_traj.add_hline(y=t_val, line_dash="dot", line_color=t_color, opacity=0.15, line_width=1)

            st.plotly_chart(fig_traj, use_container_width=True)
        else:
            st.info(t["no_trajectory"])

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='margin-top:40px;padding:16px 0;border-top:1px solid rgba(255, 255, 255, 0.08);
     display:flex;justify-content:space-between;align-items:center;'>
    <div style='font-size:11px;color:#8b949e;'>
        {t['footer_text1']}
    </div>
    <div style='font-size:11px;color:#8b949e;'>
        {t['footer_text2']}
    </div>
</div>
""", unsafe_allow_html=True)