import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import numpy as np
import json
import copy
import os

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000/api/v1"

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

st.set_page_config(
    page_title="CSI-DiCRA | Farmer Advisory",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
            "label": "Normal Season", "emoji": "🟢", "color": "#3fb950", "css": "risk-1",
            "headline": "Conditions are within normal range for Soybean. Standard Kharif practices recommended.",
            "actions": [
                "🌱 Proceed with standard sowing (June–July) as per local KVK norms.",
                "💧 Maintain regular irrigation — no acute drought stress anticipated at flowering.",
                "📋 No special variety changes required; use your district's recommended cultivar.",
                "📞 Consult KVK agronomist for optimal fertiliser timing at pod-fill stage.",
            ],
        },
        2: {
            "label": "Mild Stress", "emoji": "🟡", "color": "#a8cc3a", "css": "risk-2",
            "headline": "Mild compound heat + drought stress detected during the Soybean flowering window.",
            "actions": [
                "🌡️ Monitor temperature forecasts closely during July 25 – August 31.",
                "💧 Ensure supplemental irrigation is available at pod initiation (R1-R3 stages).",
                "🌾 Apply moisture-retentive mulching between rows to reduce soil water loss.",
                "📋 Document any stress observations for KVK and PMFBY reporting.",
            ],
        },
        3: {
            "label": "Moderate Risk", "emoji": "🟠", "color": "#d29922", "css": "risk-3",
            "headline": "Moderate compound heat + drought projected during Soybean critical window. Proactive action needed.",
            "actions": [
                "⏰ Advance sowing by 7–10 days to shift flowering away from peak stress period.",
                "🌿 Switch to semi-drought-tolerant variety: JS-9560 or NRC-86 are proven for MP.",
                "💧 Install drip or sprinkler irrigation — flood irrigation is inefficient under heat stress.",
                "📞 Enroll in your district's PMFBY scheme before sowing deadline.",
                "🌡️ Follow IMD AgroMet bulletins weekly during Kharif for real-time adjustments.",
            ],
        },
        4: {
            "label": "Severe Risk", "emoji": "🔴", "color": "#f85149", "css": "risk-4",
            "headline": "Severe compound stress projected. Significant agronomic action required for Soybean.",
            "actions": [
                "🚨 Shift to a shorter-duration, drought-tolerant Soybean variety immediately.",
                "⏰ Advance sowing by 14–21 days to move flowering window before peak heat.",
                "🌾 Intercrop Soybean with drought-resistant Arhar (Pigeonpea) for risk diversification.",
                "💧 Prioritize critical irrigation at R1 (flowering) and R3 (pod fill) growth stages.",
                "📋 Ensure PMFBY enrollment is complete; document field conditions with date-stamped photos.",
                "📞 Contact your KVK for district-specific emergency advisories.",
            ],
        },
        5: {
            "label": "Catastrophic", "emoji": "🔴", "color": "#ff6b6b", "css": "risk-5",
            "headline": "Catastrophic compound stress projected. Emergency measures required for Soybean.",
            "actions": [
                "⛔ Kharif Soybean in this district carries extreme financial risk this year.",
                "🔄 Strongly consider crop substitution: Arhar, Maize, or Bajra tolerate heat+drought far better.",
                "💰 Ensure PMFBY insurance enrollment — trigger probability is very high this season.",
                "🌱 If continuing Soybean, use only NRCS-approved extreme-stress varieties.",
                "💧 Deficit irrigation: target only R1 and R3 stages; do not flood-irrigate.",
                "📞 Escalate to District Agriculture Officer for emergency seed and input support.",
                "🏦 Explore Kisan Credit Card (KCC) crop-loss advance against expected PMFBY claim.",
            ],
        },
    },

    "Rice": {
        1: {
            "label": "Normal Season", "emoji": "🟢", "color": "#3fb950", "css": "risk-1",
            "headline": "Conditions are within normal range for Rice. Standard Kharif transplanting recommended.",
            "actions": [
                "🌱 Proceed with standard transplanting schedule (June–July nursery, July–Aug transplanting).",
                "💧 Maintain adequate standing water (5–7 cm) during panicle initiation and heading.",
                "📋 No special variety changes required for this season.",
                "📞 Consult KVK for optimal top-dress urea timing at panicle initiation.",
            ],
        },
        2: {
            "label": "Mild Stress", "emoji": "🟡", "color": "#a8cc3a", "css": "risk-2",
            "headline": "Mild heat + drought stress detected during Rice heading/grain-fill window (Sep).",
            "actions": [
                "🌡️ Monitor temperatures closely during September — heat at panicle emergence causes sterility.",
                "💧 Maintain standing water during heading; do not allow fields to dry out in September.",
                "🌾 Apply potassium (MOP) at panicle initiation to improve heat tolerance.",
                "📋 Document stress observations and unusual flowering dates for KVK reporting.",
            ],
        },
        3: {
            "label": "Moderate Risk", "emoji": "🟠", "color": "#d29922", "css": "risk-3",
            "headline": "Moderate compound stress projected during Rice heading. Proactive water management needed.",
            "actions": [
                "⏰ Advance nursery sowing by 10–15 days to shift heading before peak September heat.",
                "🌿 Switch to shorter-duration, heat-tolerant variety: Sahbhagi Dhan or MTU-1010.",
                "💧 Critical irrigation at panicle initiation and heading stages — do not allow stress at these points.",
                "📞 Enroll in PMFBY before sowing deadline.",
                "🌡️ Use IMD AgroMet bulletins to time final nitrogen application away from heat peaks.",
            ],
        },
        4: {
            "label": "Severe Risk", "emoji": "🔴", "color": "#f85149", "css": "risk-4",
            "headline": "Severe compound stress projected during Rice grain-fill. Major action required.",
            "actions": [
                "🚨 Use only heat and drought-tolerant varieties: Sahbhagi Dhan, DRR Dhan 42, or Swarna Sub1.",
                "⏰ Advance transplanting by 2–3 weeks to shift the critical window before peak heat.",
                "💧 Alternate Wetting and Drying (AWD) technique — conserves water while maintaining yield.",
                "🌾 Reduce seedling age at transplanting (25 days vs 35) for faster establishment.",
                "📋 Ensure PMFBY enrollment; photograph field conditions at heading stage.",
                "📞 Contact KVK for district emergency advisory on late-season irrigation scheduling.",
            ],
        },
        5: {
            "label": "Catastrophic", "emoji": "🔴", "color": "#ff6b6b", "css": "risk-5",
            "headline": "Catastrophic stress projected during Rice heading. Emergency measures required.",
            "actions": [
                "⛔ Kharif Rice in this district faces extreme risk of spikelet sterility and total yield loss.",
                "🔄 Consider crop substitution: Maize or Soybean are more resilient under these conditions.",
                "💰 Ensure PMFBY enrollment is complete — trigger probability is very high.",
                "🌱 If continuing Rice, use only extreme stress-tolerant varieties (Sahbhagi Dhan, Nagina-22).",
                "💧 Maintain standing water at all costs during September — even one day of drying at heading causes sterility.",
                "📞 Escalate to District Agriculture Officer for emergency water allocation and input support.",
                "🏦 Explore KCC crop-loss advance given the very high trigger probability this season.",
            ],
        },
    },

    "Maize": {
        1: {
            "label": "Normal Season", "emoji": "🟢", "color": "#3fb950", "css": "risk-1",
            "headline": "Normal conditions for Maize. Standard Kharif practices apply.",
            "actions": [
                "🌱 Proceed with standard sowing (June–July) as per local recommendations.",
                "💧 Ensure one irrigation at tasseling (V14–VT stage) for maximum pollination.",
                "📋 Standard fertiliser schedule — apply top-dress urea at knee-height (V6) stage.",
                "📞 Consult KVK for hybrid selection suited to your district's soil type.",
            ],
        },
        2: {
            "label": "Mild Stress", "emoji": "🟡", "color": "#a8cc3a", "css": "risk-2",
            "headline": "Mild heat + drought stress detected during Maize tasseling/silking window.",
            "actions": [
                "🌡️ Monitor temperatures during August 10 – September 10 — heat above 35°C at tasseling causes pollen sterility.",
                "💧 Ensure one critical irrigation at tasseling (V14-VT) and one at silking (R1).",
                "🌾 Apply foliar potassium spray (1% KCl) during tasseling to improve heat tolerance.",
                "📋 Document stress symptoms (tassel blast, poor kernel set) for KVK records.",
            ],
        },
        3: {
            "label": "Moderate Risk", "emoji": "🟠", "color": "#d29922", "css": "risk-3",
            "headline": "Moderate compound stress projected during Maize critical tasseling window.",
            "actions": [
                "⏰ Advance sowing by 7–10 days to shift tasseling before peak August heat.",
                "🌿 Use drought-tolerant hybrids: Vivek QPM-9, DKC-9108, or locally recommended stress-tolerant varieties.",
                "💧 Two critical irrigations: at tasseling (V14-VT) and at silking (R1) — non-negotiable.",
                "📞 Register with PMFBY before sowing deadline.",
                "🌾 Avoid high nitrogen rates if drought stress is expected — increases lodging risk.",
            ],
        },
        4: {
            "label": "Severe Risk", "emoji": "🔴", "color": "#f85149", "css": "risk-4",
            "headline": "Severe stress projected at Maize tasseling. Significant action required.",
            "actions": [
                "🚨 Shift to drought-tolerant hybrid varieties only — standard hybrids will fail at tasseling.",
                "⏰ Advance sowing by 14–21 days to move tasseling away from the peak heat window.",
                "💧 Drip or sprinkler irrigation at tasseling and grain fill is critical — flood irrigation is insufficient.",
                "🌾 Reduce plant population by 10–15% to lower water demand per plant.",
                "📋 PMFBY enrollment mandatory — ensure it is complete before sowing.",
                "📞 Contact KVK for emergency seed substitution if standard seed lots are already purchased.",
            ],
        },
        5: {
            "label": "Catastrophic", "emoji": "🔴", "color": "#ff6b6b", "css": "risk-5",
            "headline": "Catastrophic compound stress projected at Maize tasseling. Emergency action required.",
            "actions": [
                "⛔ Kharif Maize in this district faces very high risk of total pollen failure and yield collapse.",
                "🔄 Consider crop substitution: Arhar (Pigeonpea) or Bajra are far more heat+drought tolerant.",
                "💰 PMFBY enrollment is critical — trigger probability is very high this season.",
                "🌱 If continuing Maize, use only ICAR-approved extreme heat-tolerant hybrids.",
                "💧 Deficit irrigation: prioritise tasseling and R1 stages exclusively.",
                "📞 Escalate to District Agriculture Officer for emergency seed and input support.",
                "🏦 Explore KCC crop-loss advance against expected PMFBY payout.",
            ],
        },
    },

    "Wheat": {
        1: {
            "label": "Normal Season", "emoji": "🟢", "color": "#3fb950", "css": "risk-1",
            "headline": "Normal Rabi conditions for Wheat. Standard sowing and irrigation schedule applies.",
            "actions": [
                "🌱 Proceed with standard sowing (November–December) as per district norms.",
                "💧 Maintain the standard 5–6 irrigation schedule: CRI, tillering, jointing, heading, grain fill.",
                "📋 Standard fertiliser schedule — basal DAP + top-dress urea at CRI and tillering.",
                "📞 Consult KVK for variety selection suited to your district's soil type.",
            ],
        },
        2: {
            "label": "Mild Stress", "emoji": "🟡", "color": "#a8cc3a", "css": "risk-2",
            "headline": "Mild unseasonal warming detected during the Wheat heading window (Jan–Feb).",
            "actions": [
                "🌡️ Monitor temperature forecasts in January–February — even 3–4°C above normal at heading causes sterility.",
                "💧 Ensure irrigation at heading (Zadoks 49–55) — do not allow moisture stress at this stage.",
                "🌾 Apply micronutrient spray (Zinc 0.5%) at booting stage to improve stress resilience.",
                "📋 Document any premature heading or bleaching symptoms for KVK records.",
            ],
        },
        3: {
            "label": "Moderate Risk", "emoji": "🟠", "color": "#d29922", "css": "risk-3",
            "headline": "Moderate terminal heat stress projected during Wheat heading. Proactive action needed.",
            "actions": [
                "⏰ Advance sowing by 7–10 days (early November) to complete grain fill before February heat.",
                "🌿 Use heat-tolerant varieties: HD-2781, HI-8498, or GW-496 for this district.",
                "💧 Critical irrigation at booting and heading stages — missing either one under heat stress causes 30-40% yield loss.",
                "📞 Enroll in PMFBY before the Rabi sowing deadline.",
                "🌡️ Follow IMD's weekly AgroMet bulletin for precise irrigation timing.",
            ],
        },
        4: {
            "label": "Severe Risk", "emoji": "🔴", "color": "#f85149", "css": "risk-4",
            "headline": "Severe terminal heat stress projected for Wheat. Major agronomic action required.",
            "actions": [
                "🚨 Sow only heat-tolerant varieties: HD-2781, HI-8498, K-0307 — standard HD-2967 will underperform severely.",
                "⏰ Sow in first fortnight of November — every 10 days of delayed sowing increases terminal heat exposure.",
                "💧 Irrigation at booting, heading AND grain-fill stages is mandatory under heat stress conditions.",
                "🌾 Apply 'Stay Green' foliar micronutrients (Zinc + Boron) at heading to extend grain-fill duration.",
                "📋 PMFBY enrollment is mandatory — complete before deadline.",
                "📞 Contact KVK for emergency variety and input advisory for late-sown fields.",
            ],
        },
        5: {
            "label": "Catastrophic", "emoji": "🔴", "color": "#ff6b6b", "css": "risk-5",
            "headline": "Catastrophic terminal heat projected for Wheat. Emergency measures required.",
            "actions": [
                "⛔ Rabi Wheat in this district faces extreme risk of grain shrivelling and near-total yield failure.",
                "🔄 Strongly consider substitution with Gram or Mustard — both tolerate Rabi heat far better.",
                "💰 PMFBY enrollment is critical — trigger probability is very high this season.",
                "🌱 If continuing Wheat, use only ICAR-approved extreme heat-tolerant varieties for MP.",
                "💧 Maximum irrigation support at booting, heading, and grain-fill — 3 critical irrigations.",
                "📞 Escalate to District Agriculture Officer for emergency seed access and input subsidies.",
                "🏦 Explore KCC crop-loss advance against expected PMFBY claim.",
            ],
        },
    },

    "Gram": {
        1: {
            "label": "Normal Season", "emoji": "🟢", "color": "#3fb950", "css": "risk-1",
            "headline": "Normal Rabi conditions for Gram (Chickpea). Standard practices apply.",
            "actions": [
                "🌱 Proceed with standard sowing (October–November) under rainfed or light irrigation.",
                "💧 Gram is largely rainfed — avoid excess irrigation at flowering (causes flower drop).",
                "📋 Standard fertiliser: basal P and K only — Gram fixes its own nitrogen.",
                "📞 Consult KVK for wilt-resistant variety suitable to your district's soil.",
            ],
        },
        2: {
            "label": "Mild Stress", "emoji": "🟡", "color": "#a8cc3a", "css": "risk-2",
            "headline": "Mild heat + drought stress detected during Gram flowering window (Dec–Jan).",
            "actions": [
                "🌡️ Monitor temperatures in December–January — temperatures above 30°C at flowering cause pod abortion.",
                "💧 One light pre-sowing irrigation and one at pod-fill is sufficient — avoid over-irrigation.",
                "🌾 Apply sulfur (20 kg/ha) at sowing — improves drought resilience in Gram.",
                "📋 Document any premature flower drop or wilting symptoms for KVK records.",
            ],
        },
        3: {
            "label": "Moderate Risk", "emoji": "🟠", "color": "#d29922", "css": "risk-3",
            "headline": "Moderate compound stress projected during Gram critical flowering window.",
            "actions": [
                "⏰ Advance sowing by 7–10 days (early October) to complete flowering before January heat.",
                "🌿 Use heat-tolerant variety: JAKI-9218 or JG-14 are recommended for MP conditions.",
                "💧 One critical irrigation at pod initiation (avoid at full flowering — causes flower drop).",
                "📞 Enroll in PMFBY before Rabi sowing deadline.",
                "🌾 Reduce plant population by 10% to lower water demand per plant under stress.",
            ],
        },
        4: {
            "label": "Severe Risk", "emoji": "🔴", "color": "#f85149", "css": "risk-4",
            "headline": "Severe stress projected during Gram flowering. Significant action required.",
            "actions": [
                "🚨 Shift to heat-tolerant short-duration varieties: JAKI-9218 or Radhey (JG-74).",
                "⏰ Advance sowing to first fortnight of October to avoid January heat at flowering.",
                "💧 One targeted irrigation at pod initiation ONLY — avoid irrigating during full bloom.",
                "🌾 Intercrop with Mustard (2:1 ratio) — Mustard provides partial shade and diversifies income.",
                "📋 PMFBY enrollment is mandatory — complete before deadline.",
                "📞 Contact KVK for emergency variety and input advisory.",
            ],
        },
        5: {
            "label": "Catastrophic", "emoji": "🔴", "color": "#ff6b6b", "css": "risk-5",
            "headline": "Catastrophic compound stress projected for Gram. Emergency action required.",
            "actions": [
                "⛔ Gram in this district faces extreme risk of total pod abortion and yield failure.",
                "🔄 Strongly consider substituting with Mustard or Lentil — both tolerate heat better than Gram.",
                "💰 PMFBY enrollment is critical — trigger probability is very high this season.",
                "🌱 If continuing Gram, use only ICAR extreme heat-tolerant varieties.",
                "💧 One targeted irrigation at pod initiation — any additional irrigation will cause more flower drop.",
                "📞 Escalate to District Agriculture Officer for emergency input support.",
                "🏦 Explore KCC crop-loss advance against expected PMFBY claim.",
            ],
        },
    },

    "Mustard": {
        1: {
            "label": "Normal Season", "emoji": "🟢", "color": "#3fb950", "css": "risk-1",
            "headline": "Normal Rabi conditions for Mustard. Standard practices recommended.",
            "actions": [
                "🌱 Proceed with standard sowing (October–November) for your district.",
                "💧 One irrigation at flowering and one at siliqua development is standard.",
                "📋 Standard fertiliser: basal DAP + top-dress nitrogen at flowering.",
                "📞 Consult KVK for variety selection (PM-28, RH-749, or Varuna) for your district.",
            ],
        },
        2: {
            "label": "Mild Stress", "emoji": "🟡", "color": "#a8cc3a", "css": "risk-2",
            "headline": "Mild heat + drought stress detected during the Mustard flowering window (Dec–Jan).",
            "actions": [
                "🌡️ Monitor temperatures in December–January — heat above 28°C at full bloom causes pod shatter.",
                "💧 Ensure irrigation at flowering stage — do not allow moisture stress during bloom.",
                "🌾 Watch for aphid infestation — warm Rabi conditions accelerate aphid population cycles in Mustard.",
                "📋 Document any premature siliqua drop or aphid damage for KVK reporting.",
            ],
        },
        3: {
            "label": "Moderate Risk", "emoji": "🟠", "color": "#d29922", "css": "risk-3",
            "headline": "Moderate compound stress projected during Mustard critical flowering window.",
            "actions": [
                "⏰ Advance sowing by 7–10 days (early October) to complete flowering before January heat.",
                "🌿 Use heat-tolerant variety: PM-28 or Pusa Mustard-27 are suitable for MP conditions.",
                "💧 Two irrigations: one at flowering and one at siliqua development — both are critical.",
                "📞 Enroll in PMFBY before Rabi sowing deadline.",
                "🌾 Apply thiourea spray (500 ppm) at budding stage — proven to reduce heat stress in Mustard.",
            ],
        },
        4: {
            "label": "Severe Risk", "emoji": "🔴", "color": "#f85149", "css": "risk-4",
            "headline": "Severe heat stress projected at Mustard flowering. Major action required.",
            "actions": [
                "🚨 Shift to early-maturing heat-tolerant varieties only: PM-28 or Pusa Mustard-27.",
                "⏰ Advance sowing to first fortnight of October — every 10-day delay increases heat exposure at bloom.",
                "💧 Two critical irrigations: at branching and at pod development — do not skip.",
                "🌾 Apply thiourea foliar spray (500 ppm) at full flowering to reduce pollen sterility.",
                "📋 PMFBY enrollment is mandatory — complete before deadline.",
                "📞 Contact KVK for emergency variety access if standard seed lots are already purchased.",
            ],
        },
        5: {
            "label": "Catastrophic", "emoji": "🔴", "color": "#ff6b6b", "css": "risk-5",
            "headline": "Catastrophic heat stress projected for Mustard. Emergency measures required.",
            "actions": [
                "⛔ Rabi Mustard in this district faces extreme risk of mass pod shatter and near-total yield loss.",
                "🔄 Consider substituting with Gram or Lentil — both tolerate the Rabi heat profile better in severe years.",
                "💰 PMFBY enrollment is critical — trigger probability is very high this season.",
                "🌱 If continuing Mustard, use only ICAR-approved extreme heat-tolerant varieties.",
                "💧 Two critical irrigations at branching and pod-fill — do not reduce under any cost pressure.",
                "📞 Escalate to District Agriculture Officer for emergency seed and input support.",
                "🏦 Explore KCC crop-loss advance against expected PMFBY claim this season.",
            ],
        },
    },
}
# Initialize session state for selected district and applied filters
if "selected_district" not in st.session_state:
    st.session_state["selected_district"] = "Bhopal"
if "applied_crop" not in st.session_state:
    st.session_state["applied_crop"] = "Soybean"
if "applied_year" not in st.session_state:
    st.session_state["applied_year"] = 2026
if "applied_scenario" not in st.session_state:
    st.session_state["applied_scenario"] = "SSP2_4.5"

# Sync temporary filters from applied/session state if not set
if "temp_district_select" not in st.session_state:
    st.session_state["temp_district_select"] = st.session_state["selected_district"]
if "temp_crop_select" not in st.session_state:
    st.session_state["temp_crop_select"] = st.session_state["applied_crop"]
if "temp_year_slider" not in st.session_state:
    st.session_state["temp_year_slider"] = st.session_state["applied_year"]
if "temp_scenario_radio" not in st.session_state:
    st.session_state["temp_scenario_radio"] = "SSP2-4.5 — Moderate Action" if st.session_state["applied_scenario"] == "SSP2_4.5" else "SSP5-8.5 — Worst Case"

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

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 8px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 20px;'>
        <div style='font-size:18px; font-weight:700; color:#ffffff;'>🌾 CSI-DiCRA</div>
        <div style='font-size:11px; color:#7aab7a; margin-top:2px;'>Farmer Advisory Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:11px;color:#8b949e;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;'>District</div>", unsafe_allow_html=True)
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
        "District", 
        mp_districts, 
        label_visibility="collapsed",
        key="temp_district_select"
    )

    st.markdown("<div style='font-size:11px;color:#8b949e;letter-spacing:1px;text-transform:uppercase;margin:20px 0 6px;'>Crop</div>", unsafe_allow_html=True)
    crop_options = list(CROP_DISPLAY.keys())

    st.selectbox(
        "Crop",
        crop_options,
        label_visibility="collapsed",
        key="temp_crop_select",
    )

    st.markdown("<div style='font-size:11px;color:#8b949e;letter-spacing:1px;text-transform:uppercase;margin:20px 0 6px;'>Season Year</div>", unsafe_allow_html=True)
    st.slider(
        "Season Year", 
        min_value=2026, 
        max_value=2041, 
        label_visibility="collapsed",
        key="temp_year_slider"
    )

    st.markdown("<div style='font-size:11px;color:#8b949e;letter-spacing:1px;text-transform:uppercase;margin:20px 0 6px;'>Climate Scenario</div>", unsafe_allow_html=True)
    scenario_options = ["SSP2-4.5 — Moderate Action", "SSP5-8.5 — Worst Case"]

    st.radio(
        "Climate Scenario",
        scenario_options,
        label_visibility="collapsed",
        key="temp_scenario_radio"
    )

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    search_clicked = st.button("Search 🔍", use_container_width=True, type="primary")
    if search_clicked:
        st.session_state["selected_district"] = st.session_state["temp_district_select"]
        st.session_state["applied_crop"] = st.session_state["temp_crop_select"]
        st.session_state["applied_year"] = st.session_state["temp_year_slider"]
        st.session_state["applied_scenario"] = "SSP2_4.5" if "SSP2" in st.session_state["temp_scenario_radio"] else "SSP5_8.5"
        st.rerun()

    temp_scenario_display = "SSP2-4.5" if "SSP2" in st.session_state["temp_scenario_radio"] else "SSP5-8.5"

    st.markdown("""
    <div style='margin-top:32px;padding:12px;background:rgba(22, 27, 34, 0.5);border:1px solid rgba(255,255,255,0.08);border-radius:8px;'>
        <div style='font-size:11px;color:#8b949e;'>
            <b style='color:#c9d1d9;'>Data Sources</b><br><br>
            🛰️ DiCRA API (Tmax, NDVI, SM)<br>
            🌧️ CHIRPS (SPI-3 Drought)<br>
            📊 PMFBY CCE Records<br>
            🌍 CMIP6 ({scenario})<br>
            🌱 FAO Crop Calendars
        </div>
    </div>
    """.format(scenario=temp_scenario_display), unsafe_allow_html=True)


# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="dash-header">
    <div class="dash-title">
        Compound Stress Index
        <span class="badge">MADHYA PRADESH</span>
    </div>
    <div class="dash-subtitle">
        Data in Climate Resilient Agriculture &nbsp;·&nbsp; District Advisory &nbsp;·&nbsp;
        {st.session_state["selected_district"]} &nbsp;·&nbsp; {crop_meta['emoji']} {crop} &nbsp;·&nbsp;
        {crop_meta['season']} {year} &nbsp;·&nbsp; {scenario_display}
    </div>
</div>
""", unsafe_allow_html=True)

# Fetch bulk map data
map_data = fetch_bulk_map_data(crop, year, scenario_param)

if not map_data:
    st.error("⚠️ Cannot connect to the FastAPI backend or load map data. Ensure it is running on port 8000.")
    st.code("cd CSI-DiCRA-Project && PYTHONPATH=. python -m uvicorn src.main:app --reload --port 8000")
    st.stop()

# ── Layout ──────────────────────────────────────────────────────────────────────
# Define the tabs below the CSI header
tab_map, tab_details, tab_actions, tab_trajectory = st.tabs([
    "🗺️ Interactive Map",
    "📊 Selected District Details",
    "🌾 Recommended Actions",
    "📈 15-Year Risk Trajectory"
])

with tab_map:
    st.markdown("<div class='section-label'>Interactive District Map</div>", unsafe_allow_html=True)
    
    # Layer selector control
    selected_layer = st.segmented_control(
        "Map Layer Select",
        options=["CSI Risk Tier", "SPI-3", "Soil Moisture", "Tmax", "NDVI"],
        default="CSI Risk Tier",
        label_visibility="collapsed",
        key="map_layer_control"
    )
    if not selected_layer:
        selected_layer = "CSI Risk Tier"
        
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
            "colorbar_title": "Risk Tier",
        },
        "SPI-3": {
            "colorscale": "RdYlBu",
            "zmin": -2.0,
            "zmax": 2.0,
            "colorbar_title": "SPI-3 Index",
        },
        "Soil Moisture": {
            "colorscale": "YlGnBu",
            "zmin": 0.1,
            "zmax": 0.5,
            "colorbar_title": "Soil Moisture",
        },
        "Tmax": {
            "colorscale": "YlOrRd",
            "zmin": 25.0,
            "zmax": 40.0,
            "colorbar_title": "Tmax (°C)",
        },
        "NDVI": {
            "colorscale": "YlGn",
            "zmin": 0.2,
            "zmax": 0.8,
            "colorbar_title": "NDVI",
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
        map_df["Recommended Premium Adjustment"]
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
            "<b>District: %{location}</b><br><br>"
            "🔴 CSI Risk Tier: <b>%{customdata[4]} / 5</b><br>"
            "📊 CSI Score: <b>%{customdata[5]:.2f}</b><br>"
            "🌱 NDVI: <b>%{customdata[0]:.3f}</b><br>"
            "🌡️ Tmax: <b>%{customdata[1]:.1f}°C</b><br>"
            "💧 Soil Moisture: <b>%{customdata[2]:.3f}</b><br>"
            "🌧️ SPI-3: <b>%{customdata[3]:.3f}</b><br>"
            "💼 Premium: <b>%{customdata[6]}</b><br>"
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
    st.markdown("<div class='section-label'>Selected District Details</div>", unsafe_allow_html=True)
    
    # Load selected district stats
    dist_stats = map_data.get(st.session_state["selected_district"], {})
    tier = dist_stats.get("CSI Risk Tier", 1)
    csi_score = dist_stats.get("CSI Score", 0.0)
    premium_adj = dist_stats.get("Recommended Premium Adjustment", "Base Rate")
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
            <div class="metric-label">Risk Tier</div>
            <div class="metric-value" style="color:{cfg['color']};">{tier} / 5</div>
            <div class="metric-sub">{cfg['emoji']} {cfg['label']}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CSI Score</div>
            <div class="metric-value">{csi_score:.2f}</div>
            <div class="metric-sub">0.0 = Min · 1.0 = Max</div>
        </div>""", unsafe_allow_html=True)
        
    with mc2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Soil Moisture</div>
            <div class="metric-value" style="color:#7fb5ff;">{sm_val:.3f}</div>
            <div class="metric-sub">Window Avg m³/m³</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">SPI-3 Index</div>
            <div class="metric-value" style="color:#a8cc3a;">{spi3_val:.2f}</div>
            <div class="metric-sub">Drought Metric</div>
        </div>""", unsafe_allow_html=True)
        
    with mc3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">NDVI</div>
            <div class="metric-value" style="color:#5cdb95;">{ndvi_val:.3f}</div>
            <div class="metric-sub">Vegetation Index</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tmax</div>
            <div class="metric-value" style="color:#ff6b6b;">{tmax_val:.1f}°C</div>
            <div class="metric-sub">Window Max Temp</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)
    
    # Premium Adjust Card
    st.markdown(f"""
    <div class="metric-card" style="padding: 14px 20px;">
        <div class="metric-label">Insurance Premium Signal</div>
        <div class="metric-value" style="font-size:20px; color:#ffb703;">{premium_adj}</div>
        <div class="metric-sub">Actuarial Risk Premium Adjustment</div>
    </div>
    """, unsafe_allow_html=True)

    # ── SHAP Explainability Plot ──────────────────────────────────────────────────
    explainability = dist_stats.get("explainability", {})
    contributions = explainability.get("contributions", {})
    base_value = explainability.get("base_value", 0.0)
    
    if contributions:
        import plotly.graph_objects as go
        
        # Sort contributions for a clean top-to-bottom or bottom-to-top order
        sorted_contribs = sorted(contributions.items(), key=lambda x: x[1])
        features_list = [x[0] for x in sorted_contribs]
        values_list = [x[1] for x in sorted_contribs]
        
        # Colors: Green for yield-increasing factors, Red for yield-decreasing factors
        colors = ["rgba(239, 83, 80, 0.75)" if v < 0 else "rgba(38, 166, 154, 0.75)" for v in values_list]
        borders = ["#ef5350" if v < 0 else "#26a69a" for v in values_list]
        
        fig_shap = go.Figure()
        fig_shap.add_trace(go.Bar(
            y=features_list,
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
                text=f"🔍 Actuarial Explainability: Yield Deficit/Surplus (Base: {base_value:,.0f} kg/ha)",
                font=dict(color="#e4e6eb", size=13),
                x=0.0,
                y=0.98
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="Yield Deviation Contribution (kg/ha)",
                titlefont=dict(color="#8b949e", size=10),
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
    st.markdown("<div class='section-label'>Risk Assessment & Advisories</div>", unsafe_allow_html=True)
    
    dist_stats = map_data.get(st.session_state["selected_district"], {})
    tier = dist_stats.get("CSI Risk Tier", 1)
    cfg = CROP_TIER_CONFIG[crop][tier]
    
    # Risk Assessment Banner
    st.markdown(f"""
    <div class="risk-banner {cfg['css']}">
        <div class="risk-title">{cfg['emoji']} Tier {tier} — {cfg['label']}</div>
        <div class="risk-body">{cfg['headline']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)

    # Field Advisory Panel
    with st.container(border=True):
        st.markdown(f"<div class='advisory-heading'>🌾 Recommended Actions — {crop} {year}</div>", unsafe_allow_html=True)
        for a in cfg["actions"]:
            st.markdown(f"<div class='advisory-item'><span>{a}</span></div>", unsafe_allow_html=True)

with tab_trajectory:
    st.markdown("<div class='section-label'>15-Year Risk Trajectory</div>", unsafe_allow_html=True)
    
    # 15-Year Trajectory Panel
    with st.container(border=True):
        st.markdown("<div class='advisory-heading'>📈 15-Year Risk Trajectory (2026 – 2041)</div>", unsafe_allow_html=True)
        
        trajectory_df = fetch_trajectory_data(st.session_state["selected_district"], crop, scenario_param)
        
        if not trajectory_df.empty:
            tier_colors = {1: "#3fb950", 2: "#a8cc3a", 3: "#d29922", 4: "#f85149", 5: "#ff6b6b"}
            bar_colors  = [tier_colors.get(int(t), "#8b949e") for t in trajectory_df["risk_tier"]]

            fig_traj = go.Figure()
            fig_traj.add_trace(go.Bar(
                x=trajectory_df["year"],
                y=trajectory_df["risk_tier"],
                marker_color=bar_colors,
                marker_line_color="rgba(0,0,0,0)",
                opacity=0.7,
                name="Risk Tier",
                hovertemplate="<b>%{x}</b><br>Risk Tier: %{y}<br><extra></extra>",
            ))

            selected_row = trajectory_df[trajectory_df["year"] == year]
            if not selected_row.empty:
                fig_traj.add_trace(go.Scatter(
                    x=selected_row["year"],
                    y=selected_row["risk_tier"],
                    mode="markers",
                    marker=dict(size=12, color="#ffffff", line=dict(color=cfg["color"], width=3)),
                    name=f"Selected ({year})",
                    hovertemplate=f"<b>Selected Year: {year}</b><br>Tier: {tier}<extra></extra>",
                ))

            fig_traj.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#8b949e", size=11),
                xaxis=dict(
                    title="Year", title_font=dict(color="#8b949e", size=10),
                    tickfont=dict(color="#8b949e"), gridcolor="rgba(255,255,255,0.06)",
                    tickmode="linear", dtick=3,
                ),
                yaxis=dict(
                    title="Risk Tier", title_font=dict(color="#8b949e", size=10),
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
            st.info("No trajectory data available.")

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:40px;padding:16px 0;border-top:1px solid rgba(255, 255, 255, 0.08);
     display:flex;justify-content:space-between;align-items:center;'>
    <div style='font-size:11px;color:#8b949e;'>
        CSI-DiCRA Actuarial Climate Risk Engine &nbsp;·&nbsp; Data: DiCRA · CHIRPS · PMFBY · CMIP6
    </div>
    <div style='font-size:11px;color:#8b949e;'>
        NABARD &nbsp;·&nbsp; UNDP &nbsp;·&nbsp; CSI-DiCRA-Project-V2
    </div>
</div>
""", unsafe_allow_html=True)