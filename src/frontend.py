import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000/api/v1"

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

    .main { background-color: #0f1117; }

    /* Header */
    .dash-header {
        background: linear-gradient(135deg, #1a2a1a 0%, #0d1f0d 100%);
        border: 1px solid #2d4a2d;
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
    }
    .dash-title {
        font-size: 28px; font-weight: 700;
        color: #ffffff; margin: 0; letter-spacing: -0.5px;
    }
    .dash-subtitle {
        font-size: 13px; color: #7aab7a; margin-top: 4px; font-weight: 400;
    }
    .badge {
        display: inline-block;
        background: #2d4a2d; color: #7aff7a;
        font-size: 11px; font-weight: 600; letter-spacing: 1px;
        padding: 3px 10px; border-radius: 20px;
        margin-left: 10px; vertical-align: middle;
    }

    /* Metric Cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-label {
        font-size: 11px; font-weight: 600; letter-spacing: 1.5px;
        text-transform: uppercase; color: #8b949e; margin-bottom: 8px;
    }
    .metric-value {
        font-size: 36px; font-weight: 700; color: #ffffff; line-height: 1;
    }
    .metric-sub { font-size: 12px; color: #8b949e; margin-top: 6px; }

    /* Risk Banner */
    .risk-banner {
        border-radius: 12px;
        padding: 20px 28px;
        margin: 8px 0;
        border-left: 5px solid;
    }
    .risk-1 { background: #0d2b0d; border-color: #3fb950; color: #3fb950; }
    .risk-2 { background: #1e2a0d; border-color: #a8cc3a; color: #a8cc3a; }
    .risk-3 { background: #2a1e0d; border-color: #d29922; color: #d29922; }
    .risk-4 { background: #2a110d; border-color: #f85149; color: #f85149; }
    .risk-5 { background: #2a0d0d; border-color: #ff0000; color: #ff6b6b; }

    .risk-title { font-size: 18px; font-weight: 700; margin-bottom: 6px; }
    .risk-body  { font-size: 14px; opacity: 0.85; line-height: 1.6; }

    /* Advisory Box */
    .advisory-box {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 8px;
    }
    .advisory-heading {
        font-size: 13px; font-weight: 600; letter-spacing: 1px;
        color: #8b949e; text-transform: uppercase; margin-bottom: 12px;
    }
    .advisory-item {
        font-size: 14px; color: #c9d1d9;
        padding: 6px 0; border-bottom: 1px solid #21262d;
        display: flex; align-items: flex-start; gap: 10px;
    }
    .advisory-item:last-child { border-bottom: none; }

    /* Section label */
    .section-label {
        font-size: 11px; font-weight: 600; letter-spacing: 1.5px;
        text-transform: uppercase; color: #8b949e;
        margin: 28px 0 12px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #21262d; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label { color: #8b949e !important; font-size: 12px !important; }

    /* Hide default Streamlit chrome but keep sidebar toggle */
    #MainMenu, footer { visibility: hidden; }
    header { visibility: visible; background: transparent !important; }
    .block-container { padding-top: 24px; }
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

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 8px; border-bottom: 1px solid #21262d; margin-bottom: 20px;'>
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
    district = st.selectbox("District", mp_districts, label_visibility="collapsed")

    st.markdown("<div style='font-size:11px;color:#8b949e;letter-spacing:1px;text-transform:uppercase;margin:20px 0 6px;'>Crop</div>", unsafe_allow_html=True)
    crop = st.selectbox(
        "Crop",
        list(CROP_DISPLAY.keys()),
        label_visibility="collapsed",
        key="crop_select",
    )
    crop_meta = CROP_DISPLAY[crop]

    st.markdown("<div style='font-size:11px;color:#8b949e;letter-spacing:1px;text-transform:uppercase;margin:20px 0 6px;'>Season Year</div>", unsafe_allow_html=True)
    year = st.slider("Season Year", min_value=2026, max_value=2041, value=2026, label_visibility="collapsed")

    st.markdown("<div style='font-size:11px;color:#8b949e;letter-spacing:1px;text-transform:uppercase;margin:20px 0 6px;'>Climate Scenario</div>", unsafe_allow_html=True)
    scenario_label = st.radio(
        "Climate Scenario",
        ["SSP2-4.5 — Moderate Action", "SSP5-8.5 — Worst Case"],
        label_visibility="collapsed"
    )
    scenario_param   = "SSP2_4.5" if "SSP2" in scenario_label else "SSP5_8.5"
    scenario_display = "SSP2-4.5" if "SSP2" in scenario_label else "SSP5-8.5"

    st.markdown("""
    <div style='margin-top:32px;padding:12px;background:#161b22;border:1px solid #21262d;border-radius:8px;'>
        <div style='font-size:11px;color:#8b949e;'>
            <b style='color:#c9d1d9;'>Data Sources</b><br><br>
            🛰️ DiCRA API (Tmax, NDVI, SM)<br>
            🌧️ CHIRPS (SPI-3 Drought)<br>
            📊 PMFBY CCE Records<br>
            🌍 CMIP6 ({scenario})<br>
            🌱 FAO Crop Calendars
        </div>
    </div>
    """.format(scenario=scenario_display), unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="dash-header">
    <div class="dash-title">
        Compound Stress Index
        <span class="badge">MADHYA PRADESH</span>
    </div>
    <div class="dash-subtitle">
        Data in Climate Resilient Agriculture &nbsp;·&nbsp; District Advisory &nbsp;·&nbsp;
        {district} &nbsp;·&nbsp; {crop_meta['emoji']} {crop} &nbsp;·&nbsp;
        {crop_meta['season']} {year} &nbsp;·&nbsp; {scenario_display}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Fetch current year data ─────────────────────────────────────────────────────
tier, csi_score, premium_adj = None, None, None
try:
    resp = requests.get(
        f"{API_URL}/insurance/premium_risk",
        params={"district": district, "year": year, "crop": crop, "scenario": scenario_param},
        timeout=5,
    )
    if resp.status_code == 200:
        d            = resp.json()
        tier         = d["csi_risk_tier"]
        csi_score    = d["probabilistic_score"]
        premium_adj  = d["recommended_premium_adjustment"]
except Exception:
    pass

# ── Fetch trajectory data ───────────────────────────────────────────────────────
trajectory_df = pd.DataFrame()
try:
    resp2 = requests.get(
        f"{API_URL}/dashboard/trajectory",
        params={"district": district, "crop": crop, "scenario": scenario_param},
        timeout=5,
    )
    if resp2.status_code == 200:
        trajectory_df = pd.DataFrame(resp2.json())
except Exception:
    pass

# ── Main content ────────────────────────────────────────────────────────────────
if tier is None:
    st.error("⚠️ Cannot connect to the FastAPI backend. Ensure it is running on port 8000.")
    st.code("cd CSI-DiCRA-Project && PYTHONPATH=. python -m uvicorn src.main:app --reload --port 8000")
    st.stop()

cfg = CROP_TIER_CONFIG[crop][tier]

# Row 1 — Metric Cards
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Risk Tier</div>
        <div class="metric-value" style="color:{cfg['color']};">{tier} / 5</div>
        <div class="metric-sub">{cfg['emoji']} {cfg['label']}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">CSI Score</div>
        <div class="metric-value">{csi_score:.2f}</div>
        <div class="metric-sub">0.0 = No stress &nbsp;·&nbsp; 1.0 = Max</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Crop · Season</div>
        <div class="metric-value" style="font-size:22px;padding-top:6px;">{crop_meta['emoji']} {crop}</div>
        <div class="metric-sub">{crop_meta['season']} &nbsp;·&nbsp; {crop_meta['window']}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Insurance Signal</div>
        <div class="metric-value" style="font-size:22px;padding-top:6px;">{premium_adj}</div>
        <div class="metric-sub">Actuarial Premium Adjustment</div>
    </div>""", unsafe_allow_html=True)

# Row 2 — Risk Banner
st.markdown("<div class='section-label'>Risk Assessment</div>", unsafe_allow_html=True)
st.markdown(f"""
<div class="risk-banner {cfg['css']}">
    <div class="risk-title">{cfg['emoji']} Tier {tier} — {cfg['label']}</div>
    <div class="risk-body">{cfg['headline']}</div>
</div>
""", unsafe_allow_html=True)

# Row 3 — Advisory actions + Trajectory chart
left, right = st.columns([1, 1.6], gap="large")

with left:
    st.markdown("<div class='section-label'>Field Advisory</div>", unsafe_allow_html=True)
    items_html = "".join(
        f"<div class='advisory-item'><span>{a}</span></div>"
        for a in cfg["actions"]
    )
    st.markdown(
        f"<div class='advisory-box'>"
        f"<div class='advisory-heading'>Recommended Actions — {crop} {year}</div>"
        f"{items_html}</div>",
        unsafe_allow_html=True,
    )

with right:
    st.markdown("<div class='section-label'>15-Year Risk Trajectory (2026 – 2041)</div>", unsafe_allow_html=True)

    if not trajectory_df.empty:
        tier_colors = {1: "#3fb950", 2: "#a8cc3a", 3: "#d29922", 4: "#f85149", 5: "#ff6b6b"}
        bar_colors  = [tier_colors.get(int(t), "#8b949e") for t in trajectory_df["risk_tier"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
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
            fig.add_trace(go.Scatter(
                x=selected_row["year"],
                y=selected_row["risk_tier"],
                mode="markers",
                marker=dict(size=14, color="#ffffff", line=dict(color=cfg["color"], width=3)),
                name=f"Selected ({year})",
                hovertemplate=f"<b>Selected Year: {year}</b><br>Tier: {tier}<extra></extra>",
            ))

        fig.update_layout(
            plot_bgcolor="#161b22", paper_bgcolor="#161b22",
            font=dict(family="Inter", color="#8b949e", size=12),
            xaxis=dict(
                title="Year", title_font=dict(color="#8b949e", size=11),
                tickfont=dict(color="#8b949e"), gridcolor="#21262d",
                tickmode="linear", dtick=2,
            ),
            yaxis=dict(
                title="Risk Tier", title_font=dict(color="#8b949e", size=11),
                tickfont=dict(color="#8b949e"), gridcolor="#21262d",
                range=[0, 5.5], tickvals=[1, 2, 3, 4, 5],
                ticktext=["1 — Normal", "2 — Mild", "3 — Moderate", "4 — Severe", "5 — Catastrophic"],
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=-0.15, x=0, font=dict(color="#8b949e")),
            height=320,
            bargap=0.25,
        )
        for t_val, t_color in tier_colors.items():
            fig.add_hline(y=t_val, line_dash="dot", line_color=t_color, opacity=0.2, line_width=1)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            f"No trajectory data available for {crop} in {district}. "
            "Run the CMIP6 forecasting pipeline first."
        )

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:40px;padding:16px 0;border-top:1px solid #21262d;
     display:flex;justify-content:space-between;align-items:center;'>
    <div style='font-size:11px;color:#484f58;'>
        CSI-DiCRA Actuarial Climate Risk Engine &nbsp;·&nbsp; Data: DiCRA · CHIRPS · PMFBY · CMIP6
    </div>
    <div style='font-size:11px;color:#484f58;'>
        NABARD &nbsp;·&nbsp; UNDP &nbsp;·&nbsp; CSI-DiCRA-Project-V2
    </div>
</div>
""", unsafe_allow_html=True)