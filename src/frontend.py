import streamlit as st
import pandas as pd
import requests

# Assuming FastAPI is running locally
API_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="CSI-DiCRA Dashboard", layout="wide")

st.title("🌾 CSI-DiCRA Compound Stress Index")

# Dual Persona Toggle
user_type = st.sidebar.radio("Select User Persona:", ["👨‍🌾 Farmer (Advisory)", "🏢 Insurer / Govt (Policy)"])

# District Selection
mp_districts = [
    "Agar Malwa", "Alirajpur", "Anuppur", "Ashoknagar", "Balaghat", "Barwani", "Betul", "Bhind", 
    "Bhopal", "Burhanpur", "Chhatarpur", "Chhindwara", "Damoh", "Datia", "Dewas", "Dhar", 
    "Dindori", "Guna", "Gwalior", "Harda", "Hoshangabad", "Indore", "Jabalpur", "Jhabua", 
    "Katni", "Khandwa", "Khargone", "Mandla", "Mandsaur", "Morena", "Narsinghpur", "Neemuch", 
    "Niwari", "Panna", "Raisen", "Rajgarh", "Ratlam", "Rewa", "Sagar", "Satna", "Sehore", 
    "Seoni", "Shahdol", "Shajapur", "Sheopur", "Shivpuri", "Sidhi", "Singrauli", "Tikamgarh", 
    "Ujjain", "Umaria", "Vidisha"
]
district = st.sidebar.selectbox("Select District:", mp_districts)

if user_type == "👨‍🌾 Farmer (Advisory)":
    st.header(f"Localized Advisory for {district}")
    st.write("Welcome! This dashboard helps you make data-driven decisions for your upcoming crop cycles.")
    
    # Farmers usually care about the immediate next year (e.g., 2026) under realistic conditions
    try:
        response = requests.get(f"{API_URL}/insurance/premium_risk?district={district}&year=2026&scenario=RCP_8.5")
        if response.status_code == 200:
            data = response.json()
            tier = data["csi_risk_tier"]
            
            col1, col2 = st.columns(2)
            col1.metric("Upcoming Season Risk (2026)", f"Tier {tier} / 5")
            
            if tier >= 4:
                col2.error("⚠️ HIGH RISK of compound heat & drought during flowering. Consider drought-tolerant varieties or delaying sowing.")
            else:
                col2.success("✅ Moderate Risk. Standard sowing schedules apply.")
                
        st.info("🤖 **Coming Soon:** Ask our Regional Language AI Chatbot for specific crop advice based on these metrics!")
    except Exception as e:
        st.warning("Ensure the FastAPI backend is running on port 8000.")

else:
    # Insurer / Govt View
    st.header(f"Actuarial Risk Trajectory (2026 - 2041): {district}")
    
    # RCP Scenario Toggle (Crucial for not averaging them)
    scenario = st.radio("Select Climate Scenario:", ["RCP 4.5 (Moderate Action)", "RCP 8.5 (Business-As-Usual)"])
    scenario_param = "RCP_4.5" if "4.5" in scenario else "RCP_8.5"
    
    try:
        response = requests.get(f"{API_URL}/dashboard/trajectory?district={district}&scenario={scenario_param}")
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            
            if not df.empty:
                st.line_chart(df.set_index("year")["risk_tier"])
                
                st.subheader("Data Table & Premium Adjustments")
                st.dataframe(df)
            else:
                st.write("No projection data available. Run the CMIP6 pipeline.")
    except Exception as e:
        st.warning("Ensure the FastAPI backend is running on port 8000.")