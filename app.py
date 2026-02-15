import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AutoValuate Nigeria", 
    page_icon="🇳🇬", 
    layout="centered"
)

# --- HARDCODED CAR DATA (Expanded List) ---
# This ensures you have a DROPDOWN, not a text input.
car_models_dict = {
    "Ford": ["Fiesta", "Focus", "Kuga", "EcoSport", "C-MAX", "Ka+", "Mondeo", "B-MAX", "S-MAX", "Galaxy", "Edge", "Puma", "Mustang", "Ka", "Tourneo Custom", "Grand C-MAX", "Tourneo Connect", "Grand Tourneo Connect"],
    "Volkswagen": ["Golf", "Polo", "Tiguan", "Passat", "T-Roc", "Up", "Scirocco", "Touran", "T-Cross", "Touareg", "Golf SV", "Beetle", "Sharan", "Caddy Maxi Life", "Caravelle", "CC", "Arteon", "Caddy Life", "Amarok", "Tiguan Allspace", "Sharan", "Jetta", "Eos"],
    "Mercedes": [" C Class", " A Class", " E Class", " GLC Class", " GLA Class", " B Class", " CL Class", " GLE Class", " S Class", " SLK", " CLA Class", " V Class", " M Class", " CLS Class", " GL Class", " SL CLASS", " GLS Class", " GLB Class", " X-CLASS", " G Class", " CLC Class", " R Class"],
    "BMW": [" 3 Series", " 1 Series", " 2 Series", " 5 Series", " X1", " X3", " X5", " 4 Series", " X2", " X4", " 7 Series", " X6", " i3", " 8 Series", " Z4", " X7", " M4", " M5", " M3", " M2", " i8", " 6 Series"],
    "Audi": [" A3", " A1", " A4", " Q3", " Q5", " A5", " Q2", " A6", " Q7", " TT", " A7", " A8", " Q8", " RS3", " RS4", " RS5", " RS6", " R8", " SQ5", " SQ7", " S3", " S4", " S5", " TTS"],
    "Toyota": ["Yaris", "Aygo", "Auris", "C-HR", "RAV4", "Corolla", "Prius", "Avensis", "Verso", "Hilux", "GT86", "Land Cruiser", "Camry", "Supra", "PROACE VERSO", "IQ", "Urban Cruiser"],
    "Vauxhall": ["Corsa", "Astra", "Mokka", "Crossland X", "Grandland X", "Zafira", "Insignia", "Adam", "Viva", "Mokka X", "Meriva", "GTC", "Combo Life", "Vivaro", "Antara", "Vectra", "Zafira Tourer", "Agila"],
    "Skoda": ["Octavia", "Fabia", "Superb", "Yeti", "Kodiaq", "Karoq", "Citigo", "Rapid", "Kamiq", "Scala"],
    "Hyundai": ["I10", "I20", "I30", "Tucson", "Santa Fe", "Ioniq", "Kona", "IX35", "I40", "I800"],
    "Kia": ["Sportage", "Picanto", "Rio", "Ceed", "Optima", "Sorento", "Venga", "Soul", "Stonic", "Niro", "Carens"]
}

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        # We use joblib because that's how we saved the optimized model
        model = joblib.load('uk_car_model.pkl')
        return model
    except FileNotFoundError:
        return None

model = load_model()

# --- SIDEBAR ---
st.sidebar.header("⚙️ Settings")
exchange_rate = st.sidebar.number_input("Exchange Rate (£1 = ₦?)", value=2150, step=10)

st.title("🇳🇬 AutoValuate: Car Price Predictor")
st.write("Select your vehicle specifications below.")

if model is None:
    st.error("⚠️ Model file `uk_car_model.pkl` not found. Please make sure it's in the same folder.")
    st.stop()

# --- INPUT SECTION ---

# 1. Brand Selection
# Using sorted() makes it easier to find the brand
brand_list = sorted(list(car_models_dict.keys()))
brand = st.selectbox("Select Brand", brand_list)

# 2. Model Selection (Dynamic Dropdown)
# This box updates automatically when you change the Brand
models_for_brand = sorted(car_models_dict[brand])
car_model = st.selectbox("Select Model", models_for_brand)

st.divider()

# --- MAIN FORM FOR SPECS ---
with st.form("specs_form"):
    st.subheader("Vehicle Details")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        year = st.number_input("Year", 1995, 2025, 2019)
    with col2:
        transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Semi-Auto"])
    with col3:
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric", "Other"])

    col4, col5 = st.columns(2)
    with col4:
        mileage = st.number_input("Mileage (miles)", 0, 400000, 40000, step=1000)
    with col5:
        engine_size = st.number_input("Engine Size (L)", 0.0, 8.0, 1.5, step=0.1)

    # Hidden advanced options
    with st.expander("Advanced Options (Tax & MPG)"):
        st.info("Leave these as default if you are not sure.")
        c_adv1, c_adv2 = st.columns(2)
        with c_adv1:
            mpg = st.number_input("MPG", 10.0, 200.0, 55.0)
        with c_adv2:
            tax = st.number_input("Road Tax (£)", 0, 1000, 145)

    submitted = st.form_submit_button("💰 Calculate Value", type="primary", use_container_width=True)

# --- PREDICTION LOGIC ---
if submitted:
    # Ensure consistency (some models in dataset have leading space)
    formatted_model = car_model 
    
    input_data = pd.DataFrame({
        'brand': [brand],
        'model': [formatted_model], 
        'year': [year],
        'transmission': [transmission],
        'mileage': [mileage],
        'fuelType': [fuel_type],
        'tax': [tax],
        'mpg': [mpg],
        'engineSize': [engine_size]
    })

    try:
        pred_gbp = model.predict(input_data)[0]
        pred_ngn = pred_gbp * exchange_rate

        st.success("Valuation Complete")
        
        c1, c2 = st.columns(2)
        c1.metric("Estimated Price (Naira)", f"₦{pred_ngn:,.0f}")
        c2.metric("UK Market Value", f"£{pred_gbp:,.0f}")
        
        st.caption("Note: Values are estimates based on UK historical data converted to NGN.")

    except Exception as e:
        st.error(f"Error: {e}")