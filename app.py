import streamlit as st
import pandas as pd
import joblib

# 1. Modeli, Scaler'ı, Encoder'ları ve Sütun Sırasını Yükle
@st.cache_resource
def load_assets():
    model = joblib.load('f1_pit_model_streamlit.joblib')
    scaler = joblib.load('f1_pit_scaler_2.joblib')
    ord_enc = joblib.load('ordinal_enc.joblib')
    te_race = joblib.load('target_race.joblib')
    te_driver = joblib.load('target_driver.joblib')
    feature_cols = joblib.load('feature_columns.joblib') 
    return model, scaler, ord_enc, te_race, te_driver, feature_cols

model, scaler, ord_enc, te_race, te_driver, feature_cols = load_assets()

st.title("F1 Pit Stop Predictor")

# ... (race_options listesi aynı kalabilir) ...
race_options = ["Canadian Grand Prix", "Dutch Grand Prix", "Austrian Grand Prix", "Pre-Season Testing", 
                "Azerbaijan Grand Prix", "Saudi Arabian Grand Prix", "Belgian Grand Prix", 
                "United States Grand Prix", "Italian Grand Prix", "Hungarian Grand Prix", 
                "Japanese Grand Prix", "São Paulo Grand Prix", "Bahrain Grand Prix", 
                "Las Vegas Grand Prix", "Monaco Grand Prix", "British Grand Prix", 
                "Australian Grand Prix", "Spanish Grand Prix", "Miami Grand Prix", 
                "French Grand Prix", "Abu Dhabi Grand Prix", "Chinese Grand Prix", 
                "Mexico City Grand Prix", "Emilia Romagna Grand Prix", "Singapore Grand Prix", "Qatar Grand Prix"]

with st.form("pit_form"):
    col1, col2 = st.columns(2)
    with col1:
        driver = st.text_input("Driver Code (e.g., VER)")
        compound = st.selectbox("Compound", ['Soft', 'Medium', 'Hard', 'Intermediate', 'Wet'])
        race = st.selectbox("Race Name", options=race_options)
        year = st.number_input("Year", 2020, 2026, 2025)
        lap_number = st.number_input("Lap Number", 1, 80, 1)
        stint = st.number_input("Stint", 1, 10, 1)
        tyre_life = st.number_input("Tyre Life", 0, 100, 10)
    with col2:
        position = st.number_input("Position", 1, 20, 1)
        lap_time = st.number_input("Lap Time (s)", 0.0, 3000.0, 90.0)
        lap_delta = st.number_input("LapTime Delta", -3000.0, 3000.0, 0.0)
        cumulative_deg = st.number_input("Cumulative Degradation", -2000.0, 2000.0, 0.0)
        race_progress = st.number_input("Race Progress", 0.0, 1.0, 0.0, step=0.01)
        position_change = st.number_input("Position Change", -20.0, 20.0, 0.0)
        
    submitted = st.form_submit_button("Predict")

if submitted:
    data = {
        'Driver': driver, 'Compound': compound, 'Race': race, 'Year': year,
        'LapNumber': lap_number, 'Stint': stint, 'TyreLife': tyre_life,
        'Position': position, 'LapTime (s)': lap_time, 'LapTime_Delta': lap_delta,
        'Cumulative_Degradation': cumulative_deg, 'RaceProgress': race_progress,
        'Position_Change': position_change,
        'Tyre_Usage_Rate': tyre_life / 30.0,
        'Rolling_LapTime_Std': 0.0,
        'Position_Momentum': position_change,
        'Stint_Lap_Count': lap_number,
        'Degradation_Velocity': cumulative_deg / (lap_number + 1)
    }
    
    input_df = pd.DataFrame([data])
    
    # HATA AYIKLAMA: Eksik sütunları göster
    missing_cols = [c for c in feature_cols if c not in input_df.columns]
    if missing_cols:
        st.error(f"Eksik Sütunlar bulundu: {missing_cols}")
        st.write("Mevcut Sütunlar:", input_df.columns.tolist())
    else:
        # 2. DataFrame'i sırala
        input_df = input_df[feature_cols] 
        
        # 3. ENCODING
        input_df['Compound'] = ord_enc.transform(input_df[['Compound']])
        input_df['Race'] = te_race.transform(input_df[['Race']])
        input_df['Driver'] = te_driver.transform(input_df[['Driver']])
        
        # 4. ÖLÇEKLEME VE TAHMİN
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        
        result = "Pit Stop" if prediction[0] == 1 else "No Pit Stop"
        st.success(f"Prediction: {result}")
