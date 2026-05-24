import streamlit as st
import pandas as pd
import joblib

# 1. Modeli, Scaler'ı ve tüm Encoder'ları Yükle
@st.cache_resource
def load_assets():
    model = joblib.load('model_assets/f1_pit_model.joblib')
    scaler = joblib.load('model_assets/f1_pit_scaler.joblib')
    ord_enc = joblib.load('model_assets/ordinal_enc.joblib')
    te_race = joblib.load('model_assets/target_race.joblib')
    te_driver = joblib.load('model_assets/target_driver.joblib')
    return model, scaler, ord_enc, te_race, te_driver
    
model, scaler, ord_enc, te_race, te_driver = load_assets()

st.title("F1 Pit Stop Predictor")

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
    # Ham veriyi DataFrame'e al
    input_df = pd.DataFrame([{
        'Driver': driver, 'Compound': compound, 'Race': race, 'Year': year,
        'LapNumber': lap_number, 'Stint': stint, 'TyreLife': tyre_life,
        'Position': position, 'LapTime (s)': lap_time, 'LapTime_Delta': lap_delta,
        'Cumulative_Degradation': cumulative_deg, 'RaceProgress': race_progress,
        'Position_Change': position_change
    }])

    # 1. FE ADIMLARI (Eğitimdeki mantıkla aynı)
    input_df['Tyre_Usage_Rate'] = input_df['TyreLife'] / 30.0 
    input_df['Rolling_LapTime_Std'] = 0.0 # Tahmin anında anlık veri olduğu için
    input_df['Position_Momentum'] = input_df['Position_Change'] 
    input_df['Stint_Lap_Count'] = input_df['LapNumber'] 
    input_df['Degradation_Velocity'] = input_df['Cumulative_Degradation'] / (input_df['LapNumber'] + 1)
    
    # 2. ENCODING ADIMLARI
    input_df['Compound'] = ord_enc.transform(input_df[['Compound']])
    input_df['Race'] = te_race.transform(input_df[['Race']])
    input_df['Driver'] = te_driver.transform(input_df[['Driver']])
    
    # 3. ÖLÇEKLEME VE TAHMİN (Sütun sırasının eğitimdeki ile aynı olduğundan emin ol)
    # Eğitimde id, PitStop gibi sütunlar varsa buraya eklemelisin
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    
    result = "Pit Stop" if prediction[0] == 1 else "No Pit Stop"
    st.success(f"Prediction: {result}")
