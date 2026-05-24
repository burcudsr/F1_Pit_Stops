if submitted:
    # 1. Prepare Data Dictionary
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
        'Degradation_Velocity': cumulative_deg / (lap_number + 1),
        'PitStop': 0 
    }
    
    input_df = pd.DataFrame([data])
    
    # Check for missing columns
    missing_cols = [c for c in feature_cols if c not in input_df.columns]
    
    if missing_cols:
        st.error(f"Missing columns required by the model: {missing_cols}")
    else:
        # Reorder columns to match the training set
        input_df = input_df[feature_cols] 
        
        # Apply Encoding
        input_df['Compound'] = ord_enc.transform(input_df[['Compound']])
        input_df['Race'] = te_race.transform(input_df[['Race']])
        input_df['Driver'] = te_driver.transform(input_df[['Driver']])
        
        # Scaling and Prediction
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        
        # Display Result
        # Prediction bir sayı (regresyon) döndürdüğü için yuvarlayarak yazdırıyoruz
        next_pit_lap = int(round(prediction[0]))
        st.success(f"Estimated Next Pit Stop Lap: {next_pit_lap}")
