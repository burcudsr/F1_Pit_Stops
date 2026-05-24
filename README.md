# 🏎️ F1 Pit Stop Prediction

This project focuses on predicting the timing of Formula 1 pit stops by analyzing race telemetry and historical event data, developed for the **Kaggle Playground Series - Season 5, Episode 10**.

### 🚀 Live Demo
Explore the interactive prediction model: https://huggingface.co/spaces/bdaser/F1_PitStops

### 🛠️ Advanced Feature Engineering
We engineered a custom pipeline to extract deep insights from raw race telemetry:
* **Tyre Dynamics & Volatility**: Calculated `Tyre_Usage_Rate`, `Rolling_LapTime_Std` (tyre cliff indicator), and `Position_Momentum` to track tyre performance and race trends.
* **Robust Categorical Encoding**: 
    * We utilized `OrdinalEncoder` for the `Compound` feature with `handle_unknown='use_encoded_value'` and `unknown_value=-1`. This ensures the model remains stable and avoids runtime errors if it encounters previously unseen tyre compounds during inference.
* **Target Encoding & Scaling**: Applied `TargetEncoder` to `Race` and `Driver` features to capture categorical significance relative to the target (`PitNextLap`), followed by `StandardScaler` for numerical normalization.

### 🧠 Model Methodology & Training Success
The model employs a deep learning architecture optimized for sequence-based race telemetry:
* **Feature Importance**: Analysis confirms that tyre-related metrics and stint duration are the primary drivers of pit stop decisions.
* **Confusion Matrix**: Demonstrates a high concentration of correct predictions along the diagonal, confirming the model's reliability in distinguishing between pit and non-pit laps.
* **Convergence**: Training logs show a smooth decline in `logloss` toward ~0.05. The validation loss mirrors the training loss perfectly, confirming excellent generalization and a robust lack of overfitting.




### 📈 Analysis Summary
The synergy between our specialized feature engineering—specifically the robust handling of categorical unknowns—and our deep learning architecture allows the model to predict pit stops with high precision. By mapping race-day metrics to tactical outcomes, the model serves as a dependable tool for F1 pit stop strategy analysis.
