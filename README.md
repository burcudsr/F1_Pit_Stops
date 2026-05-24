# 🏎️ F1 Pit Stop Prediction

This project focuses on predicting the timing of Formula 1 pit stops by analyzing race telemetry and historical event data, developed for the **Kaggle Playground Series - Season 5, Episode 10**.

### 🚀 Live Demo
Explore the interactive prediction model: https://f1pitstops.streamlit.app

### 🛠️ Advanced Feature Engineering
A custom pipeline was engineered to extract deep insights from raw race telemetry:
* **Tyre Dynamics & Volatility**: Metrics such as `Tyre_Usage_Rate`, `Rolling_LapTime_Std` (tyre cliff indicator), and `Position_Momentum` were calculated to track tyre performance and race trends.
* **Robust Categorical Encoding**: `OrdinalEncoder` was utilized for the `Compound` feature with `handle_unknown='use_encoded_value'` and `unknown_value=-1`. This ensures model stability and prevents runtime errors if previously unseen tyre compounds are encountered during inference.
* **Target Encoding & Scaling**: `TargetEncoder` was applied to `Race` and `Driver` features to capture categorical significance relative to the target (`PitNextLap`), followed by `StandardScaler` for numerical normalization.

### 🏆 Model Selection & Benchmarking
A wide range of machine learning algorithms was evaluated to determine the most effective strategy. The **ExtraTreesClassifier** outperformed the other models, emerging as the optimal solution:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **ExtraTreesClassifier** | **0.8918** | **0.7556** | **0.6742** | **0.7126** |

### 🧠 Model Performance Diagnostics
* **Feature Importance**: Analysis showed that tyre-related metrics and stint duration are the primary drivers of pit stop decisions.
