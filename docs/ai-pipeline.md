# BridgeTwinAI - AI/ML Pipeline

## 1. Overview
The AI Pipeline processes simulated sensor data to estimate structural health, detect anomalies, explain decisions, and recommend maintenance actions. 

## 2. Physics-Informed Structural Health Estimation Methodology
Pure data-driven models often fail to respect the laws of physics. We will employ **Physics-Informed Neural Networks (PINNs)** or hybrid models.
- **Physical Constraints:** The neural network's loss function will incorporate physical differential equations (e.g., Euler-Bernoulli beam theory for deck deflection).
- **Inputs:** Simulated strain, vibration (acceleration), and temperature.
- **Outputs:** Estimated structural degradation (stiffness reduction), health score (0-100%).
- **Process:**
  1. High-frequency raw data is downsampled/aggregated (feature extraction).
  2. Features (e.g., resonant frequencies from Fast Fourier Transform of acceleration data) are fed into the network.
  3. The PINN outputs the estimated continuous health state of specific bridge segments.

## 3. Tabular Data Classification (Historical Records)
For static and historical data (e.g., `bridge_data.csv`), the pipeline utilizes **XGBoost** for binary classification of the `Bridge_Condition`.
- **Why XGBoost:** Highly effective on small, tabular datasets containing mixed numerical (Age, Traffic) and categorical (Material, Maintenance Level) features without requiring deep neural networks.
- **Preprocessing:** Categorical variables are One-Hot or Target Encoded.
- **Integration:** Seamlessly integrates with SHAP for robust feature importance extraction.

## 4. Time-Series Analysis (Digital Twin Telemetry)
For sequential and continuous data streaming from the digital twin (e.g., `bridge_digital_twin_dataset.csv`), the pipeline utilizes **Long Short-Term Memory (LSTM)** networks.
- **Why LSTM:** LSTMs are designed to handle time-series data and sequence prediction, making them perfect for identifying trends, capturing long-term dependencies in sensor readings (e.g., gradual material fatigue), and forecasting future structural states.
- **Integration:** The LSTM predictions run in parallel with the PINN models to provide purely data-driven temporal forecasts of sensor behavior and anomalies over time.

## 5. Explainability Architecture
AI decisions must be transparent for civil engineers.
- **Technology:** SHAP (SHapley Additive exPlanations) or LIME.
- **Implementation:** For every anomaly detected or low health score, the explainability module calculates the contribution of each sensor's features.
- **Visualization:** The backend sends SHAP values to the frontend, which highlights the specific 3D bridge segments/sensors in red, allowing the user to say, "The AI thinks the bridge is failing because Sensor A (Strain Gauge on Pier 3) is showing unexpected values."

## 6. Maintenance Recommendation Engine
Maps the AI's health assessments and anomaly detections to concrete maintenance actions.
- **Rule-Based Engine (V1):** Simple thresholds. E.g., `IF health_score < 0.6 AND critical_sensor = 'Deck_Midspan' THEN action = 'Inspect midspan for fatigue cracks' WITH priority = 'High'`.
- **Predictive Optimization (V2):** Utilizing historical maintenance data (if available in `bridge_data.csv`) to predict the most cost-effective maintenance intervention.
- **Output:** Recommendations are stored in the database and surfaced as actionable alerts on the frontend dashboard.
