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

### 3.1 SVM Comparison Model
A **Support Vector Machine** (`ai_pipeline/src/train_svm.py`) is trained on the same features and the same 80/20 stratified split as XGBoost, as a margin-based model to compare against.
- **Pipeline:** `StandardScaler -> SVC(kernel='rbf', class_weight='balanced', probability=True)`. Kernel, C and gamma are picked by a 5-fold stratified grid search (scoring = F1).
- **Threshold:** tuned on out-of-fold *training* probabilities only, so the test set is never used for tuning.
- **Explainability:** permutation importance and SHAP `KernelExplainer` (SVMs have no native feature importances).
- **Artefacts:** `ai_pipeline/models/svm/`. Report and plots are in `ai_pipeline/notebooks/svm/`.

### 3.2 Dual-Model Consensus Architecture
In production serving (`backend/app/services/ai_service.py`), both XGBoost and SVM are evaluated simultaneously:
- **Consensus States:**
  - *Unanimous At-Risk:* Both models independently classify the bridge as at-risk.
  - *Unanimous Healthy:* Both models agree the bridge is in sound condition.
  - *Divergent:* Models disagree; conservatively flagged for priority civil engineer review.
- **Ensemble Probability:** $\bar{P} = (P_{\text{XGB}} + P_{\text{SVM}}) / 2$.

## 4. Time-Series Analysis (Digital Twin Telemetry)
For sequential and continuous data streaming from the digital twin (e.g., `bridge_digital_twin_dataset.csv`), the pipeline utilizes **Long Short-Term Memory (LSTM)** networks.
- **Why LSTM:** LSTMs are designed to handle time-series data and sequence prediction, making them perfect for identifying trends, capturing long-term dependencies in sensor readings (e.g., gradual material fatigue), and forecasting future structural states.
- **Epistemic Uncertainty via Monte Carlo Dropout:** Dropout is enabled during inference across $N=40$ stochastic forward passes, generating empirical 90% confidence intervals ($\pm \sigma$) on the Structural Health Index (SHI).

## 5. Explainability Architecture
AI decisions must be transparent for civil engineers.
- **Technology:** TreeSHAP (for XGBoost) and Occlusion Channel Attribution (for LSTM).
- **Implementation:** For every assessment, the explainability module calculates the contribution of each feature and sensor channel.
- **Visualization:** The backend sends attribution values to the frontend, which highlights dominant degradation drivers on the dashboard.

## 6. Unified Risk Fusion Engine & Prioritization Matrix
Synthesizes all model outputs into a single transparent score (`backend/app/services/fusion_service.py`):
$$\text{Overall Risk} = 0.35 \cdot R_{\text{ML}} + 0.35 \cdot (1 - \text{SHI}_{\text{LSTM}}) + 0.30 \cdot R_{\text{Physics}}$$
- **Priority Thresholds:** Critical ($\ge 70\%$), High ($50\text{–}70\%$), Medium ($30\text{–}50\%$), Low ($< 30\%$).
- **Grounded Maintenance:** Maps priority to targeted inspection timeframes and actions.

## 7. Predictive Maintenance & Generative Synthesis
- **Gemini 2.5 Flash Integration:** Synthesizes the fused risk, dual-model consensus, and physics indicators into an executive engineering summary and maintenance plan adhering strictly to Pydantic schemas.
- **Deterministic Rule Fallback:** Fully operational when offline or when API keys are absent.
