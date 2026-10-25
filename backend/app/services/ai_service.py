import os
import json
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# --- Model Paths ---
def _resolve_models_dir():
    candidates = [
        os.getenv('AI_MODELS_DIR'),
        '/ai_pipeline/models',
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'ai_pipeline', 'models'),
        os.path.join(os.getcwd(), 'ai_pipeline', 'models'),
        os.path.join(os.getcwd(), '..', 'ai_pipeline', 'models'),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return '/ai_pipeline/models'

AI_MODELS_DIR = _resolve_models_dir()

# XGBoost paths
XGB_MODEL_PATH = os.path.join(AI_MODELS_DIR, 'xgboost', 'condition_xgboost.joblib')
XGB_FEATURES_PATH = os.path.join(AI_MODELS_DIR, 'xgboost', 'model_features.joblib')
XGB_THRESHOLD_PATH = os.path.join(AI_MODELS_DIR, 'xgboost', 'optimal_threshold.json')

# SVM paths
SVM_MODEL_PATH = os.path.join(AI_MODELS_DIR, 'svm', 'svm_condition_model.joblib')
SVM_FEATURES_PATH = os.path.join(AI_MODELS_DIR, 'svm', 'svm_features.json')
SVM_THRESHOLD_PATH = os.path.join(AI_MODELS_DIR, 'svm', 'svm_threshold.json')
SVM_BEST_PARAMS_PATH = os.path.join(AI_MODELS_DIR, 'svm', 'svm_best_params.json')

# LSTM paths
LSTM_MODEL_PATH = os.path.join(AI_MODELS_DIR, 'lstm', 'shi_lstm_model.pth')
LSTM_SCALER_PATH = os.path.join(AI_MODELS_DIR, 'lstm', 'lstm_scaler.joblib')
LSTM_FEATURES_PATH = os.path.join(AI_MODELS_DIR, 'lstm', 'lstm_features.json')

# --- PyTorch LSTM Definition ---
class BridgeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, output_dim=1, dropout=0.2):
        super(BridgeLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_dim)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# --- Service Class ---
class AIService:
    def __init__(self):
        self.xgb_model = None
        self.xgb_features = None
        self.xgb_threshold = 0.5

        self.svm_model = None
        self.svm_features = None
        self.svm_threshold = 0.25
        self.svm_best_params = {}
        
        self.lstm_model = None
        self.lstm_scaler = None
        self.lstm_features = None
        self.lstm_seq_length = 60
        
        self._load_models()
        
    def _load_models(self):
        print("Loading AI Models...")
        # Load XGBoost
        if os.path.exists(XGB_MODEL_PATH):
            self.xgb_model = joblib.load(XGB_MODEL_PATH)
            self.xgb_features = joblib.load(XGB_FEATURES_PATH)
            with open(XGB_THRESHOLD_PATH, 'r') as f:
                self.xgb_threshold = json.load(f).get('optimal_threshold', 0.5)
            print("XGBoost loaded.")
        else:
            print("Warning: XGBoost model not found.")

        # Load SVM
        if os.path.exists(SVM_MODEL_PATH):
            self.svm_model = joblib.load(SVM_MODEL_PATH)
            if os.path.exists(SVM_FEATURES_PATH):
                with open(SVM_FEATURES_PATH, 'r') as f:
                    self.svm_features = json.load(f)
            if os.path.exists(SVM_THRESHOLD_PATH):
                with open(SVM_THRESHOLD_PATH, 'r') as f:
                    self.svm_threshold = json.load(f).get('optimal_threshold', 0.25)
            if os.path.exists(SVM_BEST_PARAMS_PATH):
                with open(SVM_BEST_PARAMS_PATH, 'r') as f:
                    self.svm_best_params = json.load(f)
            print("SVM loaded.")
        else:
            print("Warning: SVM model not found.")
            
        # Load LSTM
        if os.path.exists(LSTM_MODEL_PATH):
            with open(LSTM_FEATURES_PATH, 'r') as f:
                lstm_meta = json.load(f)
                self.lstm_features = lstm_meta['features']
                self.lstm_seq_length = lstm_meta['seq_length']
                
            self.lstm_scaler = joblib.load(LSTM_SCALER_PATH)
            
            input_dim = len(self.lstm_features)
            self.lstm_model = BridgeLSTM(input_dim=input_dim)
            self.lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=torch.device('cpu')))
            self.lstm_model.eval()
            print("LSTM loaded.")
        else:
            print("Warning: LSTM model not found.")

    def predict_shi(self, time_series_data: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Predict Structural Health Index (SHI) using the PyTorch LSTM.
        Expects exactly `self.lstm_seq_length` historical data points (e.g., 60 mins).
        """
        if not self.lstm_model:
            return {"error": "LSTM model not loaded"}
            
        if len(time_series_data) < self.lstm_seq_length:
            return {"error": f"Requires exactly {self.lstm_seq_length} historical data points. Provided: {len(time_series_data)}"}
            
        df = pd.DataFrame(time_series_data)
        
        # Ensure all required features exist
        for col in self.lstm_features:
            if col not in df.columns:
                df[col] = 0.0
                
        # Select and scale features
        X_raw = df[self.lstm_features].values
        X_scaled = self.lstm_scaler.transform(X_raw)
        
        # Take the most recent window
        X_window = X_scaled[-self.lstm_seq_length:]
        X_tensor = torch.tensor(X_window, dtype=torch.float32).unsqueeze(0) # (1, seq_length, features)
        
        # Inference
        with torch.no_grad():
            shi_pred = self.lstm_model(X_tensor).item()
            
        return {
            "predicted_shi": min(max(shi_pred, 0.0), 1.0) # Clamp between 0 and 1
        }

    def _build_xgb_frame(self, bridge_data: Dict[str, Any]) -> pd.DataFrame:
        """Feature engineering matching the XGBoost training pipeline."""
        df = pd.DataFrame([bridge_data])
        
        # Ordinal encoding for maintenance
        maintenance_order = {'No-Maintainance': 0, 'Annual': 1, 'Bi-Annual': 2}
        m_val = bridge_data.get('Maintenance_Level', 'No-Maintainance')
        df['maintenance_encoded'] = maintenance_order.get(m_val, 0)
        
        # One-hot encoding for material (fallback if missing)
        df['Material_Type_Steel'] = 1 if bridge_data.get('Material_Type') == 'Steel' else 0
        
        # Interaction & Derived features
        age = float(bridge_data.get('Age_of_Bridge', 0))
        traffic = float(bridge_data.get('Traffic_Volume', 0))
        m_enc = df['maintenance_encoded'].iloc[0]
        
        df['age_factor'] = age
        df['traffic_load_factor'] = traffic
        df['age_squared'] = age ** 2
        df['age_traffic_interaction'] = age * traffic
        df['age_maintenance_interaction'] = age * m_enc
        df['traffic_maintenance_interaction'] = traffic * m_enc
        
        # Age bins
        df['age_bin'] = pd.cut([age], bins=[0, 25, 50, 75, 100, float('inf')], labels=[0, 1, 2, 3, 4])[0]
        
        df['log_traffic'] = np.log1p(traffic)
        
        # Align features exactly with what the model expects
        X = pd.DataFrame(columns=self.xgb_features)
        for col in self.xgb_features:
            if col in df.columns:
                X[col] = df[col]
            else:
                X[col] = 0
                
        # Make sure types are float
        return X.astype(float)

    def predict_risk(self, bridge_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict binary risk condition using XGBoost.
        Expects: Age_of_Bridge, Traffic_Volume, Material_Type, Maintenance_Level
        """
        if not self.xgb_model:
            return {"error": "XGBoost model not loaded"}

        X = self._build_xgb_frame(bridge_data)
        
        # Inference
        y_prob = self.xgb_model.predict_proba(X)[0][1]
        is_at_risk = int(y_prob >= self.xgb_threshold)
        
        return {
            "is_at_risk": bool(is_at_risk),
            "risk_probability": float(y_prob),
            "threshold_used": float(self.xgb_threshold)
        }

    def predict_svm_risk(self, bridge_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict bridge risk condition using the Support Vector Machine (SVM) pipeline.
        Utilizes native decision boundary (decision_function > 0) with balanced weights
        and Platt-scaled calibrated probabilities.
        """
        if not self.svm_model:
            return {"error": "SVM model not loaded"}

        X = self._build_xgb_frame(bridge_data)
        if self.svm_features:
            X = pd.DataFrame(X[self.svm_features].values, columns=self.svm_features, dtype=float)

        y_prob = float(self.svm_model.predict_proba(X)[0][1])
        decision_val = float(self.svm_model.decision_function(X)[0])
        # Native decision boundary with class_weight='balanced' achieves 0.95 recall
        is_at_risk_native = bool(decision_val > 0)
        is_at_risk_prob = bool(y_prob >= self.svm_threshold)

        return {
            "is_at_risk": is_at_risk_native,
            "risk_probability": round(y_prob, 4),
            "decision_function": round(decision_val, 4),
            "is_at_risk_prob_threshold": is_at_risk_prob,
            "threshold_used": float(self.svm_threshold),
            "model_type": "Support Vector Classifier (RBF kernel, balanced weights)"
        }

    def predict_dual_risk(self, bridge_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate risk using both XGBoost and SVM, compute consensus, ensemble probability,
        and agreement confidence score.
        """
        xgb_res = self.predict_risk(bridge_data)
        svm_res = self.predict_svm_risk(bridge_data)

        if "error" in xgb_res and "error" in svm_res:
            return {"error": "Both risk models failed to execute"}
        elif "error" in svm_res:
            return {
                "xgb": xgb_res,
                "svm": None,
                "consensus": "at_risk" if xgb_res.get("is_at_risk") else "healthy",
                "consensus_label": "XGBoost Solo (SVM unavailable)",
                "models_agree": True,
                "ensemble_probability": xgb_res.get("risk_probability", 0.0),
                "is_at_risk": bool(xgb_res.get("is_at_risk")),
                "risk_probability": xgb_res.get("risk_probability", 0.0),
                "confidence_score": 0.80,
                "threshold_used": xgb_res.get("threshold_used", 0.5)
            }
        elif "error" in xgb_res:
            return {
                "xgb": None,
                "svm": svm_res,
                "consensus": "at_risk" if svm_res.get("is_at_risk") else "healthy",
                "consensus_label": "SVM Solo (XGBoost unavailable)",
                "models_agree": True,
                "ensemble_probability": svm_res.get("risk_probability", 0.0),
                "is_at_risk": bool(svm_res.get("is_at_risk")),
                "risk_probability": svm_res.get("risk_probability", 0.0),
                "confidence_score": 0.80,
                "threshold_used": svm_res.get("threshold_used", 0.25)
            }

        p_xgb = float(xgb_res["risk_probability"])
        p_svm = float(svm_res["risk_probability"])
        ensemble_p = round((p_xgb + p_svm) / 2.0, 4)

        flag_xgb = bool(xgb_res["is_at_risk"])
        flag_svm = bool(svm_res["is_at_risk"])
        models_agree = (flag_xgb == flag_svm)

        if flag_xgb and flag_svm:
            consensus = "unanimous_at_risk"
            consensus_label = "Unanimous At-Risk"
            is_at_risk = True
            confidence = 0.92
        elif not flag_xgb and not flag_svm:
            consensus = "unanimous_healthy"
            consensus_label = "Unanimous Healthy"
            is_at_risk = False
            confidence = 0.90
        else:
            consensus = "divergent"
            consensus_label = "Divergent (Review Recommended)"
            # Safety-first policy for critical infrastructure: flag if either model detects risk
            is_at_risk = True
            confidence = 0.65

        return {
            "xgb": xgb_res,
            "svm": svm_res,
            "consensus": consensus,
            "consensus_label": consensus_label,
            "models_agree": models_agree,
            "ensemble_probability": ensemble_p,
            "is_at_risk": is_at_risk,
            "risk_probability": ensemble_p,
            "confidence_score": confidence,
            "threshold_used": xgb_res.get("threshold_used", 0.5)
        }

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    def explain_risk(self, bridge_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exact TreeSHAP contributions for the XGBoost classifier.

        Uses XGBoost's native ``pred_contribs`` (the TreeSHAP algorithm), so no
        extra dependency is needed. Values are in log-odds space and satisfy
        ``base_value + sum(contributions) == logit(probability)``.
        """
        import xgboost as xgb

        if not self.xgb_model:
            return {"error": "XGBoost model not loaded"}
        X = self._build_xgb_frame(bridge_data)
        contribs = self.xgb_model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0]
        feature_contribs = {f: float(v) for f, v in zip(self.xgb_features, contribs[:-1])}
        return {
            "base_value_logodds": float(contribs[-1]),
            "contributions": feature_contribs,
            "feature_values": {f: float(X[f].iloc[0]) for f in self.xgb_features},
        }

    def _lstm_tensor(self, time_series_data: List[Dict[str, float]]) -> torch.Tensor:
        df = pd.DataFrame(time_series_data)
        for col in self.lstm_features:
            if col not in df.columns:
                df[col] = 0.0
        X_scaled = self.lstm_scaler.transform(df[self.lstm_features].values)
        X_window = X_scaled[-self.lstm_seq_length:]
        return torch.tensor(X_window, dtype=torch.float32).unsqueeze(0)

    def predict_shi_with_uncertainty(
        self, time_series_data: List[Dict[str, float]], n_samples: int = 40
    ) -> Dict[str, Any]:
        """
        Deterministic SHI plus Monte-Carlo-dropout uncertainty.

        Dropout is enabled at inference for ``n_samples`` stochastic passes; the
        spread approximates epistemic uncertainty of the LSTM.
        """
        if not self.lstm_model:
            return {"error": "LSTM model not loaded"}
        if len(time_series_data) < self.lstm_seq_length:
            return {"error": f"Requires {self.lstm_seq_length} data points"}

        x = self._lstm_tensor(time_series_data)
        with torch.no_grad():
            point = float(self.lstm_model(x).item())
            torch.manual_seed(42)  # reproducible uncertainty bands
            self.lstm_model.train()
            samples = np.array([self.lstm_model(x).item() for _ in range(n_samples)])
            self.lstm_model.eval()

        clamp = lambda v: float(min(max(v, 0.0), 1.0))  # noqa: E731
        return {
            "predicted_shi": clamp(point),
            "mc_mean": clamp(samples.mean()),
            "mc_std": float(samples.std()),
            "interval_low": clamp(np.percentile(samples, 5)),
            "interval_high": clamp(np.percentile(samples, 95)),
            "n_samples": n_samples,
        }

    def lstm_channel_attribution(self, time_series_data: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Occlusion attribution per sensor channel.

        Each channel is replaced by its training mean (0 in standardised space)
        across the whole window; the change in predicted SHI is that channel's
        contribution. Negative values mean the channel is *lowering* health.
        """
        if not self.lstm_model:
            return {}
        x = self._lstm_tensor(time_series_data)
        with torch.no_grad():
            base = float(self.lstm_model(x).item())
            out: Dict[str, float] = {}
            for i, name in enumerate(self.lstm_features):
                occluded = x.clone()
                occluded[:, :, i] = 0.0
                out[name] = base - float(self.lstm_model(occluded).item())
        return out

    @property
    def lstm_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """Training mean/std per LSTM feature (from the fitted StandardScaler)."""
        return {
            "mean": dict(zip(self.lstm_features, map(float, self.lstm_scaler.mean_))),
            "std": dict(zip(self.lstm_features, map(float, self.lstm_scaler.scale_))),
        }

# Singleton instance
ai_service = AIService()
