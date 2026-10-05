import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def create_sequences(data, target, seq_length):
    """
    Convert 2D tabular data into 3D sequences for LSTM input.
    data: (num_samples, num_features)
    target: (num_samples,)
    seq_length: integer window size
    
    Returns:
    X: (num_samples - seq_length, seq_length, num_features)
    y: (num_samples - seq_length,)
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(target[i + seq_length])
    return np.array(xs), np.array(ys)

def load_and_preprocess_lstm_data(filepath, seq_length=60, scaler_path=None):
    """
    Load dataset, select features, scale, and generate sequences.
    If scaler_path is provided, it saves the fitted scaler for inference.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Sort by timestamp to ensure correct temporal order
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp').reset_index(drop=True)
        df.drop(columns=['Timestamp'], inplace=True)
    
    # Define the target
    target_col = 'Structural_Health_Index_SHI'
    
    # Drop rows where target is NaN
    df = df.dropna(subset=[target_col])
    
    # Define features to use (sensors, environment, load)
    feature_cols = [
        'Strain_microstrain', 'Deflection_mm', 'Vibration_ms2', 'Tilt_deg', 'Displacement_mm',
        'Temperature_C', 'Humidity_percent', 'Wind_Speed_ms', 'Water_Level_m', 'Seismic_Activity_ms2',
        'Vehicle_Load_tons', 'Traffic_Volume_vph', 'Dynamic_Load_Distribution_percent'
    ]
    
    # Ensure columns exist, fill missing with 0 or ffill
    existing_features = [col for col in feature_cols if col in df.columns]
    X_df = df[existing_features].copy()
    y_df = df[target_col].copy()
    
    # Forward fill then fill 0
    X_df = X_df.ffill().fillna(0)
    
    print(f"Using {len(existing_features)} features for LSTM.")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    
    if scaler_path:
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
        
    y = y_df.values
    
    # Create temporal sequences
    print(f"Creating sequences with window size = {seq_length}...")
    X_seq, y_seq = create_sequences(X_scaled, y, seq_length)
    
    print(f"Final sequence shapes: X={X_seq.shape}, y={y_seq.shape}")
    return X_seq, y_seq, existing_features
