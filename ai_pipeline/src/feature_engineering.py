import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def perform_feature_engineering(df):
    """
    Perform feature engineering for Bridge Condition prediction.
    Dataset: bridge_data.csv
    Columns: Age_of_Bridge, Traffic_Volume, Material_Type, Maintenance_Level, Bridge_Condition
    
    Improvements applied:
    1. Interaction features (Age x Traffic, Age x Maintenance)
    2. Binned age categories for non-linear age effects
    3. Label encoding for ordinal Maintenance_Level
    4. One-hot encoding for nominal Material_Type
    """
    df = df.copy()
    target = 'Bridge_Condition'
    df = df.dropna(subset=[target])

    # --- Ordinal Encoding for Maintenance_Level ---
    # Maintenance has a natural order: No-Maintainance < Annual < Bi-Annual
    # (Bi-Annual = more frequent = better maintained)
    maintenance_order = {
        'No-Maintainance': 0,
        'Annual': 1,
        'Bi-Annual': 2
    }
    df['maintenance_encoded'] = df['Maintenance_Level'].map(maintenance_order)
    # Fallback for any unseen categories
    df['maintenance_encoded'] = df['maintenance_encoded'].fillna(0).astype(int)

    # --- One-Hot Encoding for Material_Type (nominal) ---
    df = pd.get_dummies(df, columns=['Material_Type'], drop_first=True)

    # --- Engineered Features ---

    # Feature 1: Age Factor (keep raw)
    df['age_factor'] = df['Age_of_Bridge']

    # Feature 2: Traffic/Load Factor (keep raw)
    df['traffic_load_factor'] = df['Traffic_Volume']

    # Feature 3: Age squared (captures non-linear aging/degradation)
    df['age_squared'] = df['Age_of_Bridge'] ** 2

    # Feature 4: Age x Traffic interaction
    # Older bridges with high traffic are at much higher risk
    df['age_traffic_interaction'] = df['Age_of_Bridge'] * df['Traffic_Volume']

    # Feature 5: Age x Maintenance interaction
    # Older bridges with no maintenance degrade faster
    df['age_maintenance_interaction'] = df['Age_of_Bridge'] * df['maintenance_encoded']

    # Feature 6: Traffic x Maintenance interaction
    # High traffic with no maintenance is very risky
    df['traffic_maintenance_interaction'] = df['Traffic_Volume'] * df['maintenance_encoded']

    # Feature 7: Binned age categories
    df['age_bin'] = pd.cut(
        df['Age_of_Bridge'],
        bins=[0, 25, 50, 75, 100, float('inf')],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # Feature 8: Log traffic (reduces skew in traffic volume)
    df['log_traffic'] = np.log1p(df['Traffic_Volume'])

    # --- Build final feature set ---
    # Drop original columns that have been transformed
    drop_cols = [target, 'Age_of_Bridge', 'Traffic_Volume', 'Maintenance_Level']
    # Only drop columns that exist
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(columns=drop_cols)
    y = df[target]

    # Ensure all columns are numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)

    return X, y
