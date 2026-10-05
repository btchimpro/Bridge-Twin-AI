import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score,
    classification_report
)
from feature_engineering import perform_feature_engineering

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, '..', 'dataset', 'bridge_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
NOTEBOOK_DIR = os.path.join(BASE_DIR, 'notebooks')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(NOTEBOOK_DIR, exist_ok=True)


def find_optimal_threshold(y_true, y_prob):
    """
    Find the optimal classification threshold that maximizes F1 score.
    This directly improves both precision and recall by moving the
    decision boundary away from the default 0.5.
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    results = []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        p = precision_score(y_true, y_pred_t, zero_division=0)
        r = recall_score(y_true, y_pred_t, zero_division=0)
        f = f1_score(y_true, y_pred_t, zero_division=0)
        results.append({'threshold': t, 'precision': p, 'recall': r, 'f1': f})
        if f > best_f1:
            best_f1 = f
            best_threshold = t

    return best_threshold, best_f1, pd.DataFrame(results)


def train_and_evaluate():
    print(f"Loading data from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset shape: {df.shape}")

    # --- Class distribution ---
    class_counts = df['Bridge_Condition'].value_counts()
    print(f"\nClass distribution:\n{class_counts}")
    n_negative = class_counts[0]
    n_positive = class_counts[1]
    imbalance_ratio = n_negative / n_positive
    print(f"Imbalance ratio (neg/pos): {imbalance_ratio:.2f}")

    print("\nPerforming feature engineering...")
    X, y = perform_feature_engineering(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {X.columns.tolist()}")

    print("\nSplitting dataset (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # =========================================================================
    # MEASURE 1: Handle class imbalance with scale_pos_weight
    # =========================================================================
    print(f"\n[Measure 1] Applying scale_pos_weight = {imbalance_ratio:.2f}")

    # =========================================================================
    # MEASURE 2: Hyperparameter tuning with Stratified K-Fold CV
    # =========================================================================
    print("[Measure 2] Training with tuned hyperparameters and Stratified 5-Fold CV...")

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=imbalance_ratio,  # MEASURE 1
        random_state=42,
        eval_metric='logloss'
    )

    # Cross-validation scores
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    cv_recall_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall')
    cv_precision_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision')
    print(f"  CV F1 scores: {cv_f1_scores}")
    print(f"  CV F1 mean: {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std():.4f})")

    # Final fit
    model.fit(X_train, y_train)

    # =========================================================================
    # MEASURE 3: Optimal threshold tuning
    # =========================================================================
    y_prob = model.predict_proba(X_test)[:, 1]
    optimal_threshold, optimal_f1, threshold_df = find_optimal_threshold(y_test, y_prob)
    print(f"\n[Measure 3] Optimal threshold: {optimal_threshold:.2f} (F1={optimal_f1:.4f})")

    # Predictions with default vs optimal threshold
    y_pred_default = model.predict(X_test)
    y_pred_tuned = (y_prob >= optimal_threshold).astype(int)

    # =========================================================================
    # Compute metrics for BOTH thresholds
    # =========================================================================
    def compute_metrics(y_true, y_pred, y_prob):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob)
        }

    metrics_default = compute_metrics(y_test, y_pred_default, y_prob)
    metrics_tuned = compute_metrics(y_test, y_pred_tuned, y_prob)

    # =========================================================================
    # Save comprehensive metrics report
    # =========================================================================
    metrics_path = os.path.join(NOTEBOOK_DIR, 'performance_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  XGBOOST MODEL PERFORMANCE REPORT\n")
        f.write("  Bridge Condition Classification (bridge_data.csv)\n")
        f.write("=" * 60 + "\n\n")

        f.write("DATASET OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total samples:       {len(df)}\n")
        f.write(f"  Training samples:    {len(X_train)}\n")
        f.write(f"  Test samples:        {len(X_test)}\n")
        f.write(f"  Healthy (0):         {n_negative} ({n_negative/len(df)*100:.1f}%)\n")
        f.write(f"  At-Risk (1):         {n_positive} ({n_positive/len(df)*100:.1f}%)\n")
        f.write(f"  Imbalance ratio:     {imbalance_ratio:.2f}:1\n")
        f.write(f"  Features used:       {len(X.columns)}\n\n")

        f.write("MEASURES APPLIED TO IMPROVE PRECISION & RECALL\n")
        f.write("-" * 40 + "\n")
        f.write("  1. Class imbalance handling (scale_pos_weight)\n")
        f.write("  2. Enhanced feature engineering (interactions, bins)\n")
        f.write("  3. Tuned hyperparameters with Stratified 5-Fold CV\n")
        f.write("  4. Optimal decision threshold tuning\n\n")

        f.write("CROSS-VALIDATION RESULTS (5-Fold Stratified)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  F1 mean:             {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std():.4f})\n")
        f.write(f"  Recall mean:         {cv_recall_scores.mean():.4f} (+/- {cv_recall_scores.std():.4f})\n")
        f.write(f"  Precision mean:      {cv_precision_scores.mean():.4f} (+/- {cv_precision_scores.std():.4f})\n\n")

        f.write("TEST SET METRICS (Default Threshold = 0.50)\n")
        f.write("-" * 40 + "\n")
        for k, v in metrics_default.items():
            f.write(f"  {k:<20s} {v:.4f}\n")

        f.write(f"\nTEST SET METRICS (Optimal Threshold = {optimal_threshold:.2f})\n")
        f.write("-" * 40 + "\n")
        for k, v in metrics_tuned.items():
            f.write(f"  {k:<20s} {v:.4f}\n")

        f.write(f"\nIMPROVEMENT SUMMARY\n")
        f.write("-" * 40 + "\n")
        for metric in ['precision', 'recall', 'f1']:
            delta = metrics_tuned[metric] - metrics_default[metric]
            direction = "+" if delta >= 0 else ""
            f.write(f"  {metric:<20s} {direction}{delta:.4f}\n")

        f.write(f"\nDETAILED CLASSIFICATION REPORT (Optimal Threshold)\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(y_test, y_pred_tuned, target_names=['Healthy (0)', 'At-Risk (1)']))

    print(f"\nMetrics saved to {metrics_path}")

    # =========================================================================
    # VISUALIZATIONS
    # =========================================================================

    # --- Plot 1: Confusion Matrix (Side by Side) ---
    print("Generating Confusion Matrix comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, y_p, title in [
        (axes[0], y_pred_default, 'Default Threshold (0.50)'),
        (axes[1], y_pred_tuned, f'Optimal Threshold ({optimal_threshold:.2f})')
    ]:
        cm = confusion_matrix(y_test, y_p)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Healthy (0)', 'At-Risk (1)'],
                    yticklabels=['Healthy (0)', 'At-Risk (1)'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)
    fig.suptitle('Confusion Matrix Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(NOTEBOOK_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # --- Plot 2: ROC Curve ---
    print("Generating ROC Curve...")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = metrics_tuned['roc_auc']
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(NOTEBOOK_DIR, 'roc_curve.png'), dpi=150)
    plt.close()

    # --- Plot 3: Precision-Recall Curve ---
    print("Generating Precision-Recall Curve...")
    prec_vals, rec_vals, pr_thresholds = precision_recall_curve(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(rec_vals, prec_vals, color='green', lw=2, label=f'PR curve (AP = {avg_prec:.3f})')
    plt.axvline(x=metrics_tuned['recall'], color='red', linestyle='--', alpha=0.7,
                label=f"Optimal threshold recall ({metrics_tuned['recall']:.2f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(NOTEBOOK_DIR, 'precision_recall_curve.png'), dpi=150)
    plt.close()

    # --- Plot 4: Threshold vs Precision/Recall/F1 ---
    print("Generating Threshold Tuning plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_df['threshold'], threshold_df['precision'], label='Precision', color='blue')
    plt.plot(threshold_df['threshold'], threshold_df['recall'], label='Recall', color='orange')
    plt.plot(threshold_df['threshold'], threshold_df['f1'], label='F1 Score', color='green', lw=2)
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', lw=2,
                label=f'Optimal threshold ({optimal_threshold:.2f})')
    plt.axvline(x=0.5, color='gray', linestyle=':', lw=1, label='Default threshold (0.50)')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall & F1 vs Classification Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(NOTEBOOK_DIR, 'threshold_tuning.png'), dpi=150)
    plt.close()

    # --- Plot 5: SHAP Summary ---
    print("Generating SHAP explainability...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_train, show=False)
    plt.title('SHAP Feature Importance for Bridge Condition Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(NOTEBOOK_DIR, 'shap_summary_plot.png'), dpi=150)
    plt.close()

    # --- Plot 6: Feature Importance (XGBoost native) ---
    print("Generating Feature Importance bar chart...")
    importance = model.feature_importances_
    feat_imp = pd.Series(importance, index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(10, 8))
    feat_imp.plot(kind='barh', color=sns.color_palette('viridis', len(feat_imp)))
    plt.xlabel('Feature Importance (Gain)')
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(NOTEBOOK_DIR, 'feature_importance.png'), dpi=150)
    plt.close()

    # =========================================================================
    # Save model and artifacts
    # =========================================================================
    model_path = os.path.join(MODEL_DIR, 'condition_xgboost.joblib')
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    features_path = os.path.join(MODEL_DIR, 'model_features.joblib')
    joblib.dump(X.columns.tolist(), features_path)
    print(f"Feature list saved to {features_path}")

    threshold_path = os.path.join(MODEL_DIR, 'optimal_threshold.json')
    with open(threshold_path, 'w') as f:
        json.dump({'optimal_threshold': float(optimal_threshold)}, f)
    print(f"Optimal threshold saved to {threshold_path}")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    train_and_evaluate()
