"""
Train a Support Vector Machine (SVM) classifier for Bridge Condition prediction.

Dataset : dataset/bridge_data.csv  (public/tabular dataset, NOT live sensor data)
Target  : Bridge_Condition (0 = Healthy, 1 = At-Risk)

Pipeline:
    feature_engineering.perform_feature_engineering
    -> StandardScaler (SVMs are distance-based, scaling is mandatory)
    -> SVC (class_weight='balanced', probability=True)

Measures applied:
    1. Class-imbalance handling via class_weight='balanced'
    2. Hyperparameter search (kernel, C, gamma) with Stratified 5-Fold CV
    3. Decision-threshold tuning on OUT-OF-FOLD training probabilities
       (the test set is never used for tuning -> leak-free estimate)
    4. Model-agnostic explainability (permutation importance + SHAP KernelExplainer)

Uses the same 80/20 stratified split (random_state=42) as train_model.py
so metrics are directly comparable with the XGBoost model.

Outputs:
    models/svm/      -> trained pipeline, feature list, threshold, best params
    notebooks/svm/   -> metrics report, metrics JSON, plots
"""
import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV,
    cross_val_predict, cross_validate, learning_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score,
    classification_report, matthews_corrcoef, balanced_accuracy_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from feature_engineering import perform_feature_engineering

warnings.filterwarnings('ignore', category=UserWarning)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, '..', 'dataset', 'bridge_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'svm')
NOTEBOOK_DIR = os.path.join(BASE_DIR, 'notebooks', 'svm')
XGB_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'xgboost')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(NOTEBOOK_DIR, exist_ok=True)

RANDOM_STATE = 42
CLASS_NAMES = ['Healthy (0)', 'At-Risk (1)']


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def find_optimal_threshold(y_true, y_prob):
    """Return the threshold that maximises F1, plus the full sweep table."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    rows = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        rows.append({
            'threshold': t,
            'precision': precision_score(y_true, y_pred_t, zero_division=0),
            'recall': recall_score(y_true, y_pred_t, zero_division=0),
            'f1': f1_score(y_true, y_pred_t, zero_division=0),
        })
    sweep = pd.DataFrame(rows)
    best = sweep.loc[sweep['f1'].idxmax()]
    return float(best['threshold']), float(best['f1']), sweep


def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) else 0.0,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
        'brier_score': brier_score_loss(y_true, y_prob),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def load_xgboost_baseline(X_test, y_test):
    """Evaluate the existing XGBoost model on the identical test split (if present)."""
    model_path = os.path.join(XGB_MODEL_DIR, 'condition_xgboost.joblib')
    feat_path = os.path.join(XGB_MODEL_DIR, 'model_features.joblib')
    thr_path = os.path.join(XGB_MODEL_DIR, 'optimal_threshold.json')
    if not all(os.path.exists(p) for p in (model_path, feat_path, thr_path)):
        print("  XGBoost artefacts not found - skipping comparison.")
        return None
    try:
        xgb_model = joblib.load(model_path)
        xgb_features = joblib.load(feat_path)
        with open(thr_path) as f:
            xgb_thr = json.load(f)['optimal_threshold']
        prob = xgb_model.predict_proba(X_test[xgb_features])[:, 1]
        return {
            'prob': prob,
            'threshold': xgb_thr,
            'default': compute_metrics(y_test, (prob >= 0.5).astype(int), prob),
            'tuned': compute_metrics(y_test, (prob >= xgb_thr).astype(int), prob),
        }
    except Exception as exc:  # noqa: BLE001 - comparison is optional
        print(f"  Could not evaluate XGBoost baseline: {exc}")
        return None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def train_and_evaluate():
    t_start = time.time()
    print(f"Loading data from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset shape: {df.shape}")

    class_counts = df['Bridge_Condition'].value_counts()
    n_negative, n_positive = int(class_counts[0]), int(class_counts[1])
    imbalance_ratio = n_negative / n_positive
    print(f"Class distribution: Healthy={n_negative}, At-Risk={n_positive} "
          f"(ratio {imbalance_ratio:.2f}:1)")

    print("\nPerforming feature engineering...")
    X, y = perform_feature_engineering(df)
    X = X.astype(float)
    feature_names = X.columns.tolist()
    print(f"Features ({len(feature_names)}): {feature_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # =========================================================================
    # MEASURE 1 + 2: class_weight='balanced' + Grid Search over kernel/C/gamma
    # =========================================================================
    print("\n[Measure 1+2] Grid search (Stratified 5-Fold CV, scoring=F1)...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(class_weight='balanced', probability=True,
                    random_state=RANDOM_STATE, cache_size=500)),
    ])
    param_grid = [
        {'svc__kernel': ['rbf'],
         'svc__C': [0.1, 0.5, 1, 5, 10, 50, 100],
         'svc__gamma': ['scale', 0.01, 0.05, 0.1, 0.5, 1.0]},
        {'svc__kernel': ['linear'],
         'svc__C': [0.01, 0.1, 1, 10]},
        {'svc__kernel': ['poly'],
         'svc__C': [0.1, 1, 10],
         'svc__degree': [2, 3],
         'svc__gamma': ['scale']},
    ]
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1',
                        n_jobs=-1, refit=True, return_train_score=True)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_params = {k.replace('svc__', ''): v for k, v in grid.best_params_.items()}
    n_candidates = len(grid.cv_results_['params'])
    print(f"  Evaluated {n_candidates} candidate configurations")
    print(f"  Best params: {best_params}")
    print(f"  Best CV F1:  {grid.best_score_:.4f}")

    cv_results_df = pd.DataFrame(grid.cv_results_)
    cv_results_df.to_csv(os.path.join(NOTEBOOK_DIR, 'svm_grid_search_results.csv'), index=False)

    # Best score per kernel (for the report)
    kernel_best = (cv_results_df
                   .assign(kernel=cv_results_df['param_svc__kernel'])
                   .groupby('kernel')['mean_test_score'].max()
                   .sort_values(ascending=False))

    # Full CV metrics for the chosen configuration
    cv_scores = cross_validate(
        best_model, X_train, y_train, cv=cv, n_jobs=-1,
        scoring=['f1', 'recall', 'precision', 'roc_auc', 'accuracy']
    )

    # =========================================================================
    # MEASURE 3: Threshold tuning on out-of-fold TRAINING probabilities
    # =========================================================================
    print("\n[Measure 3] Tuning threshold on out-of-fold training probabilities...")
    oof_prob = cross_val_predict(best_model, X_train, y_train, cv=cv,
                                 method='predict_proba', n_jobs=-1)[:, 1]
    optimal_threshold, oof_f1, threshold_df = find_optimal_threshold(y_train, oof_prob)
    print(f"  Optimal threshold (OOF): {optimal_threshold:.2f} (OOF F1={oof_f1:.4f})")

    # Test-set evaluation (test set untouched until here)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob >= 0.5).astype(int)
    y_pred_tuned = (y_prob >= optimal_threshold).astype(int)
    metrics_default = compute_metrics(y_test, y_pred_default, y_prob)
    metrics_tuned = compute_metrics(y_test, y_pred_tuned, y_prob)
    # Native SVC rule: decision_function > 0 (what .predict() and cross_validate use).
    # With class_weight='balanced' this boundary differs from Platt P >= 0.5.
    y_pred_native = best_model.predict(X_test)
    metrics_native = compute_metrics(y_test, y_pred_native, y_prob)

    # Reference only: what threshold would be optimal if peeking at the test set
    oracle_thr, oracle_f1, test_sweep = find_optimal_threshold(y_test, y_prob)

    # Support vector statistics
    svc = best_model.named_steps['svc']
    n_sv = svc.n_support_.tolist()

    # Overfitting check
    train_prob = best_model.predict_proba(X_train)[:, 1]
    train_metrics = compute_metrics(
        y_train, (train_prob >= optimal_threshold).astype(int), train_prob
    )

    # =========================================================================
    # MEASURE 4: Explainability
    # =========================================================================
    print("\n[Measure 4] Permutation importance (test set, scoring=ROC-AUC)...")
    perm = permutation_importance(best_model, X_test, y_test, scoring='roc_auc',
                                  n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1)
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm.importances_mean,
        'importance_std': perm.importances_std,
    }).sort_values('importance_mean', ascending=False)
    perm_df.to_csv(os.path.join(NOTEBOOK_DIR, 'svm_permutation_importance.csv'), index=False)

    shap_ok = False
    shap_mean_abs = None
    try:
        import shap
        print("  Computing SHAP values (KernelExplainer)...")
        background = shap.kmeans(X_train, 25)
        explainer = shap.KernelExplainer(
            lambda data: best_model.predict_proba(
                pd.DataFrame(data, columns=feature_names))[:, 1],
            background
        )
        X_shap = X_test.sample(n=min(100, len(X_test)), random_state=RANDOM_STATE)
        shap_values = explainer.shap_values(X_shap, nsamples=300, silent=True)
        shap_mean_abs = pd.Series(np.abs(shap_values).mean(axis=0),
                                  index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_shap, show=False)
        plt.title('SHAP Feature Impact - SVM (P[At-Risk])')
        plt.tight_layout()
        plt.savefig(os.path.join(NOTEBOOK_DIR, 'svm_shap_summary.png'), dpi=150)
        plt.close()
        shap_ok = True
    except Exception as exc:  # noqa: BLE001 - SHAP is optional
        print(f"  SHAP skipped: {exc}")

    # =========================================================================
    # XGBoost comparison on identical split
    # =========================================================================
    print("\nEvaluating XGBoost baseline on identical test split...")
    xgb = load_xgboost_baseline(X_test, y_test)

    # =========================================================================
    # VISUALISATIONS
    # =========================================================================
    print("\nGenerating plots...")

    # 1. Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, y_p, title in [
        (axes[0], y_pred_default, 'Default Threshold (0.50)'),
        (axes[1], y_pred_tuned, f'Tuned Threshold ({optimal_threshold:.2f}, OOF)'),
    ]:
        sns.heatmap(confusion_matrix(y_test, y_p), annot=True, fmt='d', cmap='Purples',
                    ax=ax, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(title)
    fig.suptitle('SVM Confusion Matrix Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(NOTEBOOK_DIR, 'svm_confusion_matrix.png'), dpi=150)
    plt.close()

    # 2. ROC curve (with XGBoost overlay)
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, color='purple', lw=2, label=f"SVM (AUC = {metrics_tuned['roc_auc']:.3f})")
    if xgb:
        fpr_x, tpr_x, _ = roc_curve(y_test, xgb['prob'])
        plt.plot(fpr_x, tpr_x, color='darkorange', lw=2, alpha=0.8,
                 label=f"XGBoost (AUC = {xgb['tuned']['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random baseline')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - SVM vs XGBoost'); plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(NOTEBOOK_DIR, 'svm_roc_curve.png'), dpi=150)
    plt.close()

    # 3. Precision-Recall curve
    plt.figure(figsize=(8, 6))
    p_vals, r_vals, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(r_vals, p_vals, color='purple', lw=2, label=f"SVM (AP = {metrics_tuned['pr_auc']:.3f})")
    if xgb:
        p_x, r_x, _ = precision_recall_curve(y_test, xgb['prob'])
        plt.plot(r_x, p_x, color='darkorange', lw=2, alpha=0.8,
                 label=f"XGBoost (AP = {xgb['tuned']['pr_auc']:.3f})")
    plt.axhline(y=y_test.mean(), color='gray', linestyle=':',
                label=f'No-skill baseline ({y_test.mean():.2f})')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - SVM vs XGBoost'); plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(NOTEBOOK_DIR, 'svm_precision_recall_curve.png'), dpi=150)
    plt.close()

    # 4. Threshold tuning (OOF train vs test)
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_df['threshold'], threshold_df['precision'], color='blue', label='Precision (OOF train)')
    plt.plot(threshold_df['threshold'], threshold_df['recall'], color='orange', label='Recall (OOF train)')
    plt.plot(threshold_df['threshold'], threshold_df['f1'], color='green', lw=2, label='F1 (OOF train)')
    plt.plot(test_sweep['threshold'], test_sweep['f1'], color='green', lw=1, ls='--', alpha=0.6,
             label='F1 (test, reference)')
    plt.axvline(optimal_threshold, color='red', ls='--', lw=2, label=f'Chosen threshold ({optimal_threshold:.2f})')
    plt.axvline(0.5, color='gray', ls=':', label='Default (0.50)')
    plt.xlabel('Classification Threshold'); plt.ylabel('Score')
    plt.title('SVM - Precision, Recall & F1 vs Threshold'); plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(NOTEBOOK_DIR, 'svm_threshold_tuning.png'), dpi=150)
    plt.close()

    # 5. Permutation importance
    plt.figure(figsize=(10, 7))
    pd_sorted = perm_df.sort_values('importance_mean')
    plt.barh(pd_sorted['feature'], pd_sorted['importance_mean'], xerr=pd_sorted['importance_std'],
             color=sns.color_palette('magma', len(pd_sorted)))
    plt.xlabel('Mean decrease in ROC-AUC when feature is shuffled')
    plt.title('SVM Permutation Feature Importance (test set, 30 repeats)')
    plt.tight_layout()
    plt.savefig(os.path.join(NOTEBOOK_DIR, 'svm_permutation_importance.png'), dpi=150)
    plt.close()

    # 6. Grid-search heatmap (RBF kernel: C x gamma)
    rbf = cv_results_df[cv_results_df['param_svc__kernel'] == 'rbf'].copy()
    rbf['C'] = rbf['param_svc__C'].astype(float)
    rbf['gamma'] = rbf['param_svc__gamma'].astype(str)
    heat = rbf.pivot_table(index='C', columns='gamma', values='mean_test_score')
    plt.figure(figsize=(9, 6))
    sns.heatmap(heat, annot=True, fmt='.3f', cmap='viridis')
    plt.title('RBF-SVM Grid Search - Mean CV F1 (C vs gamma)')
    plt.tight_layout()
    plt.savefig(os.path.join(NOTEBOOK_DIR, 'svm_grid_search_heatmap.png'), dpi=150)
    plt.close()

    # 7. Calibration curve
    plt.figure(figsize=(8, 6))
    frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=8, strategy='quantile')
    plt.plot(mean_pred, frac_pos, 'o-', color='purple', label='SVM (Platt-scaled)')
    if xgb:
        fp_x, mp_x = calibration_curve(y_test, xgb['prob'], n_bins=8, strategy='quantile')
        plt.plot(mp_x, fp_x, 's-', color='darkorange', alpha=0.8, label='XGBoost')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.xlabel('Mean predicted probability'); plt.ylabel('Observed fraction At-Risk')
    plt.title('Probability Calibration Curve'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(NOTEBOOK_DIR, 'svm_calibration_curve.png'), dpi=150)
    plt.close()

    # 8. Predicted probability distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(y_prob[y_test.values == 0], bins=25, color='seagreen', alpha=0.6,
                 label='Actual Healthy (0)', stat='density')
    sns.histplot(y_prob[y_test.values == 1], bins=25, color='crimson', alpha=0.6,
                 label='Actual At-Risk (1)', stat='density')
    plt.axvline(optimal_threshold, color='black', ls='--', label=f'Threshold {optimal_threshold:.2f}')
    plt.xlabel('Predicted P(At-Risk)'); plt.title('SVM Predicted Probability Distribution by True Class')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(NOTEBOOK_DIR, 'svm_probability_distribution.png'), dpi=150)
    plt.close()

    # 9. Learning curve
    sizes, lc_train, lc_val = learning_curve(
        best_model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 6), random_state=RANDOM_STATE
    )
    plt.figure(figsize=(9, 6))
    for scores, color, label in [(lc_train, 'blue', 'Training F1'), (lc_val, 'green', 'Validation F1')]:
        m, s = scores.mean(axis=1), scores.std(axis=1)
        plt.plot(sizes, m, 'o-', color=color, label=label)
        plt.fill_between(sizes, m - s, m + s, color=color, alpha=0.15)
    plt.xlabel('Training samples'); plt.ylabel('F1 Score')
    plt.title('SVM Learning Curve'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(NOTEBOOK_DIR, 'svm_learning_curve.png'), dpi=150)
    plt.close()

    # =========================================================================
    # Reports
    # =========================================================================
    elapsed = time.time() - t_start
    metric_keys = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'specificity',
                   'f1', 'mcc', 'roc_auc', 'pr_auc', 'brier_score']

    metrics_json = {
        'model': 'SVC (sklearn) with StandardScaler',
        'dataset': 'bridge_data.csv',
        'data_note': 'Public tabular dataset; not live structural sensor measurements.',
        'n_samples': len(df), 'n_train': len(X_train), 'n_test': len(X_test),
        'class_counts': {'healthy': n_negative, 'at_risk': n_positive},
        'features': feature_names,
        'best_params': best_params,
        'best_cv_f1': grid.best_score_,
        'kernel_best_cv_f1': kernel_best.to_dict(),
        'cv_metrics': {k.replace('test_', ''): {'mean': float(v.mean()), 'std': float(v.std())}
                       for k, v in cv_scores.items() if k.startswith('test_')},
        'optimal_threshold_oof': optimal_threshold,
        'oracle_test_threshold_reference': oracle_thr,
        'n_support_vectors': {'healthy': n_sv[0], 'at_risk': n_sv[1]},
        'train_metrics_tuned': train_metrics,
        'test_metrics_default': metrics_default,
        'test_metrics_tuned': metrics_tuned,
        'test_metrics_native_decision_function': metrics_native,
        'permutation_importance': perm_df.set_index('feature')['importance_mean'].to_dict(),
        'shap_mean_abs': shap_mean_abs.to_dict() if shap_ok else None,
        'xgboost_comparison': ({'threshold': xgb['threshold'], 'default': xgb['default'],
                                'tuned': xgb['tuned']} if xgb else None),
        'training_time_seconds': elapsed,
    }
    with open(os.path.join(NOTEBOOK_DIR, 'svm_metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2, default=float)

    metrics_path = os.path.join(NOTEBOOK_DIR, 'svm_performance_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("=" * 64 + "\n")
        f.write("  SVM MODEL PERFORMANCE REPORT\n")
        f.write("  Bridge Condition Classification (bridge_data.csv)\n")
        f.write("=" * 64 + "\n\n")

        f.write("DATASET OVERVIEW\n" + "-" * 44 + "\n")
        f.write(f"  Total samples:       {len(df)}\n")
        f.write(f"  Training samples:    {len(X_train)}\n")
        f.write(f"  Test samples:        {len(X_test)}\n")
        f.write(f"  Healthy (0):         {n_negative} ({n_negative/len(df)*100:.1f}%)\n")
        f.write(f"  At-Risk (1):         {n_positive} ({n_positive/len(df)*100:.1f}%)\n")
        f.write(f"  Imbalance ratio:     {imbalance_ratio:.2f}:1\n")
        f.write(f"  Features used:       {len(feature_names)}\n\n")

        f.write("MODEL CONFIGURATION\n" + "-" * 44 + "\n")
        f.write("  Pipeline:            StandardScaler -> SVC\n")
        f.write("  class_weight:        balanced\n")
        f.write("  Probabilities:       Platt scaling (probability=True)\n")
        for k, v in best_params.items():
            f.write(f"  {k:<20s} {v}\n")
        f.write(f"  Support vectors:     {sum(n_sv)} (Healthy={n_sv[0]}, At-Risk={n_sv[1]}) "
                f"= {sum(n_sv)/len(X_train)*100:.1f}% of train\n")
        f.write(f"  Grid candidates:     {n_candidates}\n\n")

        f.write("BEST CV F1 BY KERNEL\n" + "-" * 44 + "\n")
        for k, v in kernel_best.items():
            f.write(f"  {k:<20s} {v:.4f}\n")
        f.write("\n")

        f.write("CROSS-VALIDATION RESULTS (5-Fold Stratified, native SVC boundary)\n" + "-" * 44 + "\n")
        for k, v in metrics_json['cv_metrics'].items():
            f.write(f"  {k:<20s} {v['mean']:.4f} (+/- {v['std']:.4f})\n")
        f.write("\n")

        f.write("THRESHOLD SELECTION\n" + "-" * 44 + "\n")
        f.write(f"  Tuned on OOF train:  {optimal_threshold:.2f} (OOF F1 = {oof_f1:.4f})\n")
        f.write(f"  Test-optimal (ref):  {oracle_thr:.2f} (test F1 = {oracle_f1:.4f}) "
                f"[not used - would leak]\n\n")

        f.write(f"{'TEST SET METRICS':<22s}{'Native f(x)>0':>14s}{'P>=0.50':>14s}{'P>=' + format(optimal_threshold, '.2f'):>14s}\n")
        f.write("-" * 64 + "\n")
        for k in metric_keys:
            f.write(f"  {k:<20s}{metrics_native[k]:>14.4f}{metrics_default[k]:>14.4f}{metrics_tuned[k]:>14.4f}\n")
        cm_str = lambda m: '%d/%d/%d/%d' % (m['tn'], m['fp'], m['fn'], m['tp'])  # noqa: E731
        f.write(f"  {'TN/FP/FN/TP':<20s}{cm_str(metrics_native):>14s}"
                f"{cm_str(metrics_default):>14s}{cm_str(metrics_tuned):>14s}\n")
        f.write("  (ROC-AUC, PR-AUC, Brier are threshold-independent.)\n\n")

        f.write("OVERFITTING CHECK (tuned threshold)\n" + "-" * 44 + "\n")
        for k in ['f1', 'recall', 'precision', 'roc_auc']:
            f.write(f"  {k:<20s} train={train_metrics[k]:.4f}  test={metrics_tuned[k]:.4f}  "
                    f"gap={train_metrics[k]-metrics_tuned[k]:+.4f}\n")
        f.write("\n")

        if xgb:
            f.write("SVM vs XGBOOST (identical test split)\n" + "-" * 44 + "\n")
            f.write(f"  {'metric':<20s}{'SVM@' + format(optimal_threshold, '.2f'):>12s}"
                    f"{'XGB@' + format(xgb['threshold'], '.2f'):>12s}{'SVM@0.50':>12s}{'XGB@0.50':>12s}\n")
            for k in metric_keys:
                f.write(f"  {k:<20s}{metrics_tuned[k]:>12.4f}{xgb['tuned'][k]:>12.4f}"
                        f"{metrics_default[k]:>12.4f}{xgb['default'][k]:>12.4f}\n")
            f.write("  NOTE: XGBoost's threshold was tuned on the test set in train_model.py,\n"
                    "        so its tuned figures are optimistic. SVM's threshold is leak-free.\n\n")

        f.write("PERMUTATION IMPORTANCE (drop in ROC-AUC, test set)\n" + "-" * 44 + "\n")
        for _, r in perm_df.iterrows():
            f.write(f"  {r['feature']:<34s} {r['importance_mean']:+.4f} (+/- {r['importance_std']:.4f})\n")
        f.write("\n")
        if shap_ok:
            f.write("SHAP MEAN |VALUE| (impact on P[At-Risk])\n" + "-" * 44 + "\n")
            for k, v in shap_mean_abs.items():
                f.write(f"  {k:<34s} {v:.4f}\n")
            f.write("\n")

        f.write("DETAILED CLASSIFICATION REPORT (Tuned Threshold)\n" + "-" * 44 + "\n")
        f.write(classification_report(y_test, y_pred_tuned, target_names=CLASS_NAMES))
        f.write(f"\nTotal pipeline runtime: {elapsed:.1f}s\n")

    print(f"Metrics saved to {metrics_path}")

    # =========================================================================
    # Save model artefacts
    # =========================================================================
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'svm_condition_model.joblib'))
    with open(os.path.join(MODEL_DIR, 'svm_features.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)
    with open(os.path.join(MODEL_DIR, 'svm_threshold.json'), 'w') as f:
        json.dump({'optimal_threshold': optimal_threshold,
                   'selection_method': 'max F1 on 5-fold out-of-fold training probabilities'}, f, indent=2)
    with open(os.path.join(MODEL_DIR, 'svm_best_params.json'), 'w') as f:
        json.dump({'best_params': best_params, 'best_cv_f1': grid.best_score_}, f, indent=2, default=str)
    print(f"Model artefacts saved to {MODEL_DIR}")

    print("\n" + "=" * 60)
    print(f"  SVM TRAINING COMPLETE ({elapsed:.1f}s)")
    print("=" * 60)


if __name__ == "__main__":
    train_and_evaluate()
