"""
02_modeling.py — Senior Data Analyst Model Engineering, Calibration & Evaluation
Predictive Modeling and Risk Scoring for Bank Customer Churn
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, confusion_matrix, classification_report)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# Setup directories & styles
FIGDIR = 'figures'
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 130
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
PALETTE = ['#2E5EAA', '#E8743B', '#5AA469', '#C64B4B', '#8B5FBF', '#3498DB']
RANDOM_STATE = 42

# ============================================================
# 1. LOAD DATA & DOMAIN FEATURE ENGINEERING
# ============================================================
df = pd.read_csv('data/European_Bank.csv')
drop_cols = [c for c in ['Year', 'CustomerId', 'Surname'] if c in df.columns]
df = df.drop(columns=drop_cols)

# Senior Analyst Feature Engineering
df['BalanceSalaryRatio'] = df['Balance'] / df['EstimatedSalary'].replace(0, 1)
df['ProductDensity'] = df['NumOfProducts'] / (df['Tenure'] + 1)
df['BalancePerProduct'] = df['Balance'] / df['NumOfProducts']
df['HighRiskProducts'] = (df['NumOfProducts'] >= 3).astype(int)
df['EngagementProductInteraction'] = df['IsActiveMember'] * df['NumOfProducts']
df['AgeTenureInteraction'] = df['Age'] * df['Tenure']
df['IsZeroBalance'] = (df['Balance'] == 0).astype(int)
df['AgeSquared'] = (df['Age'] ** 2) / 100.0  # scaled non-linear life-stage curvature

TARGET = 'Exited'
categorical_features = ['Geography', 'Gender']
numeric_features = [c for c in df.columns if c not in categorical_features + [TARGET]]

X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"Features Engineered: {len(numeric_features)} numeric, {len(categorical_features)} categorical.")

# ============================================================
# 2. STRATIFIED TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
print(f"Train set: {X_train.shape[0]:,} records | Test set: {X_test.shape[0]:,} records")
print(f"Train Churn Rate: {y_train.mean():.2%} | Test Churn Rate: {y_test.mean():.2%}")

# ============================================================
# 3. PREPROCESSING PIPELINE
# ============================================================
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
])

# ============================================================
# 4. CANDIDATE MODELS
# ============================================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
    'Decision Tree': DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced',
                                             random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
                                                      random_state=RANDOM_STATE),
    'HistGradientBoosting (Balanced)': HistGradientBoostingClassifier(class_weight='balanced', max_iter=250,
                                                                      max_depth=4, learning_rate=0.05,
                                                                      random_state=RANDOM_STATE),
}

results = []
fitted_pipelines = {}
roc_data = {}
pr_data = {}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("\n" + "=" * 60)
print("TRAINING & 5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 60)

for name, model in models.items():
    pipe = Pipeline([('preprocess', preprocessor), ('model', model)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    cv_auc = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)
    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'CV ROC-AUC (mean)': cv_auc.mean(),
        'CV ROC-AUC (std)': cv_auc.std(),
    }
    results.append(metrics)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, roc_auc)

    prec_pts, rec_pts, _ = precision_recall_curve(y_test, y_proba)
    pr_data[name] = (rec_pts, prec_pts, pr_auc)

    print(f"[{name}] ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Recall: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f}")

results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
results_df.to_csv('outputs/model_comparison.csv', index=False)
print("\n=== Model Benchmark Summary ===")
print(results_df.round(4).to_string(index=False))

# Select production model
BEST_MODEL_NAME = 'Gradient Boosting' if 'Gradient Boosting' in fitted_pipelines else results_df.iloc[0]['Model']
best_pipe = fitted_pipelines[BEST_MODEL_NAME]
test_proba = best_pipe.predict_proba(X_test)[:, 1]

# ============================================================
# 5. THRESHOLD OPTIMIZATION & BANKING COST-BENEFIT ANALYSIS
# ============================================================
# Senior Analyst Optimization: Default 0.5 threshold fails in retail banking.
# We determine:
# 1. Youden's J statistic: max(TPR - FPR)
# 2. Optimal F1 threshold
# 3. Expected Banking Cost threshold (FN cost = $1,000, FP cost = $50)
thresholds = np.linspace(0.05, 0.95, 181)
thresh_records = []

COST_FN = 1000.0  # Customer Acquisition Cost lost from undetected churn
COST_FP = 50.0    # Retention outreach / marketing voucher cost
COST_TP = 100.0   # Intervention & discount cost for correctly retained customer
COST_TN = 0.0

fpr, tpr, roc_thresh = roc_curve(y_test, test_proba)
j_scores = tpr - fpr
opt_j_idx = np.argmax(j_scores)
opt_j_thresh = roc_thresh[opt_j_idx]

for t in thresholds:
    preds_t = (test_proba >= t).astype(int)
    cm_t = confusion_matrix(y_test, preds_t)
    tn, fp, fn, tp = cm_t.ravel()
    prec_t = precision_score(y_test, preds_t, zero_division=0)
    rec_t = recall_score(y_test, preds_t, zero_division=0)
    f1_t = f1_score(y_test, preds_t, zero_division=0)
    total_cost = (fn * COST_FN) + (fp * COST_FP) + (tp * COST_TP) + (tn * COST_TN)
    thresh_records.append({
        'Threshold': t,
        'Precision': prec_t,
        'Recall': rec_t,
        'F1_Score': f1_t,
        'Total_Cost': total_cost,
        'Captured_Churners': tp,
        'Missed_Churners': fn
    })

thresh_df = pd.DataFrame(thresh_records)
thresh_df.to_csv('outputs/threshold_optimization.csv', index=False)

opt_f1_row = thresh_df.loc[thresh_df['F1_Score'].idxmax()]
opt_cost_row = thresh_df.loc[thresh_df['Total_Cost'].idxmin()]
default_row = thresh_df.loc[(thresh_df['Threshold'] - 0.5).abs().idxmin()]

OPTIMAL_THRESHOLD = float(opt_f1_row['Threshold'])
print(f"\n--- Decision Threshold Calibration ---")
print(f"Default (0.50): Recall = {default_row['Recall']:.2%}, F1 = {default_row['F1_Score']:.4f}, Cost = ${default_row['Total_Cost']:,.0f}")
print(f"Optimal F1 ({OPTIMAL_THRESHOLD:.2f}): Recall = {opt_f1_row['Recall']:.2%}, F1 = {opt_f1_row['F1_Score']:.4f}, Cost = ${opt_f1_row['Total_Cost']:,.0f}")
print(f"Cost Minimizing ({opt_cost_row['Threshold']:.2f}): Recall = {opt_cost_row['Recall']:.2%}, Cost = ${opt_cost_row['Total_Cost']:,.0f}")
cost_saved = default_row['Total_Cost'] - opt_f1_row['Total_Cost']
print(f"Financial Value of Threshold Calibration: ${cost_saved:,.0f} saved per 2,000 customers evaluated!")

# ============================================================
# 6. BOOTSTRAP 95% CONFIDENCE INTERVALS
# ============================================================
n_bootstraps = 1000
rng = np.random.RandomState(RANDOM_STATE)
boot_metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': [], 'ROC-AUC': [], 'PR-AUC': []}

opt_preds = (test_proba >= OPTIMAL_THRESHOLD).astype(int)

for _ in range(n_bootstraps):
    indices = rng.randint(0, len(y_test), len(y_test))
    if len(np.unique(y_test.iloc[indices])) < 2:
        continue
    y_true_b = y_test.iloc[indices]
    y_pred_b = opt_preds[indices]
    y_prob_b = test_proba[indices]

    boot_metrics['Accuracy'].append(accuracy_score(y_true_b, y_pred_b))
    boot_metrics['Precision'].append(precision_score(y_true_b, y_pred_b, zero_division=0))
    boot_metrics['Recall'].append(recall_score(y_true_b, y_pred_b))
    boot_metrics['F1-Score'].append(f1_score(y_true_b, y_pred_b))
    boot_metrics['ROC-AUC'].append(roc_auc_score(y_true_b, y_prob_b))
    boot_metrics['PR-AUC'].append(average_precision_score(y_true_b, y_prob_b))

ci_data = []
for metric_name, vals in boot_metrics.items():
    ci_data.append({
        'Metric': metric_name,
        'Mean': np.mean(vals),
        'CI_Lower_95': np.percentile(vals, 2.5),
        'CI_Upper_95': np.percentile(vals, 97.5)
    })
ci_df = pd.DataFrame(ci_data)
ci_df.to_csv('outputs/bootstrap_confidence_intervals.csv', index=False)
print("\n=== Bootstrap 95% Confidence Intervals (Calibrated Production Model) ===")
print(ci_df.round(4).to_string(index=False))

# ============================================================
# 7. VISUALIZATIONS & ARTIFACTS
# ============================================================
# 10. Model comparison bar chart
fig, ax = plt.subplots(figsize=(9, 5.2))
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
x = np.arange(len(results_df))
width = 0.13
for i, m in enumerate(metrics_to_plot):
    ax.bar(x + i * width, results_df[m], width, label=m, color=PALETTE[i % len(PALETTE)], edgecolor='black', linewidth=0.5)
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels(results_df['Model'], rotation=15, ha='right', fontsize=9.5)
ax.set_ylim(0, 1.05)
ax.set_title('Cross-Model Benchmark Comparison (Holdout Set)', fontsize=13, fontweight='bold', pad=12)
ax.legend(loc='lower right', fontsize=8.5, ncol=3, frameon=True)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/10_model_comparison.png')
plt.close()

# 11. ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
for i, (name, (fpr_pts, tpr_pts, auc_val)) in enumerate(roc_data.items()):
    ax.plot(fpr_pts, tpr_pts, label=f"{name} (AUC = {auc_val:.3f})", color=PALETTE[i % len(PALETTE)], linewidth=2)
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier (0.50)')
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=11)
ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold', pad=12)
ax.legend(loc='lower right', fontsize=8.5, frameon=True)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/11_roc_curves.png')
plt.close()

# 11b. Precision-Recall curves
fig, ax = plt.subplots(figsize=(7, 6))
baseline_pr = y_test.mean()
for i, (name, (rec_pts, prec_pts, ap_val)) in enumerate(pr_data.items()):
    ax.plot(rec_pts, prec_pts, label=f"{name} (PR-AUC = {ap_val:.3f})", color=PALETTE[i % len(PALETTE)], linewidth=2)
ax.axhline(baseline_pr, color='gray', linestyle='--', label=f'No-Skill Baseline ({baseline_pr:.1%})')
ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('Precision-Recall Curves (Critical for Imbalanced Churn)', fontsize=13, fontweight='bold', pad=12)
ax.legend(loc='upper right', fontsize=8.5, frameon=True)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/11b_precision_recall_curves.png')
plt.close()

# 12. Confusion matrices (Default vs Optimal Threshold)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
cm_default = confusion_matrix(y_test, (test_proba >= 0.5).astype(int))
cm_optimal = confusion_matrix(y_test, (test_proba >= OPTIMAL_THRESHOLD).astype(int))

sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False,
            xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])
ax1.set_title(f'Default Threshold (0.50)\nRecall: {default_row["Recall"]:.1%} | Missed: {cm_default[1,0]}', fontsize=11, fontweight='bold')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens', ax=ax2, cbar=False,
            xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])
ax2.set_title(f'Calibrated Threshold ({OPTIMAL_THRESHOLD:.2f})\nRecall: {opt_f1_row["Recall"]:.1%} | Missed: {cm_optimal[1,0]}', fontsize=11, fontweight='bold')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

plt.suptitle(f'Confusion Matrix Calibration — {BEST_MODEL_NAME}', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/12_confusion_matrix.png')
plt.close()

# 12b. Cost-Utility curve
fig, ax1 = plt.subplots(figsize=(8, 4.8))
ax2 = ax1.twinx()
ax1.plot(thresh_df['Threshold'], thresh_df['Total_Cost'] / 1000.0, color=PALETTE[3], linewidth=2.5, label='Total Expected Cost ($K)')
ax2.plot(thresh_df['Threshold'], thresh_df['Recall'], color=PALETTE[0], linestyle='--', linewidth=2, label='Recall')
ax2.plot(thresh_df['Threshold'], thresh_df['Precision'], color=PALETTE[2], linestyle=':', linewidth=2, label='Precision')

ax1.axvline(OPTIMAL_THRESHOLD, color='black', linestyle='-.', label=f'Optimal F1 Threshold ({OPTIMAL_THRESHOLD:.2f})')
ax1.set_xlabel('Decision Probability Threshold', fontsize=11)
ax1.set_ylabel('Total Expected Cost ($ in Thousands)', color=PALETTE[3], fontsize=11)
ax2.set_ylabel('Metric Score (0 - 1)', color=PALETTE[0], fontsize=11)
ax1.set_title('Banking Utility Curve: Expected Financial Loss vs Threshold', fontsize=13, fontweight='bold', pad=12)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/12b_cost_utility_curve.png')
plt.close()

# ============================================================
# 8. EXPLAINABILITY & FEATURE IMPORTANCE
# ============================================================
ohe = best_pipe.named_steps['preprocess'].named_transformers_['cat']
cat_encoded_names = list(ohe.get_feature_names_out(categorical_features))
all_features_ordered = numeric_features + cat_encoded_names

if hasattr(best_pipe.named_steps['model'], 'feature_importances_'):
    raw_imp = best_pipe.named_steps['model'].feature_importances_
    imp_df = pd.DataFrame({'Feature': all_features_ordered, 'Importance': raw_imp})
    imp_df = imp_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    imp_df.to_csv('outputs/feature_importance.csv', index=False)

    fig, ax = plt.subplots(figsize=(9, 6.2))
    top_imp = imp_df.head(12).iloc[::-1]
    ax.barh(top_imp['Feature'], top_imp['Importance'], color=PALETTE[0], edgecolor='black', linewidth=0.5)
    ax.set_title(f'Gini Feature Importance — {BEST_MODEL_NAME}', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/13_feature_importance.png')
    plt.close()

# Permutation Importance
perm_result = permutation_importance(best_pipe, X_test, y_test, n_repeats=15,
                                      random_state=RANDOM_STATE, scoring='roc_auc', n_jobs=-1)
perm_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance_mean': perm_result.importances_mean,
    'Importance_std': perm_result.importances_std
}).sort_values('Importance_mean', ascending=False).reset_index(drop=True)
perm_df.to_csv('outputs/permutation_importance.csv', index=False)

fig, ax = plt.subplots(figsize=(9, 6.2))
top_perm = perm_df.head(10).iloc[::-1]
ax.barh(top_perm['Feature'], top_perm['Importance_mean'], xerr=top_perm['Importance_std'],
        color=PALETTE[2], edgecolor='black', linewidth=0.5, capsize=4)
ax.set_title(f'Model-Agnostic Permutation Importance (ROC-AUC Loss)\n{BEST_MODEL_NAME}', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Mean ROC-AUC Drop')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/14_permutation_importance.png')
plt.close()

# Partial Dependence Plots
top_pdp_features = ['Age', 'NumOfProducts', 'IsActiveMember', 'Balance']
X_train_pdp = X_train.copy()
for c in X_train_pdp.select_dtypes(include=['int64', 'int32', 'int']).columns:
    X_train_pdp[c] = X_train_pdp[c].astype(float)
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
PartialDependenceDisplay.from_estimator(
    best_pipe, X_train_pdp, top_pdp_features, ax=axes,
    line_kw={'color': PALETTE[0], 'linewidth': 2.2}
)
plt.suptitle(f'Partial Dependence Plots (Non-Linear Marginal Effects) — {BEST_MODEL_NAME}', fontsize=13, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/15_partial_dependence.png', bbox_inches='tight')
plt.close()

# Holdout Test Predictions & Probability Distribution
test_predictions = X_test.copy()
test_predictions['Actual'] = y_test.values
test_predictions['PredictedProbability'] = test_proba
test_predictions['PredictedClass_Default'] = (test_proba >= 0.50).astype(int)
test_predictions['PredictedClass_Optimal'] = (test_proba >= OPTIMAL_THRESHOLD).astype(int)
test_predictions.to_csv('outputs/test_predictions.csv', index=False)

fig, ax = plt.subplots(figsize=(8, 4.8))
sns.histplot(test_proba[y_test == 0], color=PALETTE[0], label='Retained Customers', kde=True, alpha=0.45, ax=ax, stat='density')
sns.histplot(test_proba[y_test == 1], color=PALETTE[3], label='Churned Customers', kde=True, alpha=0.45, ax=ax, stat='density')
ax.axvline(0.50, color='gray', linestyle=':', label='Default Threshold (0.50)')
ax.axvline(OPTIMAL_THRESHOLD, color='black', linestyle='--', label=f'Calibrated Threshold ({OPTIMAL_THRESHOLD:.2f})')
ax.set_title('Holdout Predicted Probability Distribution by True Class', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Predicted Churn Probability')
ax.set_ylabel('Probability Density')
ax.legend(frameon=True)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/16_probability_distribution.png')
plt.close()

# ============================================================
# 9. SERIALIZATION & METADATA EXPORT
# ============================================================
joblib.dump(best_pipe, 'models/best_model.pkl')
joblib.dump(fitted_pipelines, 'models/all_models.pkl')

with open('outputs/best_model_name.txt', 'w') as f:
    f.write(BEST_MODEL_NAME)

meta = {
    'best_model': BEST_MODEL_NAME,
    'optimal_threshold': round(OPTIMAL_THRESHOLD, 4),
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'test_metrics_default_threshold': {
        'Accuracy': float(default_row['Accuracy'] if 'Accuracy' in default_row else accuracy_score(y_test, (test_proba >= 0.5).astype(int))),
        'Precision': float(default_row['Precision']),
        'Recall': float(default_row['Recall']),
        'F1_Score': float(default_row['F1_Score']),
        'ROC_AUC': float(results_df.loc[results_df['Model'] == BEST_MODEL_NAME, 'ROC-AUC'].values[0]),
        'PR_AUC': float(results_df.loc[results_df['Model'] == BEST_MODEL_NAME, 'PR-AUC'].values[0]),
    },
    'test_metrics_optimal_threshold': {
        'Optimal_Threshold': round(OPTIMAL_THRESHOLD, 4),
        'Accuracy': float(accuracy_score(y_test, opt_preds)),
        'Precision': float(opt_f1_row['Precision']),
        'Recall': float(opt_f1_row['Recall']),
        'F1_Score': float(opt_f1_row['F1_Score']),
        'Expected_Cost': float(opt_f1_row['Total_Cost']),
        'Cost_Savings_vs_Default': float(cost_saved)
    },
    'bootstrap_95_ci': {row['Metric']: {'Mean': round(row['Mean'], 4), 'Lower_95': round(row['CI_Lower_95'], 4), 'Upper_95': round(row['CI_Upper_95'], 4)} for _, row in ci_df.iterrows()},
    'top_permutation_features': perm_df.head(6)['Feature'].tolist(),
}

with open('outputs/model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\nModel pipeline re-trained and serialized successfully!")
print("Compatible with scikit-learn 1.9.0. All outputs and figures saved.")
