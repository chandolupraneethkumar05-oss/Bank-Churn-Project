"""
02_modeling.py — Feature Engineering, Model Development & Evaluation
Predictive Modeling and Risk Scoring for Bank Customer Churn
"""
import pandas as pd
import numpy as np
import json
import joblib
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, confusion_matrix, classification_report)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120
PALETTE = ['#2E5EAA', '#E8743B', '#5AA469', '#C64B4B', '#8B5FBF']
FIGDIR = 'figures'
RANDOM_STATE = 42

# ============================================================
# 1. LOAD & FEATURE ENGINEERING
# ============================================================
df = pd.read_csv('data/European_Bank.csv')
df = df.drop(columns=['Year', 'CustomerId', 'Surname'])  # non-informative identifiers

# Derived features
df['BalanceSalaryRatio'] = df['Balance'] / df['EstimatedSalary'].replace(0, 1)
df['ProductDensity'] = df['NumOfProducts'] / df['Tenure'].replace(0, 1)
df['EngagementProductInteraction'] = df['IsActiveMember'] * df['NumOfProducts']
df['AgeTenureInteraction'] = df['Age'] * df['Tenure']
df['IsZeroBalance'] = (df['Balance'] == 0).astype(int)

TARGET = 'Exited'
categorical_features = ['Geography', 'Gender']
numeric_features = [c for c in df.columns if c not in categorical_features + [TARGET]]

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ============================================================
# 2. TRAIN-TEST SPLIT (stratified)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print("Train churn rate:", y_train.mean().round(4), "| Test churn rate:", y_test.mean().round(4))

# ============================================================
# 3. PREPROCESSING PIPELINE
# ============================================================
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
])

# ============================================================
# 4. MODELS
# ============================================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
    'Decision Tree': DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced',
                                             random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
                                                      random_state=RANDOM_STATE),
}

results = []
fitted_pipelines = {}
roc_data = {}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    pipe = Pipeline([('preprocess', preprocessor), ('model', model)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    cv_auc = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)

    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'CV ROC-AUC (mean)': cv_auc.mean(),
        'CV ROC-AUC (std)': cv_auc.std(),
    }
    results.append(metrics)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, metrics['ROC-AUC'])

    print(f"\n{name}")
    for k, v in metrics.items():
        if k != 'Model':
            print(f"  {k}: {v:.4f}")

results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
results_df.to_csv('outputs/model_comparison.csv', index=False)
print("\n=== Model Comparison ===")
print(results_df.round(4).to_string(index=False))

BEST_MODEL_NAME = results_df.iloc[0]['Model']
best_pipe = fitted_pipelines[BEST_MODEL_NAME]
print(f"\nBest model: {BEST_MODEL_NAME}")

# ============================================================
# 5. VISUALIZATIONS — MODEL COMPARISON
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(results_df))
width = 0.15
for i, m in enumerate(metrics_to_plot):
    ax.bar(x + i * width, results_df[m], width, label=m,
           color=PALETTE[i % len(PALETTE)])
ax.set_xticks(x + width * 2)
ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
ax.set_ylim(0, 1.05)
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/10_model_comparison.png')
plt.close()

# ROC curves
fig, ax = plt.subplots(figsize=(6.5, 5.5))
for i, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=PALETTE[i % len(PALETTE)], linewidth=2)
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Model Comparison', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/11_roc_curves.png')
plt.close()

# Confusion matrix for best model
cm = confusion_matrix(y_test, best_pipe.predict(X_test))
fig, ax = plt.subplots(figsize=(5, 4.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix — {BEST_MODEL_NAME}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/12_confusion_matrix.png')
plt.close()

# ============================================================
# 6. EXPLAINABILITY
# ============================================================
# 6a. Feature importance (tree-based) for best model if applicable, else permutation importance
feature_names_num = numeric_features
ohe = best_pipe.named_steps['preprocess'].named_transformers_['cat']
feature_names_cat = list(ohe.get_feature_names_out(categorical_features))
all_feature_names = feature_names_num + feature_names_cat

if hasattr(best_pipe.named_steps['model'], 'feature_importances_'):
    importances = best_pipe.named_steps['model'].feature_importances_
    imp_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
else:
    importances = np.abs(best_pipe.named_steps['model'].coef_[0])
    imp_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})

imp_df = imp_df.sort_values('Importance', ascending=False).reset_index(drop=True)
imp_df.to_csv('outputs/feature_importance.csv', index=False)

fig, ax = plt.subplots(figsize=(9, 6))
top_imp = imp_df.head(12).iloc[::-1]
ax.barh(top_imp['Feature'], top_imp['Importance'], color=PALETTE[0])
ax.set_title(f'Feature Importance\n{BEST_MODEL_NAME}', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/13_feature_importance.png')
plt.close()

# 6b. Permutation importance (model-agnostic, more reliable)
perm_result = permutation_importance(best_pipe, X_test, y_test, n_repeats=15,
                                      random_state=RANDOM_STATE, scoring='roc_auc', n_jobs=-1)
perm_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance_mean': perm_result.importances_mean,
    'Importance_std': perm_result.importances_std
}).sort_values('Importance_mean', ascending=False).reset_index(drop=True)
perm_df.to_csv('outputs/permutation_importance.csv', index=False)

fig, ax = plt.subplots(figsize=(9, 6))
top_perm = perm_df.head(10).iloc[::-1]
ax.barh(top_perm['Feature'], top_perm['Importance_mean'],
        xerr=top_perm['Importance_std'], color=PALETTE[2])
ax.set_title(f'Permutation Importance (ROC-AUC drop)\n{BEST_MODEL_NAME}', fontsize=13, fontweight='bold')
ax.set_xlabel('Mean ROC-AUC decrease')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/14_permutation_importance.png')
plt.close()

# 6c. Partial dependence plots for top drivers
top_features_for_pdp = [f for f in ['Age', 'NumOfProducts', 'IsActiveMember', 'Balance']
                         if f in X_train.columns]
X_train_pdp = X_train.copy()
for c in X_train_pdp.select_dtypes(include=['int64', 'int32']).columns:
    X_train_pdp[c] = X_train_pdp[c].astype(float)
fig, axes = plt.subplots(1, len(top_features_for_pdp), figsize=(4.2 * len(top_features_for_pdp), 4))
PartialDependenceDisplay.from_estimator(best_pipe, X_train_pdp, top_features_for_pdp, ax=axes,
                                         line_kw={'color': PALETTE[0], 'linewidth': 2})
plt.suptitle(f'Partial Dependence Plots — {BEST_MODEL_NAME}', fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/15_partial_dependence.png', bbox_inches='tight')
plt.close()

# ============================================================
# 7. PROBABILITY DISTRIBUTION (for Streamlit dashboard reference)
# ============================================================
test_proba = best_pipe.predict_proba(X_test)[:, 1]
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.histplot(test_proba[y_test == 0], color=PALETTE[0], label='Retained', kde=True, alpha=0.5, ax=ax, stat='density')
sns.histplot(test_proba[y_test == 1], color=PALETTE[3], label='Churned', kde=True, alpha=0.5, ax=ax, stat='density')
ax.set_title('Predicted Churn Probability Distribution', fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted Churn Probability')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGDIR}/16_probability_distribution.png')
plt.close()

# ============================================================
# 8. SAVE ARTIFACTS
# ============================================================
joblib.dump(best_pipe, 'models/best_model.pkl')
joblib.dump(fitted_pipelines, 'models/all_models.pkl')

with open('outputs/best_model_name.txt', 'w') as f:
    f.write(BEST_MODEL_NAME)

meta = {
    'best_model': BEST_MODEL_NAME,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'test_metrics': results_df.iloc[0].to_dict(),
    'top_permutation_features': perm_df.head(6)['Feature'].tolist(),
}
with open('outputs/model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2, default=str)

print("\nAll artifacts saved: models/best_model.pkl, outputs/*.csv, figures/*.png")
