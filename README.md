# Predictive Modeling and Risk Scoring for Bank Customer Churn
**Submission for Unified Mentor**

## Contents

| File | Description |
|---|---|
| `Research_Paper_Bank_Customer_Churn.docx` | Full technical research paper — EDA, methodology, model evaluation, explainability, insights & recommendations |
| `Executive_Summary_Bank_Customer_Churn.docx` | 2-page non-technical summary for government/regulatory stakeholders |
| `Bank_Customer_Churn_Analysis.ipynb` | End-to-end Jupyter notebook (EDA → feature engineering → modeling → explainability) |
| `app.py` | Streamlit dashboard (risk calculator, probability distribution, feature importance, what-if simulator) |
| `01_eda.py` / `02_modeling.py` | Standalone scripts that generate all figures and the trained model |
| `data/European_Bank.csv` | Source dataset (10,000 customers) |
| `models/best_model.pkl` | Trained production model (Gradient Boosting), pickled as a full sklearn Pipeline |
| `models/all_models.pkl` | All four trained models, for comparison |
| `figures/` | All 16 generated charts (EDA + model evaluation + explainability) |
| `outputs/*.csv`, `outputs/*.json` | Model comparison table, feature importance, permutation importance, EDA summary stats |

## How to run

**Regenerate everything from scratch:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python 01_eda.py
python 02_modeling.py
```

**Launch the dashboard:**
```bash
pip install streamlit plotly joblib
streamlit run app.py
```

## Headline Result

Best model: **Gradient Boosting** — ROC-AUC 0.870, Accuracy 87.1%, on a held-out 20% test set (stratified).

Top churn drivers (permutation importance): **Age**, **Number of Products**, **Geography**, **Balance**, **Engagement × Product interaction**.

Key finding: customers holding 3–4 products churn at 83–100% (vs. 8% for 2 products) — a strong signal worth product-level investigation, not a loyalty effect.
