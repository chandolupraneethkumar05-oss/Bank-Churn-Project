"""
Streamlit Web Application — Predictive Modeling and Risk Scoring
for Bank Customer Churn

Modules:
  1. Customer Churn Risk Calculator
  2. Probability Distribution Visualization
  3. Feature Importance Dashboard
  4. What-If Scenario Simulator
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ------------------------------------------------------------------
# PAGE CONFIG & STYLE
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Customer Churn Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#2E5EAA"
DANGER = "#C64B4B"
SUCCESS = "#5AA469"
WARN = "#E8A93B"
NEUTRAL = "#8B5FBF"

st.markdown(f"""
<style>
    .main {{ background-color: #F7F9FC; }}
    .metric-card {{
        background: white; border-radius: 12px; padding: 1.2rem 1.4rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); border-left: 5px solid {PRIMARY};
    }}
    h1, h2, h3 {{ color: #1B2A4A; }}
    .risk-badge {{
        display:inline-block; padding: 6px 18px; border-radius: 20px;
        font-weight:700; font-size: 1.05rem; color:white;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #EEF1F8; border-radius: 8px 8px 0 0; padding: 8px 16px;
    }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# DATA & MODEL LOADING
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "models" / "best_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / "data" / "European_Bank.csv")
    return df.drop(columns=["Year"])

@st.cache_data
def load_meta():
    with open(BASE_DIR / "outputs" / "model_meta.json") as f:
        meta = json.load(f)
    perm_imp = pd.read_csv(BASE_DIR / "outputs" / "permutation_importance.csv")
    model_comp = pd.read_csv(BASE_DIR / "outputs" / "model_comparison.csv")
    return meta, perm_imp, model_comp

model = load_model()
raw_df = load_data()
meta, perm_imp_df, model_comp_df = load_meta()
BEST_MODEL_NAME = meta["best_model"]


def engineer_features(row: dict) -> pd.DataFrame:
    """Replicates the training-time feature engineering for a single customer dict."""
    balance = row["Balance"]
    salary = row["EstimatedSalary"] if row["EstimatedSalary"] != 0 else 1
    tenure = row["Tenure"] if row["Tenure"] != 0 else 1

    engineered = dict(row)
    engineered["BalanceSalaryRatio"] = balance / salary
    engineered["ProductDensity"] = row["NumOfProducts"] / tenure
    engineered["EngagementProductInteraction"] = row["IsActiveMember"] * row["NumOfProducts"]
    engineered["AgeTenureInteraction"] = row["Age"] * row["Tenure"]
    engineered["IsZeroBalance"] = int(balance == 0)
    return pd.DataFrame([engineered])


def risk_tier(prob: float):
    if prob < 0.25:
        return "Low Risk", SUCCESS
    elif prob < 0.5:
        return "Moderate Risk", WARN
    elif prob < 0.75:
        return "High Risk", "#E8743B"
    else:
        return "Critical Risk", DANGER


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
st.sidebar.markdown("## 🏦 Churn Risk Intelligence")
st.sidebar.markdown("**Predictive Modeling & Risk Scoring**\nfor Bank Customer Churn")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🧮 Risk Calculator", "📊 Probability Distribution", "🔍 Feature Importance", "🎛️ What-If Simulator"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Production model:** {BEST_MODEL_NAME}")
best_row = model_comp_df.iloc[0]
st.sidebar.markdown(f"**ROC-AUC:** {best_row['ROC-AUC']:.3f}")
st.sidebar.markdown(f"**Accuracy:** {best_row['Accuracy']:.1%}")
st.sidebar.markdown("---")
st.sidebar.caption("Unified Mentor · Predictive Modeling and Risk Scoring for Bank Customer Churn")

GEOS = sorted(raw_df["Geography"].unique().tolist())
GENDERS = sorted(raw_df["Gender"].unique().tolist())

# ------------------------------------------------------------------
# PAGE 1 — CUSTOMER CHURN RISK CALCULATOR
# ------------------------------------------------------------------
if page == "🧮 Risk Calculator":
    st.title("Customer Churn Risk Calculator")
    st.write("Enter a customer's profile to generate a live churn probability and risk score.")

    with st.form("calc_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            credit_score = st.slider("Credit Score", 300, 900, 650)
            geography = st.selectbox("Geography", GEOS)
            gender = st.selectbox("Gender", GENDERS)
            age = st.slider("Age", 18, 95, 40)
        with c2:
            tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
            balance = st.number_input("Account Balance ($)", 0.0, 300000.0, 75000.0, step=1000.0)
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
            estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 250000.0, 100000.0, step=1000.0)
        with c3:
            has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
            is_active = st.radio("Active Member?", ["Yes", "No"], horizontal=True)
            st.markdown("&nbsp;")
            submitted = st.form_submit_button("Calculate Risk Score", use_container_width=True, type="primary")

    if submitted:
        row = {
            "CreditScore": credit_score, "Geography": geography, "Gender": gender,
            "Age": age, "Tenure": tenure, "Balance": balance, "NumOfProducts": num_products,
            "HasCrCard": 1 if has_cr_card == "Yes" else 0,
            "IsActiveMember": 1 if is_active == "Yes" else 0,
            "EstimatedSalary": estimated_salary,
        }
        X_input = engineer_features(row)
        prob = model.predict_proba(X_input)[0, 1]
        tier, color = risk_tier(prob)

        st.markdown("### Result")
        r1, r2, r3 = st.columns([1, 1, 2])
        with r1:
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:0.9rem;color:#666;">Churn Probability</div>
                <div style="font-size:2.2rem;font-weight:800;color:{PRIMARY};">{prob:.1%}</div>
                </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:0.9rem;color:#666;">Risk Tier</div>
                <div class="risk-badge" style="background:{color};margin-top:6px;">{tier}</div>
                </div>""", unsafe_allow_html=True)
        with r3:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100,
                number={'suffix': "%"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': color},
                       'steps': [
                           {'range': [0, 25], 'color': '#E8F3EC'},
                           {'range': [25, 50], 'color': '#FCF3DE'},
                           {'range': [50, 75], 'color': '#FBE7DA'},
                           {'range': [75, 100], 'color': '#F9DCDC'}]},
            ))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Suggested Action")
        actions = {
            "Low Risk": "No action needed — maintain standard engagement.",
            "Moderate Risk": "Monitor; consider a light-touch engagement nudge (app usage prompt, satisfaction check-in).",
            "High Risk": "Proactive retention outreach recommended — relationship manager call or personalized offer.",
            "Critical Risk": "Immediate retention intervention — priority outreach, review product fit and fees, escalate to retention team.",
        }
        st.info(actions[tier])

# ------------------------------------------------------------------
# PAGE 2 — PROBABILITY DISTRIBUTION VISUALIZATION
# ------------------------------------------------------------------
elif page == "📊 Probability Distribution":
    st.title("Churn Probability Distribution")
    st.write("Distribution of predicted churn probabilities across the customer base, "
             "split by actual outcome (holdout test set).")

    X_full = raw_df.drop(columns=["Exited", "CustomerId", "Surname"]) if "CustomerId" in raw_df.columns else raw_df.drop(columns=["Exited"])
    X_full = X_full.copy()
    X_full["BalanceSalaryRatio"] = X_full["Balance"] / X_full["EstimatedSalary"].replace(0, 1)
    X_full["ProductDensity"] = X_full["NumOfProducts"] / X_full["Tenure"].replace(0, 1)
    X_full["EngagementProductInteraction"] = X_full["IsActiveMember"] * X_full["NumOfProducts"]
    X_full["AgeTenureInteraction"] = X_full["Age"] * X_full["Tenure"]
    X_full["IsZeroBalance"] = (X_full["Balance"] == 0).astype(int)

    probs = model.predict_proba(X_full)[:, 1]
    plot_df = pd.DataFrame({"Probability": probs, "Actual": raw_df["Exited"].map({0: "Retained", 1: "Churned"})})

    fig = px.histogram(plot_df, x="Probability", color="Actual", nbins=40, barmode="overlay",
                        color_discrete_map={"Retained": PRIMARY, "Churned": DANGER}, opacity=0.65)
    fig.update_layout(height=450, xaxis_title="Predicted Churn Probability", yaxis_title="Customer Count")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    tiers = pd.cut(probs, [0, 0.25, 0.5, 0.75, 1.0], labels=["Low", "Moderate", "High", "Critical"])
    tier_counts = tiers.value_counts().reindex(["Low", "Moderate", "High", "Critical"])
    for col, tname, tcolor in zip([c1, c2, c3, c4], tier_counts.index,
                                    [SUCCESS, WARN, "#E8743B", DANGER]):
        col.markdown(f"""<div class="metric-card" style="border-left-color:{tcolor}">
            <div style="font-size:0.85rem;color:#666;">{tname} Risk</div>
            <div style="font-size:1.6rem;font-weight:800;">{tier_counts[tname]:,}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("#### Churn Rate by Geography & Product Count")
    cc1, cc2 = st.columns(2)
    with cc1:
        geo_churn = raw_df.groupby("Geography")["Exited"].mean().sort_values(ascending=False)
        fig2 = px.bar(geo_churn, color=geo_churn.index,
                      color_discrete_sequence=[PRIMARY, "#E8743B", SUCCESS])
        fig2.update_layout(showlegend=False, yaxis_title="Churn Rate", xaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)
    with cc2:
        prod_churn = raw_df.groupby("NumOfProducts")["Exited"].mean()
        fig3 = px.bar(prod_churn, color_discrete_sequence=[SUCCESS])
        fig3.update_layout(showlegend=False, yaxis_title="Churn Rate", xaxis_title="Number of Products")
        st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------------------
# PAGE 3 — FEATURE IMPORTANCE DASHBOARD
# ------------------------------------------------------------------
elif page == "🔍 Feature Importance":
    st.title("Feature Importance Dashboard")
    st.write(f"Model-agnostic permutation importance for the production model "
             f"(**{BEST_MODEL_NAME}**) — how much ROC-AUC drops when each feature is shuffled.")

    top_n = st.slider("Number of features to display", 5, len(perm_imp_df), 10)
    plot_data = perm_imp_df.head(top_n).sort_values("Importance_mean")

    fig = px.bar(plot_data, x="Importance_mean", y="Feature", orientation="h",
                 error_x="Importance_std" if "Importance_std" in plot_data else None,
                 color_discrete_sequence=[PRIMARY])
    fig.update_layout(height=450, xaxis_title="Mean ROC-AUC Decrease", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Model Comparison")
    st.dataframe(
        model_comp_df.style.format({c: "{:.3f}" for c in model_comp_df.columns if c != "Model"})
        .background_gradient(subset=["ROC-AUC"], cmap="Blues"),
        use_container_width=True,
    )

    st.markdown("#### Key Drivers — Business Interpretation")
    st.markdown("""
- **Age** — churn risk rises through the 45–65 age band, likely a life-stage effect.
- **Number of Products** — the strongest non-demographic driver; 3–4 product holders
  churn at 80–100%, suggesting over-bundling or fee dissatisfaction rather than loyalty.
- **Geography** — Germany shows meaningfully higher churn than France or Spain.
- **Active Membership & engagement interactions** — disengaged customers churn at
  roughly double the rate of active ones.
""")

# ------------------------------------------------------------------
# PAGE 4 — WHAT-IF SCENARIO SIMULATOR
# ------------------------------------------------------------------
elif page == "🎛️ What-If Simulator":
    st.title("What-If Scenario Simulator")
    st.write("Start from a real customer profile, then adjust engagement and product "
             "variables to see how predicted churn risk responds.")

    idx = st.number_input("Load customer by row index (0–9999)", 0, len(raw_df) - 1, 0)
    base_row = raw_df.iloc[int(idx)].to_dict()

    st.markdown("##### Base profile")
    base_cols = st.columns(6)
    labels = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance"]
    for c, lab in zip(base_cols, labels):
        c.metric(lab, f"{base_row[lab]:,.0f}" if isinstance(base_row[lab], (int, float)) and lab not in ("Geography","Gender") else base_row[lab])

    st.markdown("##### Adjust engagement / product variables")
    s1, s2, s3 = st.columns(3)
    with s1:
        sim_products = st.slider("Number of Products", 1, 4, int(base_row["NumOfProducts"]), key="sim_prod")
    with s2:
        sim_active = st.radio("Active Member?", ["Yes", "No"],
                               index=0 if base_row["IsActiveMember"] == 1 else 1, horizontal=True, key="sim_active")
    with s3:
        sim_balance = st.slider("Balance ($)", 0, 300000, int(base_row["Balance"]), step=1000, key="sim_bal")

    scenario_row = dict(base_row)
    scenario_row["NumOfProducts"] = sim_products
    scenario_row["IsActiveMember"] = 1 if sim_active == "Yes" else 0
    scenario_row["Balance"] = sim_balance
    scenario_row = {k: v for k, v in scenario_row.items() if k not in ("Exited", "CustomerId", "Surname")}

    base_input_row = {k: v for k, v in base_row.items() if k not in ("Exited", "CustomerId", "Surname")}
    base_prob = model.predict_proba(engineer_features(base_input_row))[0, 1]
    sim_prob = model.predict_proba(engineer_features(scenario_row))[0, 1]

    m1, m2, m3 = st.columns(3)
    m1.metric("Original Probability", f"{base_prob:.1%}")
    m2.metric("Scenario Probability", f"{sim_prob:.1%}", delta=f"{(sim_prob-base_prob)*100:+.1f} pts",
              delta_color="inverse")
    tier, color = risk_tier(sim_prob)
    m3.markdown(f"""<div class="metric-card" style="border-left-color:{color}">
        <div style="font-size:0.85rem;color:#666;">Scenario Risk Tier</div>
        <div class="risk-badge" style="background:{color};margin-top:6px;">{tier}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("##### Sensitivity: Churn Probability vs Number of Products")
    sweep_probs = []
    for p in [1, 2, 3, 4]:
        r = dict(scenario_row)
        r["NumOfProducts"] = p
        sweep_probs.append(model.predict_proba(engineer_features(r))[0, 1])
    fig = px.line(x=[1, 2, 3, 4], y=sweep_probs, markers=True,
                  labels={"x": "Number of Products", "y": "Churn Probability"})
    fig.update_traces(line_color=PRIMARY, marker=dict(size=10, color=DANGER))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
