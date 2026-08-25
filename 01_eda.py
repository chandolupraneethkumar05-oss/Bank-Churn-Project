"""
01_eda.py — Exploratory Data Analysis
Predictive Modeling and Risk Scoring for Bank Customer Churn
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120
PALETTE = ['#2E5EAA', '#E8743B', '#5AA469', '#C64B4B', '#8B5FBF']
FIGDIR = 'figures'

df = pd.read_csv('data/European_Bank.csv')
df = df.drop(columns=['Year'])  # constant, non-informative

print("Shape:", df.shape)
print(df['Exited'].value_counts(normalize=True))

# ---------- 1. Target distribution ----------
fig, ax = plt.subplots(figsize=(5, 4))
counts = df['Exited'].value_counts().sort_index()
labels = ['Retained', 'Churned']
ax.bar(labels, counts.values, color=[PALETTE[0], PALETTE[3]])
for i, v in enumerate(counts.values):
    ax.text(i, v + 50, f"{v}\n({v/len(df):.1%})", ha='center', fontweight='bold')
ax.set_title('Customer Churn Distribution', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Customers')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/01_target_distribution.png')
plt.close()

# ---------- 2. Churn by Geography ----------
fig, ax = plt.subplots(figsize=(6, 4))
geo_churn = df.groupby('Geography')['Exited'].mean().sort_values(ascending=False)
ax.bar(geo_churn.index, geo_churn.values, color=PALETTE[:3])
for i, v in enumerate(geo_churn.values):
    ax.text(i, v + 0.005, f"{v:.1%}", ha='center', fontweight='bold')
ax.set_title('Churn Rate by Geography', fontsize=13, fontweight='bold')
ax.set_ylabel('Churn Rate')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/02_churn_by_geography.png')
plt.close()

# ---------- 3. Churn by Gender ----------
fig, ax = plt.subplots(figsize=(5, 4))
gen_churn = df.groupby('Gender')['Exited'].mean()
ax.bar(gen_churn.index, gen_churn.values, color=[PALETTE[4], PALETTE[1]])
for i, v in enumerate(gen_churn.values):
    ax.text(i, v + 0.005, f"{v:.1%}", ha='center', fontweight='bold')
ax.set_title('Churn Rate by Gender', fontsize=13, fontweight='bold')
ax.set_ylabel('Churn Rate')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/03_churn_by_gender.png')
plt.close()

# ---------- 4. Age distribution by churn ----------
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.kdeplot(df.loc[df.Exited == 0, 'Age'], fill=True, color=PALETTE[0], label='Retained', ax=ax, alpha=0.5)
sns.kdeplot(df.loc[df.Exited == 1, 'Age'], fill=True, color=PALETTE[3], label='Churned', ax=ax, alpha=0.5)
ax.set_title('Age Distribution: Retained vs Churned', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGDIR}/04_age_distribution.png')
plt.close()

# ---------- 5. Number of Products vs churn ----------
fig, ax = plt.subplots(figsize=(6, 4))
prod_churn = df.groupby('NumOfProducts')['Exited'].mean()
ax.bar(prod_churn.index.astype(str), prod_churn.values, color=PALETTE[2])
for i, v in enumerate(prod_churn.values):
    ax.text(i, v + 0.01, f"{v:.1%}", ha='center', fontweight='bold', fontsize=9)
ax.set_title('Churn Rate by Number of Products Held', fontsize=13, fontweight='bold')
ax.set_xlabel('Number of Products')
ax.set_ylabel('Churn Rate')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/05_churn_by_numproducts.png')
plt.close()

# ---------- 6. Active member vs churn ----------
fig, ax = plt.subplots(figsize=(5, 4))
act_churn = df.groupby('IsActiveMember')['Exited'].mean()
ax.bar(['Inactive', 'Active'], act_churn.values, color=[PALETTE[3], PALETTE[0]])
for i, v in enumerate(act_churn.values):
    ax.text(i, v + 0.005, f"{v:.1%}", ha='center', fontweight='bold')
ax.set_title('Churn Rate: Active vs Inactive Members', fontsize=13, fontweight='bold')
ax.set_ylabel('Churn Rate')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/06_churn_by_active_member.png')
plt.close()

# ---------- 7. Balance distribution by churn ----------
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.boxplot(x='Exited', y='Balance', data=df, palette=[PALETTE[0], PALETTE[3]], ax=ax)
ax.set_xticklabels(['Retained', 'Churned'])
ax.set_title('Account Balance: Retained vs Churned', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/07_balance_by_churn.png')
plt.close()

# ---------- 8. Correlation heatmap ----------
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']
fig, ax = plt.subplots(figsize=(8, 6.5))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
            cbar_kws={'label': 'Correlation'})
ax.set_title('Correlation Matrix — Numerical Features', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/08_correlation_heatmap.png')
plt.close()

# ---------- 9. Credit Score distribution ----------
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.kdeplot(df.loc[df.Exited == 0, 'CreditScore'], fill=True, color=PALETTE[0], label='Retained', ax=ax, alpha=0.5)
sns.kdeplot(df.loc[df.Exited == 1, 'CreditScore'], fill=True, color=PALETTE[3], label='Churned', ax=ax, alpha=0.5)
ax.set_title('Credit Score Distribution: Retained vs Churned', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGDIR}/09_creditscore_distribution.png')
plt.close()

print("EDA complete. Figures saved to figures/")

# Save summary stats for reporting
summary = {
    'total_customers': len(df),
    'churn_rate': df['Exited'].mean(),
    'churn_by_geography': geo_churn.to_dict(),
    'churn_by_gender': gen_churn.to_dict(),
    'churn_by_active': act_churn.to_dict(),
    'churn_by_numproducts': prod_churn.to_dict(),
    'avg_age_churned': df.loc[df.Exited == 1, 'Age'].mean(),
    'avg_age_retained': df.loc[df.Exited == 0, 'Age'].mean(),
    'avg_balance_churned': df.loc[df.Exited == 1, 'Balance'].mean(),
    'avg_balance_retained': df.loc[df.Exited == 0, 'Balance'].mean(),
}
import json
with open('outputs/eda_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(json.dumps(summary, indent=2, default=str))
