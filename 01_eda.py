"""
01_eda.py — Senior Data Analyst Exploratory Data Analysis & Statistical Profiling
Predictive Modeling and Risk Scoring for Bank Customer Churn
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Ensure directories exist
FIGDIR = 'figures'
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs('outputs', exist_ok=True)

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 130
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
PALETTE = ['#2E5EAA', '#E8743B', '#5AA469', '#C64B4B', '#8B5FBF']

# Load data
df = pd.read_csv('data/European_Bank.csv')
if 'Year' in df.columns:
    df = df.drop(columns=['Year'])  # constant, non-informative

print("=" * 60)
print(f"Dataset Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"Overall Churn Rate: {df['Exited'].mean():.2%}")
print("=" * 60)

# ============================================================
# 1. Target Distribution
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4.5))
counts = df['Exited'].value_counts().sort_index()
labels = ['Retained (0)', 'Churned (1)']
bars = ax.bar(labels, counts.values, color=[PALETTE[0], PALETTE[3]], width=0.55, edgecolor='black', linewidth=0.8)
for i, v in enumerate(counts.values):
    pct = v / len(df) * 100
    ax.text(i, v + 90, f"{v:,}\n({pct:.1f}%)", ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_title('Customer Target Distribution (Class Imbalance)', fontsize=13, fontweight='bold', pad=12)
ax.set_ylabel('Number of Customers', fontsize=11)
ax.set_ylim(0, counts.max() * 1.18)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/01_target_distribution.png')
plt.close()

# ============================================================
# 2. Churn Rate by Geography
# ============================================================
fig, ax = plt.subplots(figsize=(6.5, 4.5))
geo_stats = df.groupby('Geography').agg(
    total=('Exited', 'count'),
    churned=('Exited', 'sum'),
    churn_rate=('Exited', 'mean')
).sort_values('churn_rate', ascending=False)

bars = ax.bar(geo_stats.index, geo_stats['churn_rate'], color=PALETTE[:3], width=0.5, edgecolor='black', linewidth=0.8)
for i, (geo, row) in enumerate(geo_stats.iterrows()):
    ax.text(i, row['churn_rate'] + 0.008, 
            f"{row['churn_rate']:.1%}\n({int(row['churned']):,}/{int(row['total']):,})", 
            ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.set_title('Churn Rate by Geography (Market Breakdown)', fontsize=13, fontweight='bold', pad=12)
ax.set_ylabel('Churn Rate', fontsize=11)
ax.set_ylim(0, geo_stats['churn_rate'].max() * 1.25)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
plt.tight_layout()
plt.savefig(f'{FIGDIR}/02_churn_by_geography.png')
plt.close()

# ============================================================
# 2b. The German Zero-Balance Paradox (Senior Analyst Discovery)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
geo_zero = df.groupby('Geography').agg(
    total=('Balance', 'count'),
    zero_bal=('Balance', lambda x: (x == 0).sum()),
    zero_pct=('Balance', lambda x: (x == 0).mean())
)
ax1.bar(geo_zero.index, geo_zero['zero_pct'], color=[PALETTE[0], PALETTE[3], PALETTE[2]], width=0.5, edgecolor='black')
for i, v in enumerate(geo_zero['zero_pct']):
    ax1.text(i, v + 0.01, f"{v:.1%}", ha='center', fontweight='bold', fontsize=10)
ax1.set_title('Zero-Balance Account Share by Market', fontsize=12, fontweight='bold')
ax1.set_ylabel('Proportion with Zero Balance')
ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax1.set_ylim(0, 0.6)

# Churn rate for zero vs positive balance by geography
df['HasZeroBalance'] = (df['Balance'] == 0).map({True: 'Zero Balance', False: 'Positive Balance'})
geo_bal_churn = df.groupby(['Geography', 'HasZeroBalance'], observed=False)['Exited'].mean().unstack()
geo_bal_churn.plot(kind='bar', ax=ax2, color=[PALETTE[0], PALETTE[1]], width=0.6, edgecolor='black')
ax2.set_title('Churn Rate: Zero vs Positive Balance', fontsize=12, fontweight='bold')
ax2.set_ylabel('Churn Rate')
ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax2.legend(title='', frameon=True)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/02b_geography_zero_balance_paradox.png')
plt.close()

# ============================================================
# 3. Churn by Gender
# ============================================================
fig, ax = plt.subplots(figsize=(5.5, 4.5))
gen_stats = df.groupby('Gender').agg(
    total=('Exited', 'count'),
    churned=('Exited', 'sum'),
    churn_rate=('Exited', 'mean')
)
ax.bar(gen_stats.index, gen_stats['churn_rate'], color=[PALETTE[4], PALETTE[1]], width=0.5, edgecolor='black', linewidth=0.8)
for i, (gen, row) in enumerate(gen_stats.iterrows()):
    ax.text(i, row['churn_rate'] + 0.006, 
            f"{row['churn_rate']:.1%}\n({int(row['churned']):,}/{int(row['total']):,})", 
            ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.set_title('Churn Rate by Gender', fontsize=13, fontweight='bold', pad=12)
ax.set_ylabel('Churn Rate', fontsize=11)
ax.set_ylim(0, gen_stats['churn_rate'].max() * 1.25)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
plt.tight_layout()
plt.savefig(f'{FIGDIR}/03_churn_by_gender.png')
plt.close()

# ============================================================
# 4. Age Distribution by Churn Status
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 4.8))
sns.kdeplot(df.loc[df['Exited'] == 0, 'Age'], fill=True, color=PALETTE[0], label='Retained (Mean: 37.4 yrs)', ax=ax, alpha=0.45)
sns.kdeplot(df.loc[df['Exited'] == 1, 'Age'], fill=True, color=PALETTE[3], label='Churned (Mean: 44.8 yrs)', ax=ax, alpha=0.45)
ax.axvline(df.loc[df['Exited'] == 0, 'Age'].mean(), color=PALETTE[0], linestyle='--', alpha=0.8)
ax.axvline(df.loc[df['Exited'] == 1, 'Age'].mean(), color=PALETTE[3], linestyle='--', alpha=0.8)
ax.set_title('Age Distribution: Retained vs Churned Customers', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Age (Years)', fontsize=11)
ax.set_ylabel('Kernel Density', fontsize=11)
ax.legend(frameon=True, loc='upper right')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/04_age_distribution.png')
plt.close()

# ============================================================
# 5. Number of Products vs Churn (With Sample Size Annotations)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.8))
prod_stats = df.groupby('NumOfProducts').agg(
    total=('Exited', 'count'),
    churned=('Exited', 'sum'),
    churn_rate=('Exited', 'mean')
)
bars = ax.bar(prod_stats.index.astype(str), prod_stats['churn_rate'], color=PALETTE[2], width=0.55, edgecolor='black', linewidth=0.8)
for i, (prod, row) in enumerate(prod_stats.iterrows()):
    ax.text(i, row['churn_rate'] + 0.02, 
            f"{row['churn_rate']:.1%}\n(N={int(row['total']):,})", 
            ha='center', va='bottom', fontweight='bold', fontsize=9.5)
ax.set_title('Churn Rate by Number of Products Held', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Number of Products Held', fontsize=11)
ax.set_ylabel('Churn Rate', fontsize=11)
ax.set_ylim(0, 1.18)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
plt.tight_layout()
plt.savefig(f'{FIGDIR}/05_churn_by_numproducts.png')
plt.close()

# ============================================================
# 6. Active Member vs Churn
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4.5))
act_stats = df.groupby('IsActiveMember').agg(
    total=('Exited', 'count'),
    churned=('Exited', 'sum'),
    churn_rate=('Exited', 'mean')
)
ax.bar(['Inactive Member', 'Active Member'], act_stats['churn_rate'], color=[PALETTE[3], PALETTE[0]], width=0.5, edgecolor='black', linewidth=0.8)
for i, (act, row) in enumerate(act_stats.iterrows()):
    ax.text(i, row['churn_rate'] + 0.008, 
            f"{row['churn_rate']:.1%}\n({int(row['churned']):,}/{int(row['total']):,})", 
            ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.set_title('Churn Rate: Active vs Inactive Members', fontsize=13, fontweight='bold', pad=12)
ax.set_ylabel('Churn Rate', fontsize=11)
ax.set_ylim(0, act_stats['churn_rate'].max() * 1.25)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
plt.tight_layout()
plt.savefig(f'{FIGDIR}/06_churn_by_active_member.png')
plt.close()

# ============================================================
# 7. Balance Distribution by Churn Status
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 4.8))
sns.boxplot(x='Exited', y='Balance', data=df, hue='Exited', palette=[PALETTE[0], PALETTE[3]], legend=False, ax=ax, width=0.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Retained', 'Churned'], fontsize=11)
ax.set_title('Account Balance Distribution: Retained vs Churned', fontsize=13, fontweight='bold', pad=12)
ax.set_ylabel('Account Balance (€)', fontsize=11)
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
plt.tight_layout()
plt.savefig(f'{FIGDIR}/07_balance_by_churn.png')
plt.close()

# ============================================================
# 8. Correlation Heatmap
# ============================================================
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']
fig, ax = plt.subplots(figsize=(8.5, 7))
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
            cbar_kws={'label': 'Pearson Correlation'}, linewidths=0.5, annot_kws={'fontsize': 9})
ax.set_title('Correlation Matrix of Numerical Features & Target', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/08_correlation_heatmap.png')
plt.close()

# ============================================================
# 9. Credit Score Distribution
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 4.8))
sns.kdeplot(df.loc[df['Exited'] == 0, 'CreditScore'], fill=True, color=PALETTE[0], label='Retained', ax=ax, alpha=0.45)
sns.kdeplot(df.loc[df['Exited'] == 1, 'CreditScore'], fill=True, color=PALETTE[3], label='Churned', ax=ax, alpha=0.45)
ax.set_title('Credit Score Distribution: Retained vs Churned', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Credit Score', fontsize=11)
ax.set_ylabel('Kernel Density', fontsize=11)
ax.legend(frameon=True)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/09_creditscore_distribution.png')
plt.close()

# ============================================================
# Statistical Hypothesis Testing
# ============================================================
# Chi-Square test: Geography vs Exited
contingency_geo = pd.crosstab(df['Geography'], df['Exited'])
chi2_geo, p_geo, _, _ = stats.chi2_contingency(contingency_geo)

# Chi-Square test: NumOfProducts vs Exited
contingency_prod = pd.crosstab(df['NumOfProducts'], df['Exited'])
chi2_prod, p_prod, _, _ = stats.chi2_contingency(contingency_prod)

# Mann-Whitney U test: Age (Retained vs Churned)
u_age, p_age = stats.mannwhitneyu(df.loc[df['Exited'] == 0, 'Age'], df.loc[df['Exited'] == 1, 'Age'])

# Save comprehensive summary statistics for reporting
summary = {
    'total_customers': len(df),
    'churn_rate': float(df['Exited'].mean()),
    'retained_count': int((df['Exited'] == 0).sum()),
    'churned_count': int((df['Exited'] == 1).sum()),
    'churn_by_geography': geo_stats['churn_rate'].to_dict(),
    'geography_zero_balance_pct': geo_zero['zero_pct'].to_dict(),
    'churn_by_gender': gen_stats['churn_rate'].to_dict(),
    'churn_by_active': act_stats['churn_rate'].to_dict(),
    'churn_by_numproducts': prod_stats['churn_rate'].to_dict(),
    'product_counts': prod_stats['total'].to_dict(),
    'avg_age_churned': float(df.loc[df['Exited'] == 1, 'Age'].mean()),
    'avg_age_retained': float(df.loc[df['Exited'] == 0, 'Age'].mean()),
    'avg_balance_churned': float(df.loc[df['Exited'] == 1, 'Balance'].mean()),
    'avg_balance_retained': float(df.loc[df['Exited'] == 0, 'Balance'].mean()),
    'statistical_tests': {
        'chi2_geography': {'chi2_stat': float(chi2_geo), 'p_value': float(p_geo)},
        'chi2_num_products': {'chi2_stat': float(chi2_prod), 'p_value': float(p_prod)},
        'mann_whitney_u_age': {'u_stat': float(u_age), 'p_value': float(p_age)}
    }
}

with open('outputs/eda_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nEDA Completed Successfully! All figures saved to figures/, stats to outputs/eda_summary.json")
