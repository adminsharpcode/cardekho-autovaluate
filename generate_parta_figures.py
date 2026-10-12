"""
Generate Part A Figures matching Submission 8 for full notebook rendering.
"""
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

FIG_DIR = r"c:\Users\ROG\Desktop\Mtech Classwork\Machine Learning\figures"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(r"c:\Users\ROG\Desktop\Mtech Classwork\Machine Learning\Car details v3.csv")
REFERENCE_YEAR = 2020
df['vehicle_age'] = REFERENCE_YEAR - df['year']
df['brand'] = df['name'].apply(lambda x: str(x).split()[0].strip())
df['mileage_val'] = df['mileage'].str.extract(r'([0-9.]+)').astype(float)
df.loc[df['mileage_val'] == 0, 'mileage_val'] = np.nan
df['engine_val'] = df['engine'].str.extract(r'([0-9.]+)').astype(float)
df['max_power_val'] = df['max_power'].str.extract(r'([0-9.]+)').astype(float)
df.loc[df['max_power_val'] == 0, 'max_power_val'] = np.nan

for col in ['mileage_val', 'engine_val', 'max_power_val', 'seats']:
    df[col] = df.groupby('brand')[col].transform(lambda s: s.fillna(s.median()))
    df[col] = df[col].fillna(df[col].median())

df['log_selling_price'] = np.log(df['selling_price'])

FUEL_PALETTE = {'Diesel': '#1f77b4', 'Petrol': '#ff7f0e', 'CNG': '#2ca02c', 'LPG': '#d62728'}
fuel_order = ['Diesel', 'Petrol', 'CNG', 'LPG']

# 1. Price Distribution & Log Transformation
raw_skew = df['selling_price'].skew()
log_skew = df['log_selling_price'].skew()
raw_mean = df['selling_price'].mean()
raw_median = df['selling_price'].median()
log_mean = df['log_selling_price'].mean()
log_median = df['log_selling_price'].median()

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
sns.histplot(df['selling_price'], kde=True, color='#d9534f', bins=50, ax=axes[0, 0])
axes[0, 0].set_title(f'Raw Selling Price Distribution (Skewness = +{raw_skew:.2f})', fontweight='bold')
axes[0, 0].set_xlabel('Selling Price (₹ in Millions)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(raw_mean, color='blue', linestyle='--', label=f'Mean (₹{raw_mean/1e5:.1f} Lakh)')
axes[0, 0].axvline(raw_median, color='black', linestyle='-', label=f'Median (₹{raw_median/1e5:.1f} Lakh)')
axes[0, 0].legend()

sns.histplot(df['log_selling_price'], kde=True, color='#2b5c8f', bins=50, ax=axes[0, 1])
axes[0, 1].set_title(f'Log-Transformed Selling Price Distribution (Skewness = +{log_skew:.2f})', fontweight='bold')
axes[0, 1].set_xlabel('ln(Selling Price)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(log_mean, color='blue', linestyle='--', label=f'Mean ({log_mean:.2f})')
axes[0, 1].axvline(log_median, color='black', linestyle='-', label=f'Median ({log_median:.2f})')
axes[0, 1].legend()

stats.probplot(df['selling_price'], dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Normal Q-Q Plot: Raw Selling Price (Extreme Departure)', fontweight='bold')
axes[1, 0].get_lines()[0].set_color('#d9534f')
axes[1, 0].get_lines()[0].set_markersize(3)
axes[1, 0].get_lines()[1].set_color('black')

stats.probplot(df['log_selling_price'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Normal Q-Q Plot: Log-Transformed Selling Price (Near-Linear Fit)', fontweight='bold')
axes[1, 1].get_lines()[0].set_color('#2b5c8f')
axes[1, 1].get_lines()[0].set_markersize(3)
axes[1, 1].get_lines()[1].set_color('black')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'price_distribution_log.png'), dpi=150)
plt.close()

# 2. Resale Price vs Km Driven
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.scatterplot(data=df[df['km_driven'] <= 300000], x='km_driven', y='selling_price', hue='fuel', palette=FUEL_PALETTE, alpha=0.6, s=25, ax=axes[0])
axes[0].set_title('Resale Price vs. Kilometers Driven (Raw Scale)', fontweight='bold')
axes[0].set_xlabel('Kilometers Driven (km)')
axes[0].set_ylabel('Selling Price (₹ in Millions)')
axes[0].legend(title='Fuel Type', frameon=True)

sns.regplot(data=df[df['km_driven'] <= 300000], x='km_driven', y='log_selling_price', scatter_kws={'alpha': 0.3, 'color': '#2b5c8f', 's': 15}, line_kws={'color': '#d9534f', 'linewidth': 2.5, 'label': 'Depreciation Trendline'}, ax=axes[1])
axes[1].set_title('Log Resale Price vs. Kilometers Driven (Elasticity Curve)', fontweight='bold')
axes[1].set_xlabel('Kilometers Driven (km)')
axes[1].set_ylabel('ln(Selling Price)')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'km_driven_trends.png'), dpi=150)
plt.close()

# 3. Fuel Type Boxplot & Violin Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.boxplot(data=df, x='fuel', y='log_selling_price', order=fuel_order, palette=FUEL_PALETTE, ax=axes[0], showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"})
axes[0].set_title('Log Resale Price Distribution across Fuel Types (Boxplot)', fontweight='bold')
axes[0].set_xlabel('Fuel Type')
axes[0].set_ylabel('ln(Selling Price)')

medians_log = df.groupby('fuel')['log_selling_price'].median().loc[fuel_order]
for i, (fuel, med) in enumerate(medians_log.items()):
    axes[0].text(i, med + 0.15, f"{med:.2f}\n(₹{df[df['fuel']==fuel]['selling_price'].median()/1e5:.1f}L)", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#111111')

sns.violinplot(data=df, x='fuel', y='log_selling_price', order=fuel_order, palette=FUEL_PALETTE, inner='quartile', cut=0, ax=axes[1])
axes[1].set_title('Resale Price Density Estimation across Fuel Types (Violin Plot)', fontweight='bold')
axes[1].set_xlabel('Fuel Type')
axes[1].set_ylabel('ln(Selling Price)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fuel_distribution_plots.png'), dpi=150)
plt.close()

# 4. Pearson & Spearman Heatmaps
corr_features = ['selling_price', 'log_selling_price', 'vehicle_age', 'km_driven', 'mileage_val', 'engine_val', 'max_power_val']
corr_p = df[corr_features].corr(method='pearson')
corr_s = df[corr_features].corr(method='spearman')

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
sns.heatmap(corr_p, annot=True, fmt=".2f", cmap='vlag', vmin=-1, vmax=1, cbar_kws={'label': 'Pearson Correlation (r)'}, ax=axes[0], linewidths=0.8)
axes[0].set_title('Pearson Correlation Heatmap (Linear Association)', fontweight='bold', pad=12)

sns.heatmap(corr_s, annot=True, fmt=".2f", cmap='vlag', vmin=-1, vmax=1, cbar_kws={'label': 'Spearman Rank Correlation (rho)'}, ax=axes[1], linewidths=0.8)
axes[1].set_title('Spearman Rank Correlation Heatmap (Monotonic Association)', fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'correlation_heatmaps.png'), dpi=150)
plt.close()

print("All Part A figures generated successfully in figures/")
