"""
Massive Feature Engineering, VIF, ElasticNet, Learning Curves & Regularization Lab
Student: Vedant Nimkar | 26MML0045 | MACSE504
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, RidgeCV, LassoCV,
    ElasticNet, ElasticNetCV, HuberRegressor
)
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

WORKSPACE = r"c:\Users\ROG\Desktop\Mtech Classwork\Machine Learning"
FIG_DIR = os.path.join(WORKSPACE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(WORKSPACE, "Car details v3.csv"))
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

# Domain Engineering
df['km_per_year'] = df['km_driven'] / (df['vehicle_age'] + 0.5)
df['age_to_mileage_ratio'] = (df['vehicle_age'] + 0.1) / (df['km_driven'] + 1000)
df['power_per_engine_cc'] = df['max_power_val'] / (df['engine_val'] + 1)
df['power_per_seat'] = df['max_power_val'] / df['seats']

# Quadratic terms
df['age_squared'] = df['vehicle_age'] ** 2
df['log_km_driven'] = np.log1p(df['km_driven'])

# 3-Tier Brand Categorization
tier1_luxury = {'BMW', 'Mercedes-Benz', 'Audi', 'Volvo', 'Jaguar', 'Land', 'Lexus', 'Porsche'}
tier2_premium = {'Toyota', 'Honda', 'Skoda', 'Volkswagen', 'Jeep', 'Kia', 'MG'}
df['brand_tier_luxury'] = df['brand'].apply(lambda b: 1 if b in tier1_luxury else 0)
df['brand_tier_premium'] = df['brand'].apply(lambda b: 1 if b in tier2_premium else 0)

cat_cols = ['fuel', 'seller_type', 'transmission', 'owner']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

model_features = [c for c in df_encoded.columns if c not in [
    'name', 'year', 'selling_price', 'log_selling_price', 'brand', 'mileage', 'engine', 'max_power', 'torque'
]]

X = df_encoded[model_features].values
y = df_encoded['log_selling_price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Variance Inflation Factor (VIF) Calculation
from numpy.linalg import inv
# Correlation matrix condition number
corr_mat = np.corrcoef(X_train, rowvar=False)
vif_diag = np.diag(inv(corr_mat))
vif_df = pd.DataFrame({'Feature': model_features, 'VIF': vif_diag}).sort_values(by='VIF', ascending=False)
print("Top 10 Features with High Multicollinearity (VIF > 5):")
print(vif_df.head(10).to_string(index=False))

# 2. Mutual Information Regression
mi_scores = mutual_info_regression(X_train_scaled, y_train, random_state=42)
mi_df = pd.DataFrame({'Feature': model_features, 'Mutual_Info': mi_scores}).sort_values(by='Mutual_Info', ascending=False)

fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(data=mi_df.head(12), x='Mutual_Info', y='Feature', palette='crest', ax=ax)
ax.set_title('Top 12 Features Ranked by Mutual Information with Log Price', fontweight='bold', fontsize=13)
ax.set_xlabel('Mutual Information Dependency Score')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'mutual_information_ranking.png'), dpi=150)
plt.close()

# 3. Model Tuning (Ridge, Lasso, ElasticNet, Huber)
alphas = np.logspace(-4, 4, 100)
cv_split = KFold(n_splits=5, shuffle=True, random_state=42)

ridge_cv = RidgeCV(alphas=alphas, cv=cv_split, scoring='neg_mean_squared_error').fit(X_train_scaled, y_train)
lasso_cv = LassoCV(alphas=alphas, cv=cv_split, max_iter=10000, random_state=42).fit(X_train_scaled, y_train)
elastic_cv = ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.5, 0.7, 0.9], cv=cv_split, max_iter=10000, random_state=42).fit(X_train_scaled, y_train)

best_alpha_ridge = ridge_cv.alpha_
best_alpha_lasso = lasso_cv.alpha_
best_alpha_elastic = elastic_cv.alpha_
best_l1_ratio = elastic_cv.l1_ratio_

ols = LinearRegression().fit(X_train_scaled, y_train)
ridge = Ridge(alpha=best_alpha_ridge).fit(X_train_scaled, y_train)
lasso = Lasso(alpha=best_alpha_lasso, max_iter=10000).fit(X_train_scaled, y_train)
elastic = ElasticNet(alpha=best_alpha_elastic, l1_ratio=best_l1_ratio, max_iter=10000).fit(X_train_scaled, y_train)
huber = HuberRegressor(max_iter=1000).fit(X_train_scaled, y_train)

print(f"Optimal Alpha Ridge: {best_alpha_ridge:.4f}")
print(f"Optimal Alpha Lasso: {best_alpha_lasso:.4f}")
print(f"Optimal ElasticNet: alpha={best_alpha_elastic:.4f}, l1_ratio={best_l1_ratio}")

# 4. Learning Curves: Diagnosing Bias-Variance Tradeoff
train_sizes, train_scores, test_scores = learning_curve(
    ridge, X_train_scaled, y_train, cv=cv_split,
    train_sizes=np.linspace(0.1, 1.0, 8), scoring='r2', random_state=42
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='#3b82f6', label='Training R² Score')
ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='#10b981', label='Cross-Validation R² Score')
ax.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.15, color='#3b82f6')
ax.fill_between(train_sizes, np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                np.mean(test_scores, axis=1) + np.std(test_scores, axis=1), alpha=0.15, color='#10b981')
ax.set_title('Learning Curves for Ridge Regression (Generalization Dynamics)', fontweight='bold', fontsize=13)
ax.set_xlabel('Training Sample Size')
ax.set_ylabel('R² Score')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'learning_curves.png'), dpi=150)
plt.close()

# 5. Multi-Model Benchmark Comparison Table
all_models = {
    'Multiple Linear Regression (OLS)': ols,
    f'Ridge Regression (α={best_alpha_ridge:.2f}) [L2 Optimal]': ridge,
    f'Lasso Regression (α={best_alpha_lasso:.4f}) [L1 Sparse]': lasso,
    f'ElasticNet (α={best_alpha_elastic:.4f}, l1={best_l1_ratio})': elastic,
    'Huber Robust Regressor [Outlier Robust]': huber
}

bench_rows = []
for name, m in all_models.items():
    tr_r2 = r2_score(y_train, m.predict(X_train_scaled))
    te_r2 = r2_score(y_test, m.predict(X_test_scaled))
    log_mae = mean_absolute_error(y_test, m.predict(X_test_scaled))
    log_rmse = np.sqrt(mean_squared_error(y_test, m.predict(X_test_scaled)))
    
    p_true = np.exp(y_test)
    p_pred = np.exp(m.predict(X_test_scaled))
    inr_mae = mean_absolute_error(p_true, p_pred)
    inr_rmse = np.sqrt(mean_squared_error(p_true, p_pred))
    mape = np.mean(np.abs((p_true - p_pred) / p_true)) * 100
    med_ape = np.median(np.abs((p_true - p_pred) / p_true)) * 100
    
    bench_rows.append({
        'Model': name,
        'Train R²': round(tr_r2, 4),
        'Test R²': round(te_r2, 4),
        'Log MAE': round(log_mae, 4),
        'Log RMSE': round(log_rmse, 4),
        'MAE (₹ Lakhs)': round(inr_mae / 1e5, 3),
        'RMSE (₹ Lakhs)': round(inr_rmse / 1e5, 3),
        'MAPE (%)': round(mape, 2),
        'MedAPE (%)': round(med_ape, 2)
    })

bench_df = pd.DataFrame(bench_rows)
print("\nCOMPREHENSIVE 5-MODEL BENCHMARK MATRIX:")
print(bench_df.to_string(index=False))

print("\nMassive Part B & C experiments successfully executed!")
