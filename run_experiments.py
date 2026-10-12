"""
Assessment 3 / QP 9: Complete Experimental Pipeline for Parts A, B, and C
Student: Vedant Nimkar | 26MML0045 | MACSE504
Dataset: Car details v3.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Fix Windows console UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Output directory for high-res figures
FIG_DIR = r"c:\Users\ROG\Desktop\Mtech Classwork\Machine Learning\figures"
os.makedirs(FIG_DIR, exist_ok=True)

DATASET_PATH = r"c:\Users\ROG\Desktop\Mtech Classwork\Machine Learning\Car details v3.csv"
print(f"Loading dataset: {DATASET_PATH}")
df_raw = pd.read_csv(DATASET_PATH)
df = df_raw.copy()
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

# ==========================================
# PART A REPLICATION: Preprocessing & Cleaning
# ==========================================
REFERENCE_YEAR = 2020
df['vehicle_age'] = REFERENCE_YEAR - df['year']
df['brand'] = df['name'].apply(lambda x: str(x).split()[0].strip())

# Clean numeric values from strings
df['mileage_val'] = df['mileage'].str.extract(r'([0-9.]+)').astype(float)
df.loc[df['mileage_val'] == 0, 'mileage_val'] = np.nan

df['engine_val'] = df['engine'].str.extract(r'([0-9.]+)').astype(float)
df['max_power_val'] = df['max_power'].str.extract(r'([0-9.]+)').astype(float)
df.loc[df['max_power_val'] == 0, 'max_power_val'] = np.nan

# Group-conditioned median imputation by brand
for col in ['mileage_val', 'engine_val', 'max_power_val', 'seats']:
    df[col] = df.groupby('brand')[col].transform(lambda s: s.fillna(s.median()))
    df[col] = df[col].fillna(df[col].median())

# Target log-transformation
df['log_selling_price'] = np.log(df['selling_price'])
print(f"Part A Preprocessing complete. Missing values remaining: {df[['mileage_val', 'engine_val', 'max_power_val', 'seats']].isnull().sum().sum()}")

# ==========================================
# PART B: Feature Engineering & Model Fitting
# ==========================================
print("\n--- PART B: FEATURE ENGINEERING ---")

# 1. Domain Interaction Terms
# A. Annual driving intensity: km driven per year of age
df['km_per_year'] = df['km_driven'] / (df['vehicle_age'] + 0.5)

# B. Age-to-Mileage ratio
df['age_to_mileage_ratio'] = (df['vehicle_age'] + 0.1) / (df['km_driven'] + 1000)

# C. Specific Power Output: Max power per unit displacement (bhp / CC)
df['power_per_engine_cc'] = df['max_power_val'] / (df['engine_val'] + 1)

# D. Power-to-Seats ratio
df['power_per_seat'] = df['max_power_val'] / df['seats']

# E. Luxury Brand Flag
luxury_brands = {'BMW', 'Mercedes-Benz', 'Audi', 'Volvo', 'Jaguar', 'Land', 'Lexus', 'Porsche'}
df['is_luxury_brand'] = df['brand'].apply(lambda b: 1 if b in luxury_brands else 0)

# Select features for regression
num_features = [
    'vehicle_age', 'km_driven', 'mileage_val', 'engine_val', 'max_power_val',
    'seats', 'km_per_year', 'age_to_mileage_ratio', 'power_per_engine_cc',
    'power_per_seat', 'is_luxury_brand'
]

cat_features = ['fuel', 'seller_type', 'transmission', 'owner']

# One-Hot Encoding on categorical attributes (drop_first=True to prevent dummy variable trap)
df_encoded = pd.get_dummies(df, columns=cat_features, drop_first=True, dtype=float)

# List of all predictor feature names
feature_cols = [c for c in df_encoded.columns if c not in [
    'name', 'year', 'selling_price', 'log_selling_price', 'brand', 'mileage', 'engine', 'max_power', 'torque'
]]

print(f"Total Model Features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

X = df_encoded[feature_cols].values
y = df_encoded['log_selling_price'].values

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

# StandardScaler (Crucial for Ridge and Lasso regularization!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Hyperparameter Tuning via Cross-Validation
print("\n--- 5-FOLD CROSS VALIDATION HYPERPARAMETER TUNING ---")
alphas = np.logspace(-4, 4, 100)
cv_kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Ridge CV
ridge_cv = RidgeCV(alphas=alphas, cv=cv_kfold, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_scaled, y_train)
best_alpha_ridge = ridge_cv.alpha_
print(f"Optimal Alpha for Ridge: {best_alpha_ridge:.6f}")

# Lasso CV
lasso_cv = LassoCV(alphas=alphas, cv=cv_kfold, max_iter=10000, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)
best_alpha_lasso = lasso_cv.alpha_
print(f"Optimal Alpha for Lasso: {best_alpha_lasso:.6f}")

# Fit Base Models
ols_model = LinearRegression()
ols_model.fit(X_train_scaled, y_train)

ridge_model = Ridge(alpha=best_alpha_ridge, random_state=42)
ridge_model.fit(X_train_scaled, y_train)

lasso_model = Lasso(alpha=best_alpha_lasso, max_iter=10000, random_state=42)
lasso_model.fit(X_train_scaled, y_train)

# Regularization Paths (Trace Plots)
ridge_coefs = []
lasso_coefs = []
path_alphas = np.logspace(-3, 3, 50)
for a in path_alphas:
    r = Ridge(alpha=a).fit(X_train_scaled, y_train)
    l = Lasso(alpha=a, max_iter=5000).fit(X_train_scaled, y_train)
    ridge_coefs.append(r.coef_)
    lasso_coefs.append(l.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

# Plot Regularization Paths
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].plot(path_alphas, ridge_coefs)
axes[0].set_xscale('log')
axes[0].axvline(best_alpha_ridge, color='black', linestyle='--', label=f'Optimal Alpha = {best_alpha_ridge:.2f}')
axes[0].set_title('Ridge (L2) Coefficient Shrinkage Path', fontweight='bold', fontsize=13)
axes[0].set_xlabel('Regularization Hyperparameter Alpha (log scale)')
axes[0].set_ylabel('Standardized Coefficient Magnitude')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(path_alphas, lasso_coefs)
axes[1].set_xscale('log')
axes[1].axvline(best_alpha_lasso, color='black', linestyle='--', label=f'Optimal Alpha = {best_alpha_lasso:.4f}')
axes[1].set_title('Lasso (L1) Sparsity & Feature Selection Path', fontweight='bold', fontsize=13)
axes[1].set_xlabel('Regularization Hyperparameter Alpha (log scale)')
axes[1].set_ylabel('Standardized Coefficient Magnitude')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
path_fig_path = os.path.join(FIG_DIR, 'regularization_paths.png')
plt.savefig(path_fig_path, dpi=150)
plt.close()
print(f"Saved: {path_fig_path}")

# ==========================================
# PART C: Evaluation, Visualization & Inference
# ==========================================
print("\n--- PART C: MODEL EVALUATION & INFERENCE ---")

models = {
    'Multiple Linear Regression (OLS)': ols_model,
    f'Ridge Regression (Alpha={best_alpha_ridge:.2f})': ridge_model,
    f'Lasso Regression (Alpha={best_alpha_lasso:.4f})': lasso_model
}

results = []
preds = {}

for name, m in models.items():
    # Predictions on Log Scale
    y_train_pred = m.predict(X_train_scaled)
    y_test_pred = m.predict(X_test_scaled)
    preds[name] = y_test_pred
    
    # Log scale metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    n = len(y_test)
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)
    
    log_mae = mean_absolute_error(y_test, y_test_pred)
    log_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Original Currency Scale Metrics (Rupees)
    price_true = np.exp(y_test)
    price_pred = np.exp(y_test_pred)
    
    rupee_mae = mean_absolute_error(price_true, price_pred)
    rupee_rmse = np.sqrt(mean_squared_error(price_true, price_pred))
    mape = np.mean(np.abs((price_true - price_pred) / price_true)) * 100
    med_ape = np.median(np.abs((price_true - price_pred) / price_true)) * 100
    
    results.append({
        'Model Architecture': name,
        'Train R²': round(train_r2, 4),
        'Test R²': round(test_r2, 4),
        'Adj. Test R²': round(adj_r2, 4),
        'Log MAE': round(log_mae, 4),
        'Log RMSE': round(log_rmse, 4),
        'MAE (₹ Lakhs)': round(rupee_mae / 1e5, 3),
        'RMSE (₹ Lakhs)': round(rupee_rmse / 1e5, 3),
        'MAPE (%)': round(mape, 2),
        'Median APE (%)': round(med_ape, 2)
    })

eval_df = pd.DataFrame(results)
print("\nMODEL PERFORMANCE COMPARISON MATRIX:")
print(eval_df.to_string(index=False))

# Plot Residual Analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

best_pred = preds[f'Ridge Regression (Alpha={best_alpha_ridge:.2f})']
residuals = y_test - best_pred

# 1. Residuals vs Predicted
axes[0, 0].scatter(best_pred, residuals, alpha=0.35, color='#2b5c8f', edgecolors='none', s=25)
axes[0, 0].axhline(0, color='#d9534f', linestyle='--', linewidth=2)
axes[0, 0].set_title('Residuals vs. Predicted Values (Homoscedasticity Test)', fontweight='bold')
axes[0, 0].set_xlabel('Predicted ln(Selling Price)')
axes[0, 0].set_ylabel('Residuals [Actual - Predicted]')
axes[0, 0].grid(True, alpha=0.3)

# 2. Residual Distribution (Histogram & KDE)
sns.histplot(residuals, kde=True, color='#2ca02c', bins=40, ax=axes[0, 1])
axes[0, 1].set_title(f'Residual Error Distribution (Mean = {residuals.mean():.4f}, Std = {residuals.std():.4f})', fontweight='bold')
axes[0, 1].set_xlabel('Residual Error')
axes[0, 1].set_ylabel('Density')
axes[0, 1].grid(True, alpha=0.3)

# 3. Residual Q-Q Plot
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Normal Q-Q Plot of Residuals', fontweight='bold')
axes[1, 0].get_lines()[0].set_color('#2b5c8f')
axes[1, 0].get_lines()[0].set_markersize(3)
axes[1, 0].get_lines()[1].set_color('#d9534f')
axes[1, 0].grid(True, alpha=0.3)

# 4. Actual vs Predicted Values
axes[1, 1].scatter(y_test, best_pred, alpha=0.35, color='#8b5cf6', edgecolors='none', s=25)
min_val = min(y_test.min(), best_pred.min())
max_val = max(y_test.max(), best_pred.max())
axes[1, 1].plot([min_val, max_val], [min_val, max_val], color='#d9534f', linestyle='--', linewidth=2, label='Ideal 45° Line (y = x)')
axes[1, 1].set_title(f'Actual vs. Predicted Resale Price (Test R² = {eval_df.loc[1, "Test R²"]:.4f})', fontweight='bold')
axes[1, 1].set_xlabel('Actual ln(Selling Price)')
axes[1, 1].set_ylabel('Predicted ln(Selling Price)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
resid_fig_path = os.path.join(FIG_DIR, 'residual_diagnostics.png')
plt.savefig(resid_fig_path, dpi=150)
plt.close()
print(f"Saved: {resid_fig_path}")

# Plot Coefficient Magnitude Comparison
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'OLS': ols_model.coef_,
    'Ridge': ridge_model.coef_,
    'Lasso': lasso_model.coef_
})

# Sort by absolute Ridge magnitude
coef_df['Abs_Ridge'] = np.abs(coef_df['Ridge'])
coef_df = coef_df.sort_values(by='Abs_Ridge', ascending=False)

fig, ax = plt.subplots(figsize=(14, 9))
y_pos = np.arange(len(coef_df))
bar_height = 0.28

ax.barh(y_pos - bar_height, coef_df['OLS'], height=bar_height, label='OLS', color='#6baed6', alpha=0.85)
ax.barh(y_pos, coef_df['Ridge'], height=bar_height, label=f'Ridge (α={best_alpha_ridge:.2f})', color='#3182bd', alpha=0.9)
ax.barh(y_pos + bar_height, coef_df['Lasso'], height=bar_height, label=f'Lasso (α={best_alpha_lasso:.4f})', color='#de2d26', alpha=0.9)

ax.set_yticks(y_pos)
ax.set_yticklabels(coef_df['Feature'], fontsize=10, fontweight='500')
ax.invert_yaxis()
ax.axvline(0, color='black', linewidth=1)
ax.set_title('Feature Coefficient Magnitude Comparison across Regularized Regressors', fontweight='bold', fontsize=14)
ax.set_xlabel('Standardized Regression Coefficient (β)', fontsize=12)
ax.legend(loc='lower right', frameon=True, fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
coef_fig_path = os.path.join(FIG_DIR, 'coefficient_magnitudes.png')
plt.savefig(coef_fig_path, dpi=150)
plt.close()
print(f"Saved: {coef_fig_path}")

# Market & Elasticity Inference Calculations
print("\n--- COMMERCIAL & MARKET ELASTICITY INFERENCES ---")
# Feature indices
feat_dict = {f: i for i, f in enumerate(feature_cols)}

# Vehicle age depreciation elasticity:
# In unscaled terms:
unscaled_ols = LinearRegression().fit(X_train, y_train)
unscaled_ridge = Ridge(alpha=best_alpha_ridge).fit(X_train, y_train)

age_idx = feat_dict['vehicle_age']
beta_age_unscaled = unscaled_ridge.coef_[age_idx]
annual_depreciation_pct = (np.exp(beta_age_unscaled) - 1) * 100

print(f"1. Vehicle Age Coefficient: {beta_age_unscaled:.4f}")
print(f"   -> Annual Compound Depreciation Rate: {annual_depreciation_pct:.2f}% per year holding all specs constant.")

# Fuel premium:
fuel_diesel_idx = feat_dict.get('fuel_Diesel', -1)
if fuel_diesel_idx >= 0:
    beta_diesel_unscaled = unscaled_ridge.coef_[fuel_diesel_idx]
    diesel_ceteris_paribus_pct = (np.exp(beta_diesel_unscaled) - 1) * 100
    print(f"2. Diesel vs Petrol/Base Fuel Coefficient: {beta_diesel_unscaled:.4f}")
    print(f"   -> Ceteris Paribus Resale Premium of Diesel: +{diesel_ceteris_paribus_pct:.2f}% over reference fuel.")

# Transmission premium:
trans_auto_idx = feat_dict.get('transmission_Manual', -1)
if trans_auto_idx >= 0:
    beta_manual_unscaled = unscaled_ridge.coef_[trans_auto_idx]
    manual_penalty_pct = (np.exp(beta_manual_unscaled) - 1) * 100
    auto_premium_pct = ((1 / np.exp(beta_manual_unscaled)) - 1) * 100
    print(f"3. Manual Transmission Penalty: {manual_penalty_pct:.2f}% (Automatic Premium: +{auto_premium_pct:.2f}%)")

print("\nAll pipeline experiments completed successfully!")
