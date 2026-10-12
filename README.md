# Cardekho AutoValuate: Used Car Econometric Valuation Suite

[![Vercel Deployment](https://img.shields.io/badge/Vercel-Live%20App-black?logo=vercel)](https://cardekho-autovaluate.vercel.app)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![Licence](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An econometric valuation engine and regularized regression suite built for the **Machine Learning Lab (MACSE504)** at Vellore Institute of Technology (VIT).

- **Student:** Vedant Nimkar
- **Register Number:** 26MML0045
- **Assigned Task:** Question 3 — Used Car Valuation & Elasticity Analysis using Regularized Regression (Ridge & Lasso)
- **Live Interactive Web Application:** [https://cardekho-autovaluate.vercel.app](https://cardekho-autovaluate.vercel.app)

---

## 🚀 Live Interactive Web Platform
The web application provides client-side, zero-latency regression inference powered by $Z$-score standardized OLS, Ridge, and Lasso parameters:
- **Real-Time Sliders:** Vehicle Age, Odometer Kilometers, Engine Displacement (CC), Maximum Power (BHP).
- **Interactive Toggles:** 32 Marques/Brands, Fuel Types (Petrol, Diesel, CNG, LPG), Transmission (Manual/Automatic), Seller Channels, and Ownership History.
- **Dynamic Econometric Outputs:** Resale Valuation in ₹ Lakhs, 90% Confidence Interval Bands ($\pm 1.645 \cdot \text{RMSE}$), and an 8-Year Compounding Depreciation Curve.
- **Regularization Lab:** Interactive $\alpha$-penalty sliders with live shrinkage tracing and sparsity monitors.
- **Diagnostics Dashboard:** Gauss-Markov residual plots (Residuals vs Fitted, Normal Q-Q, Scale-Location, Actual vs Predicted).
- **100% Mobile & Tablet Responsive:** Verified on mobile viewports (320px, 360px, 393px, 412px, 768px, 1024px, 1440px).

---

## 📊 Dataset & Pipeline Overview
- **Dataset:** `Car details v3.csv` (8,128 verified vehicle listings, 13 attributes).
- **Engineered Features (24 Predictors):**
  - Annual Usage Intensity: $\text{km\_per\_year} = \frac{\text{km\_driven}}{\text{vehicle\_age} + 0.5}$
  - Age-to-Mileage Ratio: $\text{age\_to\_mileage\_ratio} = \frac{\text{vehicle\_age} + 0.1}{\text{km\_driven} + 1000}$
  - Specific Power Density: $\text{power\_per\_engine\_cc} = \frac{\text{max\_power}}{\text{engine\_cc} + 1}$
  - Power per Seat: $\text{power\_per\_seat} = \frac{\text{max\_power}}{\text{seats}}$
  - Quadratic Aging: $\text{age\_squared} = \text{vehicle\_age}^2$
  - Log Odometer: $\text{log\_km\_driven} = \ln(1 + \text{km\_driven})$
  - 3-Tier Brand Equity Stratification (Tier 1 Luxury, Tier 2 Premium, Tier 3 Mass-Market).
  - One-Hot Categorical Encoding with `drop_first=True` to prevent the dummy variable trap.

---

## 📈 Model Performance Benchmark (1,626 Test Instances)

| Architecture | Train $R^2$ | Test $R^2$ | Log MAE | Log RMSE | MAE (₹ Lakhs) | RMSE (₹ Lakhs) | MAPE (%) | MedAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OLS Baseline** | 0.8919 | 0.8827 | 0.2205 | 0.2835 | ₹1.189 L | ₹2.060 L | 23.17% | 17.52% |
| **Ridge ($L_2$, $\alpha^*=8.50$)** | 0.8918 | **0.8829** | **0.2203** | **0.2832** | **₹1.190 L** | **₹2.069 L** | **23.17%** | **17.51%** |
| **Lasso ($L_1$, $\alpha^*=0.0006$)** | 0.8918 | **0.8830** | **0.2202** | **0.2832** | ₹1.192 L | ₹2.075 L | 23.16% | 17.52% |
| **ElasticNet ($\alpha^*=0.0006, l_1=0.90$)** | 0.8918 | **0.8829** | 0.2202 | 0.2832 | ₹1.192 L | ₹2.074 L | 23.16% | 17.57% |
| **Huber Robust Regressor** | 0.8905 | 0.8825 | **0.2176** | 0.2838 | **₹1.173 L** | ₹2.071 L | 23.35% | **17.42%** |

---

## 📌 Econometric Market Inferences
1. **Compounding Annual Depreciation:** Unscaled $\beta_{\text{age}} = -0.1151$, corresponding to a compound value decay of **$10.87\%$ per calendar year**.
2. **Diesel Valuation Premium:** Holding all mechanical variables constant, Diesel passenger cars command an **$18.42\%$ ceteris paribus premium** over petrol cars.
3. **Automatic Transmission Elasticity:** Automatic transmissions command an **$11.32\%$ price premium** over manual transmissions.

---

## 📁 Repository Structure
```
├── 26MML0045_Vedant_ML_MiniProject_Complete_PartABC.ipynb  # Complete Phase 1, 2, 3 Notebook
├── 26MML0045_Vedant_ML_MiniProject_PartA_PartB.ipynb       # Phase 1 & 2 Notebook
├── 26MML0045_Vedant_ML_MiniProject_PartB.ipynb             # Phase 2 Standalone Notebook
├── Car details v3.csv                                      # Verified Dataset (8,128 rows)
├── figures/                                                # Diagnostic Plots & Heatmaps
├── web_app/                                                # Production Web Application
│   ├── index.html                                          # Responsive UI Architecture
│   ├── styles.css                                          # Modern Dark-Glassmorphism CSS
│   ├── app.js                                              # Client-Side ML Inference Engine
│   └── echarts.min.js                                      # Local Bundled ECharts
├── run_massive_pipeline.py                                 # Feature & Evaluation Script
├── model_parameters.json                                   # Trained Coefficients & Scalers
├── submission no.10.pdf                                    # Complete PDF Deliverable
├── Machine Learning Submission 9.pdf                       # Part A + Part B Combined PDF
└── Machine Learning Submission Part B.pdf                  # Standalone Part B PDF
```
