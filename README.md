# 🏠 Real Estate Price Prediction — KNN Regression & OLS Analysis

A machine learning notebook that predicts house prices per unit area using a K-Nearest Neighbors (KNN) Regressor, paired with OLS regression for statistical analysis. The workflow covers full EDA, outlier visualization, skewness testing, dual scaling comparison (Min-Max vs. Standard), k-value tuning across 50 candidates, and 10-fold cross-validation on a Taiwanese real estate dataset of 414 transactions.

---

## 📁 Dataset Requirements

This notebook requires **one CSV file** at the following path:

| File | Path in notebook | Rows | Columns | Description |
|------|-----------------|------|---------|-------------|
| `Real estate.csv` | `../input/real-estate-price-prediction/Real estate.csv` | 414 | 8 | Taiwanese real estate transactions with location and property features |

> **Source:** [Kaggle — Real Estate Price Prediction](https://www.kaggle.com/quantbruce/real-estate-price-prediction)  
> Download and either place the file at the relative Kaggle path above, or update the `pd.read_csv()` call in cell 2 to your local path.

### Dataset Schema (8 columns, 414 rows)

| Column | Type | Description |
|---|---|---|
| `No` | int | Row index (dropped) |
| `X1 transaction date` | float | Transaction year as decimal (e.g., 2012.917) — dropped |
| `X2 house age` | float | Age of house in years (0–43.8) |
| `X3 distance to the nearest MRT station` | float | Distance in meters (23.4–6,488) |
| `X4 number of convenience stores` | int | Count of nearby convenience stores (0–10) |
| `X5 latitude` | float | Geographic latitude |
| `X6 longitude` | float | Geographic longitude |
| `Y house price of unit area` | float | **Target** — price per unit area (10,000 NTD/ping) |

**No null values** in any column.

### Key Statistics

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| House age (yrs) | 17.7 | 11.4 | 0.0 | 43.8 |
| MRT distance (m) | 1,083.9 | 1,262.1 | 23.4 | 6,488.0 |
| Convenience stores | — | — | 0 | 10 |
| **House price** | — | — | — | — |

---

## 🛠️ Dependencies

Install all required packages via pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels scipy
```

| Library | Version (recommended) | Purpose |
|---|---|---|
| `numpy` | ≥ 1.21 | Array operations, k-value range |
| `pandas` | ≥ 1.3 | Data loading, slicing, output comparison table |
| `matplotlib` | ≥ 3.4 | Histograms, scatter plots, k-tuning curve |
| `seaborn` | ≥ 0.11 | Heatmap, barplot, regplot, boxplots |
| `scikit-learn` | ≥ 0.24 | `train_test_split`, `MinMaxScaler`, `StandardScaler`, `KNeighborsRegressor`, `cross_val_score`, `r2_score` |
| `statsmodels` | ≥ 0.12 | OLS regression, QQ plot |
| `scipy` | ≥ 1.7 | `skew()`, `shapiro()` normality test |

**Python version:** 3.7+

---

## ▶️ How to Execute

### Option 1 — Jupyter Notebook (recommended)

```bash
pip install notebook
jupyter notebook real-estate-dataset-analysis.ipynb
```

Run all cells via **Cell → Run All**, or step through with **Shift + Enter**.

### Option 2 — JupyterLab

```bash
pip install jupyterlab
jupyter lab real-estate-dataset-analysis.ipynb
```

### Option 3 — VS Code

Open the `.ipynb` file in VS Code with the **Jupyter extension** installed and click **Run All**.

> ⚠️ Update the `pd.read_csv()` path in cell 2 from the Kaggle-relative path to your local file location before running.

---

## 🔄 Pipeline Overview

```
Load CSV → Drop Unused Cols → EDA (histograms, heatmap, scatter/regression plots) →
Outlier Detection (boxplots) → Split X/y → Statistical Analysis (QQ plot, skewness, Shapiro-Wilk) →
Dual Scaling (MinMax + Standard) → OLS Regression → KNN Regression →
k-Value Tuning (k=1–99 odd) → Cross-Validation → Final Predictions Table
```

1. **Load** `Real estate.csv`; drop `No` (index) and `X1 transaction date`
2. **EDA** — descriptive stats, histograms for all 6 remaining columns, correlation heatmap
3. **Bivariate plots** — barplot (convenience stores vs. price), regplots (house age vs. price; MRT distance vs. price), lineplot (house age vs. MRT distance)
4. **Outlier detection** — side-by-side boxplots for all 6 features
5. **Split** features (`X2`–`X6`) and target (`Y`); 80/20 train/test split (`random_state=1`) → 331 train / 83 test rows
6. **Statistical analysis** — QQ plot on feature matrix, skewness per feature, Shapiro-Wilk normality test on OLS residuals
7. **Dual scaling** — fit both `MinMaxScaler` and `StandardScaler` on training set; transform test set; visualize first feature comparison
8. **OLS regression** — fit on Min-Max scaled training data; print full summary (R², F-statistic, coefficients); plot residuals and residual distribution
9. **KNN Regressor** — initial fit with `k=3`, `p=1` (Manhattan distance), `algorithm='brute'`; score on test set
10. **k-Value sweep** — iterate `k` over all odd values 1–99; record train and test R² scores; plot learning curves
11. **10-fold cross-validation** — on the initial `k=3` model; compute mean CV R²
12. **Prediction table** — side-by-side DataFrame of estimated vs. actual prices for the 83 test samples

---

## 📊 Results

### Correlation Highlights

| Feature Pair | Correlation | Relationship |
|---|---|---|
| MRT distance ↔ Convenience stores | **−0.603** | Strong negative — further from MRT → fewer stores |
| House age ↔ Price | −0.211 | Older houses → lower prices |
| MRT distance ↔ Price | negative | Closer to MRT → higher prices |
| Convenience stores ↔ Price | positive | More stores → higher prices |

### Skewness (raw features)

| Feature | Skew | Direction |
|---|---|---|
| X2 house age | 0.38 | Slight positive |
| X3 MRT distance | **1.88** | Strong positive (right-skewed) |
| X4 convenience stores | 0.15 | Near-symmetric |
| X5 latitude | −0.44 | Slight negative |
| X6 longitude | **−1.22** | Strong negative (left-skewed) |

Data is **not normally distributed** — confirmed by Shapiro-Wilk on OLS residuals (statistic = 0.882, p = 2.9×10⁻¹⁵).

---

### OLS Regression (Min-Max scaled features, no intercept)

| Metric | Value |
|---|---|
| R² (uncentered) | **0.940** |
| Adjusted R² | 0.939 |
| F-statistic | 1,023 |
| Prob (F-statistic) | 7.79×10⁻¹⁹⁷ |

The OLS model explains 94% of variance in house prices. However, the Shapiro-Wilk test confirms residuals are **non-normally distributed**, which violates an OLS assumption and means prediction intervals and p-values should be interpreted cautiously.

---

### KNN Regressor — k-Value Tuning

Selected results from the k=1 to k=99 sweep (odd values only):

| k | Train R² | Test R² |
|---|---|---|
| 1 | 98.40% | 49.11% ← severe overfit |
| 3 | 82.85% | 62.70% |
| 5 | 76.51% | 65.10% |
| 9 | 69.54% | 69.09% |
| 11 | 67.99% | 69.85% |
| 17 | 65.20% | **70.20%** |
| 19 | 64.80% | 71.18% |
| 27 | 62.85% | 72.35% |
| 31 | 62.12% | **72.90%** ← best seen |

> The notebook identifies **k=9 to k=13** as optimal based on convergence; test R² peaks near **k=27–31** at ~72–73% in the sweep output.

### KNN Final Metrics (k=3 model — as evaluated in notebook)

| Metric | Value |
|---|---|
| Test R² (`model4.score`) | **0.630** |
| Test R² (`r2_score`) | 0.630 |
| 10-Fold CV R² (mean) | **0.654** |
| 10-Fold CV R² (range) | 0.341 – 0.935 |

The wide CV range (0.341–0.935) reflects high variance across folds due to the small dataset size (331 training rows), making individual fold results unstable.

### Sample Predictions vs. Actuals (first 5 rows)

| Estimated Price | Actual Price |
|---|---|
| 25.50 | 27.3 |
| 47.57 | 54.4 |
| 23.73 | 22.0 |
| 15.27 | 11.6 |
| 46.20 | 45.4 |

*Prices in 10,000 NTD per ping (Taiwanese unit of area).*

---

## 📂 File Structure

```
project/
│
├── real-estate-dataset-analysis.ipynb          # Main notebook
├── ../input/real-estate-price-prediction/
│   └── Real estate.csv                         # Dataset (Kaggle path)
└── README.md
```

> For local execution, update the path in cell 2 to wherever `Real estate.csv` is saved on your machine.

---

## 📌 Notes

- **Min-Max vs. Standard Scaler comparison** — Min-Max scaling is correctly chosen for KNN (which is distance-based and sensitive to feature magnitude). The notebook notes it gives "more uniform scaling" than Standard Scaler, which is accurate for bounded-range features.
- **OLS without intercept** — `sm.OLS(y, X)` with no constant term is used. This inflates R² (uncentered R² is always higher than centered) and is noted in the summary output. Adding `sm.add_constant(X)` would give a more standard interpretation.
- **k=3 used for CV despite tuning showing higher k is better** — the cross-validation is run on the initial `k=3` model rather than the tuning-identified optimum (~k=27–31). Re-running CV at the optimal k would likely yield a meaningfully higher mean R².
- **Small dataset (414 rows)** — high variance in 10-fold CV (R² ranging 0.34–0.93) is a natural consequence. Ensemble methods or regularized regression (Ridge, Lasso) may generalize more stably on data this small.
- **`X1 transaction date`** is dropped early — encoding it as a cyclical or ordinal feature could add temporal signal, as market conditions shift across the 2012–2013 transaction window.
