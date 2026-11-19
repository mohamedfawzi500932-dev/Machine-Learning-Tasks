# Ames Housing Price Prediction

A machine learning notebook for exploring, cleaning, preprocessing, and evaluating regression models on the **Ames Housing Dataset** (`Ames Housing Dataset.csv`) to predict house `SalePrice`.

---

## Overview

This notebook walks through an end-to-end tabular regression workflow on a dataset of **1,460 rows** and initially **81 columns** (including the target `SalePrice`), ultimately reducing to **286 encoded/scaled feature columns** after feature selection, missing-value imputation, encoding, and scaling.

### Workflow Summary
1. **Setup & Dependencies**: Installs and imports gradient boosting and scientific Python libraries (`xgboost`, `lightgbm`, `catboost`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `tqdm`).
2. **Exploratory Data Analysis (EDA)**: Inspects dataframe shape, column data types, numerical summary statistics, missing values (`7,829` total nulls initially), duplicate rows (`0` duplicates), and categorical cardinality.
3. **Data Cleaning & Imputation**:
   * Drops `PoolQC`, `MiscFeature`, `Alley`, `Fence`, and `Id` (reducing columns from `81` to `76`).
   * Fills missing categorical features (`FireplaceQu`, `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`, `MasVnrType`) and `GarageYrBlt` with `'None'`.
   * Fills `MasVnrArea` missing values with `0` and `Electrical` missing values with its mode.
   * Uses `KNNImputer(n_neighbors=5)` on `['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']`.
4. **Encoding & Train/Test Split**:
   * Applies `OneHotEncoder(handle_unknown='ignore')` across 39 object columns.
   * Applies `LabelEncoder` on remaining object column(s) (`GarageYrBlt`).
   * Splits data into `80%` train / `20%` test (`test_size=0.2, random_state=100, stratify=None`) with `SalePrice` as target.
   * Scales 36 numerical columns (`numcols`) using `StandardScaler` fit on `x_train` and applied to `x_test`.
5. **Model Training & Evaluation**: Evaluates models on `x_test` and `y_test` using **R² Score** and **Mean Absolute Error (MAE)**.

---

## Model Results

Based on the test-set evaluation table (`res`) recorded in the notebook:

| Method | R² Score | MAE | Notes from Notebook Execution |
| :--- | :---: | :---: | :--- |
| **LinearRegression** | `0.805131` | `18663.322662` | Fitted `LinearRegression()` and predicted `y_pred` on `x_test`. |
| **KNN** | `0.827544` | `21180.428082` | `KNeighborsRegressor(n_neighbors=3, metric='minkowski', p=1)`. |
| **Naive Bayes** | `0.827544` | `21180.428082` | Imported `GaussianNB`, but called `model.fit` / `model.predict` on the existing `model` instance and appended `y_pred_test`. |
| **SVR** | `0.827544` | `21180.428082` | Fitted `SVR(C=0.1, kernel='rbf', gamma=1.0)` and assigned `y_pred = model.predict(x_test)`, while appending `y_pred_test` metrics. |
| **DecisionTreeRegressor** | `0.790244` | `25192.469268` | Configured with `max_depth=5`, `min_samples_split=10`, `min_samples_leaf=5`, `min_impurity_decrease=0.01`, `ccp_alpha=0.01`, `random_state=100`, and plotted via `plot_tree`. |

---

## Requirements & Usage

### Prerequisites
* Python 3 (executed in Google Colab with Python 3.13 in the notebook outputs)
* Dataset file: `Ames Housing Dataset.csv`

### Installation
```bash
pip install xgboost lightgbm catboost numpy pandas matplotlib seaborn plotly tqdm scikit-learn
```

### Running the Notebook
1. Place `Ames Housing Dataset.csv` at `/content/Ames Housing Dataset.csv` (or update the path in `pd.read_csv(...)`).
2. Run the cells sequentially from top to bottom to preprocess the data, train the models, and view the `res` comparison dataframe and Decision Tree plot.

---

## Observations & Potential Improvements

* **Variable Reuse in Evaluation Cells**: In the Naive Bayes and SVR sections, `model` and `y_pred_test` reuse variables from earlier steps rather than evaluating `GaussianNB` or the newly predicted `y_pred` from `SVR`. Ensuring each model has its own variable (e.g., `svr_pred = svr_model.predict(x_test)`) will make metric logging distinct per estimator.
* **In-Place Drop Assignment**: `df_clean = df.drop(columns=dropcols, inplace=True)` sets `df_clean` to `None` while mutating `df` in place.
* **Additional Imported Models**: `Lasso`, `Ridge`, `BaggingRegressor`, `AdaBoostRegressor`, `GradientBoostingRegressor`, `HistGradientBoostingRegressor`, `VotingRegressor`, `StackingRegressor`, `XGBRegressor`, `CatBoostRegressor`, and `LGBMRegressor` are imported in the first cell and can be added to `results` for further comparison.