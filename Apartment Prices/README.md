📊 Dataset Overview
Target Variable: price (Apartment price in EGP)

Initial Shape: 5,578 rows × 11 columns

Features:

Numerical / Parsed Numerical: area_sqm, rooms, bathrooms, floor_level (Ground mapped to 0), year_built, price_per_sqm

Categorical: seller_type, view, payment_method, finishing_type

🛠️ Data Preprocessing & Feature Engineering
Numeric Extraction & Cleaning: Removed commas and non-numeric text/currency units from price, area_sqm, rooms, bathrooms, and price_per_sqm, converting them to float64.

Floor Level Standardization: Replaced 'Ground' with '0' and converted floor_level to integer (int64).

Missing Values Imputation:

Dropped rows with missing target price (6 rows removed, leaving 5,572 rows).

Imputed numerical missing values (rooms, bathrooms, area_sqm, price_per_sqm) using the median (SimpleImputer(strategy='median')).

Imputed categorical missing values (seller_type, view, payment_method) using the most frequent value (SimpleImputer(strategy='most_frequent')).

Encoding & Scaling:

Applied One-Hot Encoding (OneHotEncoder(handle_unknown='ignore')) to categorical columns, expanding the dataframe to 35 columns.

Split data into 80% Train and 20% Test (random_state=42).

Scaled numerical features (area_sqm, rooms, bathrooms, price_per_sqm, floor_level, year_built) using MinMaxScaler (fitted on train and transformed on test).

🤖 Models Evaluated
The pipeline trains and evaluates 15 regression models, logging MAE, MSE, RMSE, and R² on both training and test sets along with actual vs. predicted plots on the first 20 test samples:

Linear Regression, Lasso, Ridge

K-Neighbors Regressor (KNN)

Support Vector Regressor (SVR)

Decision Tree Regressor

Random Forest Regressor

Extra Trees Regressor

Bagging Regressor

AdaBoost Regressor

Gradient Boosting Regressor

HistGradientBoosting Regressor

XGBoost (XGBRegressor)

CatBoost (CatBoostRegressor)

LightGBM (LGBMRegressor)

📦 Requirements
Bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost tqdm
▶️ How to Run
Place Apartments Prices Dataset(Target column price).csv and apartment_price_prediction.ipynb in the same folder.

Open the notebook in Google Colab or Jupyter Notebook.

Run all cells sequentially from top to bottom.