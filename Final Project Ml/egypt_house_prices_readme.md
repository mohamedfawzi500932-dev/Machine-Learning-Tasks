# Egypt House Prices Prediction & Analysis

## 📌 Project Overview
This project explores, cleans, and analyzes residential real estate listings across Egypt to understand key pricing drivers and build predictive machine learning models for property valuation. Using a dataset of over 25,000 property listings, the notebook walks through end-to-end data preprocessing, exploratory data analysis (EDA), feature engineering, and regression modeling.

---

## 📊 Dataset Description
The raw dataset contains real estate listings across major Egyptian cities and compounds, capturing property specifications, financial terms, and geographic attributes.

After initial data cleaning and deduplication, the working dataset consists of **25,329 entries** and **12 features**:

| Feature | Description | Data Type |
| :--- | :--- | :--- |
| **`Type`** | Property category (e.g., Apartment, Villa, Twin House, Chalet, Duplex) | Categorical |
| **`Price`** | Listing price in Egyptian Pounds (EGP) *(Target Variable)* | Integer (`int64`) |
| **`Bedrooms`** | Number of bedrooms | Numeric |
| **`Bathrooms`** | Number of bathrooms | Numeric |
| **`Area`** | Total property size in square meters ($m^2$) | Numeric |
| **`Furnished`** | Furnishing status (`Yes`, `No`, or `Unknown`) | Categorical |
| **`Level`** | Floor level (for apartments/duplexes) | Categorical / Numeric |
| **`Compound`** | Name of the residential compound (if applicable) | Categorical |
| **`Payment_Option`** | Available payment method (`Cash`, `Installments`, `Cash or Installment`) | Categorical |
| **`Delivery_Date`** | Expected handover timeline (e.g., `Ready to move`, `Soon`, specific year) | Categorical |
| **`Delivery_Term`** | Finishing status at handover (e.g., `Finished`, `Semi Finished`, `Core & Shell`) | Categorical |
| **`City`** | Standardized city or district name (e.g., New Cairo, Sheikh Zayed, North Coast) | Categorical |

---

## 🛠️ Workflow & Methodology

### 1. Data Cleaning & Preprocessing
* **Standardization:** Cleaned and normalized `City` names to resolve spelling inconsistencies and duplicate location labels.
* **Type Conversion:** Stripped formatting characters and converted `Price` into a clean 64-bit integer (`int64`).
* **Missing Value & Outlier Handling:** Handled null values across compound, level, and furnishing fields, and filtered out invalid or extreme anomalous entries to leave **25,329** reliable records.

### 2. Exploratory Data Analysis (EDA)
* **Price Distributions:** Analyzed overall property price distributions and identified right-skewness driven by luxury villas and coastal properties.
* **Location Impact:** Compared median and average property prices across top Egyptian real estate markets (New Cairo, Sheikh Zayed, 6th of October, New Administrative Capital, North Coast).
* **Property Specs vs. Price:** Visualized correlations between `Area`, `Bedrooms`, `Bathrooms`, `Delivery_Term`, and listing `Price`.

### 3. Feature Engineering & Encoding
* Encoded categorical variables (`City`, `Type`, `Furnished`, `Payment_Option`, `Delivery_Term`) using One-Hot / Target / Ordinal encoding techniques.
* Engineered domain-specific features (such as price per square meter for analysis and room-to-area ratios).
* Applied feature scaling where required for distance- and gradient-based algorithms.

### 4. Model Building & Evaluation
Trained and evaluated multiple regression models to predict property prices:
* **Baseline Model:** Linear Regression / Ridge / Lasso
* **Tree-Based Ensembles:** Random Forest Regressor, Gradient Boosting, XGBoost / LightGBM
* **Evaluation Metrics:** Assessed performance using **MAE** (Mean Absolute Error), **RMSE** (Root Mean Squared Error), and **$R^2$ Score**.

---

## 💻 Tech Stack & Requirements
* **Python 3.8+**
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`, `plotly`
* **Machine Learning:** `scikit-learn`, `xgboost` / `lightgbm`
* **Environment:** Jupyter Notebook / Google Colab

---

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/egypt-house-prices.git
   cd egypt-house-prices
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
   ```

3. **Run the notebook:**
   ```bash
   jupyter notebook
   ```
   Open the `.ipynb` file and run the cells sequentially from data loading to model evaluation.

---

## 🔮 Future Improvements
* **Geospatial Enrichment:** Incorporate latitude/longitude coordinates and proximity to key landmarks (schools, highways, metro lines).
* **Inflation & Temporal Adjustment:** Adjust historical listing prices for EGP inflation rates and currency fluctuations.
* **Hyperparameter Tuning:** Perform Bayesian optimization (e.g., Optuna) on top-performing gradient boosting models.
* **Deployment:** Wrap the trained pipeline into a Streamlit or FastAPI web app for real-time price estimation.