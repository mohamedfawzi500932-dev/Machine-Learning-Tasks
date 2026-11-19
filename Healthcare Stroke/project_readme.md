# Stroke Prediction Machine Learning Pipeline

An exploratory data analysis (EDA) and machine learning preprocessing pipeline built in Python to analyze clinical and demographic factors associated with stroke risk.

---

## 📋 Project Overview

This notebook analyzes the **Healthcare Stroke Dataset** (`healthcare-dataset-stroke-data.csv`), which contains **5,110 patient records** across **12 attributes** (11 features + 1 binary target variable `stroke`). 

The current workflow covers environment setup, data inspection, missing value imputation, feature cleanup, and initial exploratory visualizations.

---

## 📊 Dataset Summary

* **Total Records:** 5,110
* **Duplicate Rows:** 0
* **Target Variable:** `stroke` (`1` = patient had a stroke, `0` = no stroke)
* **Class Distribution:** Highly imbalanced (~4.87% positive stroke cases)

### Features
| Feature | Type | Description |
| :--- | :--- | :--- |
| `id` | Numerical | Unique patient identifier *(dropped during cleaning)* |
| `gender` | Categorical | `"Male"`, `"Female"`, or `"Other"` |
| `age` | Numerical | Age of the patient |
| `hypertension` | Binary | `0` if no hypertension, `1` if patient has hypertension |
| `heart_disease` | Binary | `0` if no heart disease, `1` if patient has heart disease |
| `ever_married` | Categorical | `"No"` or `"Yes"` |
| `work_type` | Categorical | `"children"`, `"Govt_job"`, `"Never_worked"`, `"Private"`, or `"Self-employed"` |
| `Residence_type` | Categorical | `"Rural"` or `"Urban"` |
| `avg_glucose_level` | Numerical | Average glucose level in blood |
| `bmi` | Numerical | Body Mass Index *(contains 201 missing values)* |
| `smoking_status` | Categorical | `"formerly smoked"`, `"never smoked"`, `"smokes"`, or `"Unknown"` |

---

## 🛠️ Current Notebook Workflow

1. **Environment & Library Setup**
   * Installs and imports core data science and gradient boosting libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`).
2. **Data Loading & Inspection**
   * Loads `/content/healthcare-dataset-stroke-data.csv` and inspects data types, summary statistics (`describe()`), duplicate counts, and missing values.
3. **Data Cleaning & Imputation**
   * Imputes the **201 missing `bmi` entries** using the column **median** via `SimpleImputer(strategy='median')`.
   * Drops the non-predictive `id` column.
4. **Exploratory Data Analysis (EDA)**
   * Generates scatter plots exploring relationships across:
     * `stroke` vs. `bmi`
     * `stroke` vs. `avg_glucose_level`
     * `smoking_status` vs. `bmi`
     * `age` vs. `bmi` (grouped by `stroke`)

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost
```

### Running the Notebook
1. Upload `healthcare-dataset-stroke-data.csv` to your working directory (or `/content/` if running in Google Colab).
2. Open the `.ipynb` notebook in Jupyter Notebook, JupyterLab, or Google Colab.
3. Run the cells sequentially from top to bottom.

---

## 🔮 Planned Next Steps

* **Feature Encoding & Scaling:** Apply One-Hot Encoding to categorical variables and standardize numerical features inside a `ColumnTransformer` / `Pipeline`.
* **Train/Test Split & Leakage Prevention:** Split data using stratified sampling (`stratify=y`) before fitting transformers or resamplers.
* **Imbalance Handling:** Evaluate class weighting (`class_weight='balanced'`) or resampling methods (e.g., SMOTE) to address the ~4.9% positive class ratio.
* **Model Training & Evaluation:** Train and compare classifiers (`LogisticRegression`, `RandomForestClassifier`, `XGBClassifier`, `LGBMClassifier`, `CatBoostClassifier`) using **ROC-AUC**, **PR-AUC**, **Recall**, and **F1-score**.