# Adult Census Income Classification

This project presents a comprehensive machine learning pipeline for predicting whether an individual's annual income exceeds **$50,000** (`>50K` vs. `<=50K`) using the classic **Adult Census Income** dataset.

---

## 📊 Dataset Overview

The dataset (`Adult Census Income.csv`) contains **32,561 records** across **15 attributes** (6 numerical and 9 categorical):

* **Demographic Features:** `age`, `sex`, `race`, `marital.status`, `relationship`, `native.country`
* **Education & Employment:** `education`, `education.num`, `workclass`, `occupation`, `hours.per.week`
* **Financial Indicators:** `capital.gain`, `capital.loss`, `fnlwgt` (final sampling weight)
* **Target Variable:** `income` (`<=50K` or `>50K`)

---

## 🧹 Data Preprocessing & Exploratory Analysis

The notebook performs the following preprocessing steps:
1. **Missing Value Detection & Imputation:** Placeholder `"?"` strings in `workclass` (1,836 missing), `occupation` (1,843 missing), and `native.country` (583 missing) are converted to `NaN` and imputed using `SimpleImputer` with the `most_frequent` strategy.
2. **Duplicate Inspection:** Identifies 24 duplicate rows in the dataset.
3. **Exploratory Data Analysis (EDA):** Inspects feature distributions, summary statistics (`df.describe()`), and pairwise feature relationships via `seaborn.pairplot`.

---

## 🤖 Models & Evaluation Metrics

The environment is configured to benchmark a wide suite of classification algorithms and ensemble techniques:

* **Linear & Distance-Based Models:** Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machines (SVC), Gaussian Naive Bayes
* **Tree-Based & Bagging Ensembles:** Decision Tree, Random Forest, Extra Trees
* **Boosting & Gradient Boosting Frameworks:** AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost
* **Meta-Ensembles:** Voting Classifier, Stacking Classifier

**Evaluation Metrics:** Accuracy, Balanced Accuracy, Precision, Recall, F1-Score, ROC-AUC, Average Precision (PR-AUC), Confusion Matrix, and Classification Reports.

---

## 🛠️ Tech Stack & Dependencies

* **Data Manipulation:** `numpy`, `pandas`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
* **Utilities:** `tqdm`

---

## 🚀 Getting Started

1. **Install Dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tqdm xgboost lightgbm catboost
   ```
2. **Dataset Setup:**
   Ensure `Adult Census Income.csv` is placed in your working directory (or update the file path in `pd.read_csv()` if running locally outside of Google Colab).
3. **Run the Notebook:**
   Execute the cells sequentially in Jupyter Notebook or Google Colab to preprocess the data, generate visualizations, and train the classification models.