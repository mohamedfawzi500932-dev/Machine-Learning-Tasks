# 🏦 Bank Marketing Term Deposit Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?logo=lightgbm&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?logo=catboost&logoColor=black)

## 📌 Project Overview
This project focuses on building a supervised machine learning classification pipeline to predict whether a bank client will subscribe to a **term deposit** (`deposit`: `yes` / `no`) based on direct marketing campaign data (phone calls). By accurately identifying potential subscribers, banks can optimize their marketing resources and improve campaign conversion rates.

---

## 📊 Dataset Description
The dataset (`bank.csv`) initially contains **8,912 records** and **17 features** spanning client demographics, financial status, and marketing campaign interactions:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `age` | Numerical | Age of the client |
| `job` | Categorical | Type of job (e.g., admin, technician, management, services) |
| `marital` | Categorical | Marital status (`married`, `single`, `divorced`) |
| `education` | Categorical | Education level (`primary`, `secondary`, `tertiary`, `unknown`) |
| `default` | Categorical | Has credit in default? (`yes`, `no`) |
| `balance` | Numerical | Average yearly balance (in euros) |
| `housing` | Categorical | Has a housing loan? (`yes`, `no`) |
| `loan` | Categorical | Has a personal loan? (`yes`, `no`) |
| `contact` | Categorical | Contact communication type (`cellular`, `telephone`, `unknown`) |
| `day` | Numerical | Last contact day of the month |
| `month` | Categorical | Last contact month of the year |
| `duration` | Numerical | Last contact duration, in seconds |
| `campaign` | Numerical | Number of contacts performed during this campaign |
| `pdays` | Numerical | Days passed after client was last contacted (`-1` means not previously contacted) |
| `previous` | Numerical | Number of contacts performed before this campaign |
| `poutcome` | Categorical | Outcome of the previous marketing campaign (`success`, `failure`, `other`, `unknown`) |
| **`deposit`** | **Target** | **Has the client subscribed to a term deposit? (`yes`, `no`)** |

---

## 🛠️ Workflow & Methodology

### 1. Exploratory Data Analysis (EDA) & Data Inspection
* Checked dataset dimensions (`8,912 rows × 17 columns`) and verified **0 duplicate rows**.
* Identified missing values across multiple numerical and categorical columns (`age`, `job`, `marital`, `education`, `balance`, `housing`, `loan`, `contact`, `duration`, `deposit`).

### 2. Data Cleaning & Preprocessing
* **Categorical Imputation:** Handled missing values in categorical columns (`job`, `marital`, `education`, `housing`, `loan`, `contact`) using `SimpleImputer(strategy='most_frequent')`.
* **Numerical Imputation:** Handled missing values in numerical columns (`age`, `balance`, `duration`) using `SimpleImputer(strategy='median')` to remain robust against outliers.
* **Target Cleaning:** Dropped rows with missing target labels (`deposit`), resulting in a clean dataset of **8,903 records**.

### 3. Feature Engineering & Transformation *(In Progress / Next Steps)*
* Encoding categorical features (`OneHotEncoder` / `OrdinalEncoder`).
* Scaling numerical features using `MinMaxScaler` / `StandardScaler`.
* Handling special values in `pdays` (`-1` indicator) and evaluating feature importance (e.g., impact of `duration`).

### 4. Modeling & Evaluation *(Planned)*
The environment is configured to train and benchmark multiple classification algorithms:
* **Linear & Distance-Based Models:** `LogisticRegression`, `KNeighborsClassifier`, `SVC`
* **Tree-Based & Ensemble Models:** `DecisionTreeClassifier`, `RandomForestClassifier`, `AdaBoostClassifier`, `GradientBoostingClassifier`
* **Gradient Boosting Frameworks:** `XGBClassifier`, `LGBMClassifier`, `CatBoostClassifier`
* **Meta-Ensembles:** `VotingClassifier`, `StackingClassifier`
* **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost
```

### Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bank-marketing-prediction.git
   cd bank-marketing-prediction
   ```
2. Place `bank.csv` in the project directory (or update the dataset path in the notebook).
3. Launch Jupyter Notebook and run all cells:
   ```bash
   jupyter notebook
   ```

---

## 📂 Project Structure
```text
├── bank.csv                  # Raw dataset
├── bank_marketing.ipynb      # Main Jupyter Notebook (EDA, Preprocessing, Modeling)
└── README.md                 # Project documentation
```

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.