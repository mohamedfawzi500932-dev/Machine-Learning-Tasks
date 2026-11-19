📊 Dataset Overview
Target Variable: Diabetes outcome (0 = No Diabetes, 1 = Diabetes)

Features: Clinical & medical attributes (such as Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age — update according to your exact dataset columns).

🛠️ Data Preprocessing & Workflow
Exploratory Data Analysis (EDA): Inspected dataset shape, data types, summary statistics, missing values, and class distribution.

Data Cleaning: Handled missing or invalid zero values in clinical features where applicable.

Train / Test Split: Split the dataset into training and testing sets (e.g., 80% train / 20% test).

Feature Scaling: Standardized or normalized numerical features (e.g., StandardScaler or MinMaxScaler) to improve distance- and gradient-based models.

🤖 Models Evaluated
Trained and compared multiple classification algorithms on the dataset, including:

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVC)

Decision Tree Classifier

Random Forest Classifier

Gradient Boosting / XGBoost / LightGBM / CatBoost (as included in the notebook)

Evaluated using classification metrics such as Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

📦 Requirements
Bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost
▶️ How to Run
Place your dataset .csv file and Diabetes_Prediction_All_Model_Mohamed_Fawzi_.ipynb in the same folder.

Open the notebook in Google Colab or Jupyter Notebook.

Run all cells sequentially from top to bottom.