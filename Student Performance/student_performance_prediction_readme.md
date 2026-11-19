# Student Performance Prediction

A machine learning project that analyzes student habits and academic history to predict their **Performance Index** using Linear Regression.

## 📊 Dataset Overview

The project uses the `Student_Performance.csv` dataset containing **10,000 records** across **6 features**:

* **Hours Studied**: Total number of hours spent studying.
* **Previous Scores**: Scores obtained by the student in previous tests.
* **Extracurricular Activities**: Whether the student participates in extracurricular activities (`Yes` / `No`).
* **Sleep Hours**: Average number of hours of sleep per day.
* **Sample Question Papers Practiced**: Number of sample question papers practiced by the student.
* **Performance Index** *(Target Variable)*: A measure of the overall performance of each student (ranging from `10` to `100`).

---

## 🛠️ Technologies & Libraries Used

* **Python 3**
* **Data Manipulation & Analysis**: `pandas`, `numpy`
* **Data Visualization**: `matplotlib`, `seaborn`
* **Machine Learning & Preprocessing**: `scikit-learn` (`SimpleImputer`, `OneHotEncoder`, `StandardScaler`, `train_test_split`, `LinearRegression`)

---

## ⚙️ Workflow & Methodology

1. **Exploratory Data Analysis (EDA)**:
   * Inspected dataset structure, summary statistics, and feature distributions.
   * Visualized relationships between study hours, sleep hours, extracurricular activities, and the Performance Index using scatter plots, histograms, and bar charts.
2. **Data Cleaning & Imputation**:
   * Identified missing values in `Extracurricular Activities` (1,572 missing) and `Sample Question Papers Practiced` (2,856 missing).
   * Applied `SimpleImputer` using the `most_frequent` strategy to fill missing entries.
3. **Feature Encoding & Scaling**:
   * Encoded the categorical column (`Extracurricular Activities`) using `OneHotEncoder`.
   * Split the dataset into training (80%) and testing (20%) sets (`random_state=42`).
   * Standardized numerical features (`Hours Studied`, `Previous Scores`, `Sleep Hours`, `Sample Question Papers Practiced`) using `StandardScaler`.
4. **Model Training & Evaluation**:
   * Trained a `LinearRegression` model on the preprocessed training set.
   * Evaluated model performance on the test set and plotted Actual vs. Predicted values.

---

## 📈 Model Results

The Linear Regression model achieved high predictive accuracy on the test set:

| Metric | Score |
| :--- | :--- |
| **Mean Squared Error (MSE)** | `4.2559` |
| **Mean Absolute Error (MAE)** | `1.6441` |
| **R² Score** | `0.9885` |

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd <repository-folder>
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost
   ```

3. **Run the Notebook**
   * Open the `.ipynb` file in **Google Colab** or **Jupyter Notebook**.
   * Ensure `Student_Performance.csv` is placed in the working directory (or update the file path in `pd.read_csv()`).
   * Run all cells sequentially.