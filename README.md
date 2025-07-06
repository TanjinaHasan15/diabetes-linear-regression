# Diabetes Prediction using Linear Regression

## 🔍 Project Overview
This project uses a machine learning technique — **Linear Regression** — to predict whether a person is likely to have diabetes based on their medical attributes.

Although linear regression is typically used for continuous output, here we round the predictions to either `0` or `1` to interpret them as binary outcomes.



## 📊 Dataset
- Source: [Kaggle - Diabetes Dataset](https://www.kaggle.com/datasets/saurab)
- Total Records: 768
- Features:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
- Target: `Outcome` (0 = No Diabetes, 1 = Diabetes)



## ⚙️ Preprocessing Steps
- Replaced zero values in these columns with **median**:
  - Glucose, BloodPressure, SkinThickness, Insulin, BMI
- Replaced the **first row's Glucose** with the **maximum** value in the Glucose column.
- For records with the **lowest age**, replaced Glucose with the **minimum** value in the Glucose column.



## 🧠 Model
- Algorithm: **Linear Regression**
- Predictions were rounded to the nearest integer (`0` or `1`) to classify as diabetic or not.



## 📈 Evaluation Metrics

| Metric     | Description                                |
|------------|--------------------------------------------|
| Accuracy   | How many predictions were correct overall  |
| Precision  | TP / (TP + FP)                             |
| Recall     | TP / (TP + FN)                             |
| F1 Score   | 2 * (Precision * Recall) / (Precision + Recall) |



## 📊 Example Results

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.75    |
| Precision  | 0.68    |
| Recall     | 0.72    |
| F1 Score   | 0.70    |
| Confusion Matrix | [[80, 20], [19, 35]] (example) |


## 🧾 Files Included

- `diabetes-linear-regression.py`: The main Python script
- `README.md`: This file


