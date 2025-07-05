import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("diabetes.csv")

# Columns where 0 is invalid and should be replaced with the column median
replace_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in replace_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

# Step 1: Replace first row's Glucose with the maximum Glucose value
df.loc[0, 'Glucose'] = df['Glucose'].max()

# Step 2: Replace glucose values for lowest age with minimum glucose
min_age = df['Age'].min()
min_glucose = df['Glucose'].min()
df.loc[df['Age'] == min_age, 'Glucose'] = min_glucose

# Features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and round to 0 or 1
y_pred = model.predict(X_test)
y_pred_rounded = np.round(y_pred).astype(int)

# Evaluation
acc = accuracy_score(y_test, y_pred_rounded)
cm = confusion_matrix(y_test, y_pred_rounded)
precision = precision_score(y_test, y_pred_rounded)
recall = recall_score(y_test, y_pred_rounded)
f1 = f1_score(y_test, y_pred_rounded)

# Results
print("==== Evaluation Metrics ====")
print("Accuracy:", round(acc, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("Confusion Matrix:\n", cm)
