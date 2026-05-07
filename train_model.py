"""
Solo Parent DSS - ML Model Training (Small Dataset Version)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("SOLO PARENT DSS - ML MODEL TRAINING")
print("=" * 70)

# LOAD DATA
print("\n[STEP 1] Loading Dataset...")
df = pd.read_csv('data/solo_parent_dataset.csv')
print(f"  - Records: {len(df)}")
print(f"  - Columns: {len(df.columns)}")
print(f"  - Eligible: {(df['Eligibility'] == 'Eligible').sum()}")
print(f"  - Not Eligible: {(df['Eligibility'] == 'Not Eligible').sum()}")

# PREPROCESSING
print("\n[STEP 2] Data Preprocessing...")

feature_columns = [
    'Age', 'Educational_Attainment', 'Employment_Status',
    'Monthly_Income', 'Number_of_Dependents', 'With_Minor',
    'With_PWD', 'Type_of_Solo_Parent'
]

X = df[feature_columns].copy()
y = (df['Eligibility'] == 'Eligible').astype(int)

print(f"  - Features: {len(feature_columns)}")

# Encode categoricals
label_encoders = {}
categorical_cols = ['Educational_Attainment', 'Employment_Status', 'With_Minor', 'With_PWD', 'Type_of_Solo_Parent']

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print(f"  - Encoded categorical features")

# TRAIN-TEST SPLIT (Without stratification for small datasets)
print("\n[STEP 3] Train-Test Split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"  - Training: {len(X_train)} records (70%)")
print(f"  - Testing: {len(X_test)} records (30%)")

# TRAIN MODELS
print("\n[STEP 4] Training Models...")

# Random Forest
print("  - Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"    Accuracy: {rf_accuracy:.1%}")

# For small datasets, just use Random Forest
best_model = rf_model
best_name = "Random Forest"

# EVALUATE
print("\n[STEP 5] Model Evaluation...")
predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

print(f"\n  Best Model: {best_name}")
print(f"  - Accuracy:  {accuracy:.1%}")
print(f"  - Precision: {precision:.1%}")
print(f"  - Recall:    {recall:.1%}")
print(f"  - F1-Score:  {f1:.1%}")

# Cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=3)
print(f"\n  Cross-Validation Scores: {[f'{s:.1%}' for s in cv_scores]}")
print(f"  Average CV Score: {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")

# SAVE
print("\n[STEP 6] Saving Models...")
joblib.dump(best_model, 'model/solo_parent_model.pkl')
joblib.dump(label_encoders, 'model/encoders.pkl')
joblib.dump(feature_columns, 'model/feature_columns.pkl')
print("  - Model saved")
print("  - Encoders saved")
print("  - Features saved")

# SUMMARY
print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"""
Model Performance:
  Accuracy:  {accuracy:.1%}
  Precision: {precision:.1%}
  Recall:    {recall:.1%}
  F1-Score:  {f1:.1%}

What This Means:
  - The model predicts eligibility with {accuracy:.0%} accuracy
  - It correctly identifies {recall:.0%} of eligible cases
  - False positive rate: {1-precision:.0%}

Ready for deployment!
""")
print("=" * 70)
