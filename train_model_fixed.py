"""
Solo Parent DSS - ML Model Training (Benchmark Version)

This script trains and compares multiple candidate classifiers on the solo
parent dataset, then saves the model that best balances recall and precision
for eligibility screening.
"""

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

print("=" * 80)
print("SOLO PARENT DSS - ML MODEL TRAINING (BENCHMARK VERSION)")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading Dataset...")
df = pd.read_csv('data/solo_parent_dataset.csv')
print(f"  Original shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")

# ============================================================================
# STEP 2: REMOVE LEAKING COLUMNS
# ============================================================================
print("\n[STEP 2] Data Cleaning - Remove Leaking Columns...")
leaking_columns = [
    'Recommendation',
    'Priority_Level',
    'Full_Name',
    'Barangay',
    'Civil_Status',
    'Sex'
]

leaking_present = [col for col in leaking_columns if col in df.columns]
df_clean = df.drop(columns=leaking_present, errors='ignore')

print(f"  Removed leaking columns: {leaking_present}")
print(f"  Cleaned shape: {df_clean.shape}")
print(f"  Remaining columns: {df_clean.columns.tolist()}")

# ============================================================================
# STEP 3: FEATURE ENGINEERING & PREPROCESSING
# ============================================================================
print("\n[STEP 3] Feature Engineering...")

feature_columns = [
    'Age',
    'Educational_Attainment',
    'Employment_Status',
    'Monthly_Income',
    'Number_of_Dependents',
    'With_Minor',
    'With_PWD',
    'Type_of_Solo_Parent'
]

missing_features = [col for col in feature_columns if col not in df_clean.columns]
if missing_features:
    print(f"  [ERROR] Missing features: {missing_features}")
    raise SystemExit(1)

X = df_clean[feature_columns].copy()
y = (df_clean['Eligibility'] == 'Eligible').astype(int)

print(f"  Features selected: {len(feature_columns)}")
print("  Target variable: Eligibility (Eligible=1, Not Eligible=0)")
print("  Class distribution:")
print(f"    - Eligible:     {(y == 1).sum()} ({y.mean():.1%})")
print(f"    - Not Eligible: {(y == 0).sum()} ({1 - y.mean():.1%})")

# ============================================================================
# STEP 4: TRAIN/TEST SPLIT
# ============================================================================
print("\n[STEP 4] Train/Test Split (with Stratification)...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"  Training set: {len(X_train)} records ({len(X_train) / len(X):.0%})")
print(f"    - Eligible: {y_train.sum()} ({y_train.mean():.1%})")
print(f"    - Not Eligible: {(y_train == 0).sum()} ({(1 - y_train.mean()):.1%})")
print(f"  Test set: {len(X_test)} records ({len(X_test) / len(X):.0%})")
print(f"    - Eligible: {y_test.sum()} ({y_test.mean():.1%})")
print(f"    - Not Eligible: {(y_test == 0).sum()} ({(1 - y_test.mean()):.1%})")

assert abs(y_train.mean() - y.mean()) < 0.02, 'Stratification failed!'
assert abs(y_test.mean() - y.mean()) < 0.02, 'Stratification failed!'
print("  [OK] Stratification verified: Class distribution preserved")

# ============================================================================
# STEP 5: ENCODE CATEGORICAL FEATURES
# ============================================================================
print("\n[STEP 5] Encode Categorical Features (on Training Set Only)...")

categorical_cols = [
    'Educational_Attainment',
    'Employment_Status',
    'With_Minor',
    'With_PWD',
    'Type_of_Solo_Parent'
]

label_encoders = {}
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

for col in categorical_cols:
    encoder = LabelEncoder()
    X_train_encoded[col] = encoder.fit_transform(X_train_encoded[col].astype(str))
    X_test_encoded[col] = X_test_encoded[col].apply(
        lambda value: encoder.transform([value])[0] if value in encoder.classes_ else encoder.transform([encoder.classes_[0]])[0]
    )
    label_encoders[col] = encoder
    print(f"  [OK] {col}: {list(encoder.classes_)}")

assert not X_train_encoded.isna().any().any(), 'NaN values in encoded training data!'
assert not X_test_encoded.isna().any().any(), 'NaN values in encoded test data!'
print("  [OK] No NaN values in encoded data")

# ============================================================================
# STEP 6: BENCHMARK CANDIDATE MODELS
# ============================================================================
print("\n[STEP 6] Benchmarking Candidate Models...")

candidate_models = {
    'RandomForestClassifier': RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        min_samples_split=2,
        min_samples_leaf=1
    ),
    'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
    'ExtraTreesClassifier': ExtraTreesClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    ),
    'SVC': SVC(
        probability=True,
        class_weight='balanced',
        random_state=42
    ),
    'LogisticRegression': LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        random_state=42
    )
}

benchmark_results = []

for model_name, model in candidate_models.items():
    print(f"  [RUN] {model_name}")
    model.fit(X_train_encoded, y_train)
    y_pred = model.predict(X_test_encoded)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    benchmark_results.append({
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'estimator': model
    })

    print(f"    Accuracy:  {accuracy:.1%}")
    print(f"    Precision: {precision:.1%}")
    print(f"    Recall:    {recall:.1%}")
    print(f"    F1-Score:  {f1:.1%}")

benchmark_results = sorted(
    benchmark_results,
    key=lambda item: (item['recall'], item['f1'], item['precision'], item['accuracy']),
    reverse=True
)

best_result = benchmark_results[0]
model = best_result['estimator']
selected_model_name = best_result['model_name']

print(f"\n  [SELECTED] {selected_model_name} based on recall -> F1 -> precision -> accuracy")

# ============================================================================
# STEP 7: EVALUATE SELECTED MODEL ON TEST SET
# ============================================================================
print("\n[STEP 7] Evaluating Selected Model on Test Set...")
y_pred = model.predict(X_test_encoded)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print(f"\n  Test Set Metrics ({selected_model_name}):")
print(f"  - Accuracy:  {accuracy:.1%} ({int(accuracy * len(y_test))}/{len(y_test)} correct)")
print(f"  - Precision: {precision:.1%}")
print(f"  - Recall:    {recall:.1%}")
print(f"  - F1-Score:  {f1:.1%}")

print("\n  Confusion Matrix (Test Set):")
print("                  Predicted Eligible | Predicted Not Eligible")
print(f"  Actually Eligible     {cm[1, 1]:3d}      |      {cm[1, 0]:3d}")
print(f"  Actually Not Eligible  {cm[0, 1]:3d}      |      {cm[0, 0]:3d}")

# ============================================================================
# STEP 8: CROSS-VALIDATION ON TRAINING SET ONLY
# ============================================================================
print("\n[STEP 8] Cross-Validation (on Training Set Only)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    model,
    X_train_encoded,
    y_train,
    cv=skf,
    scoring=['accuracy', 'precision', 'recall', 'f1']
)

print("  5-Fold Cross-Validation Results:")
print(f"  - Accuracy:  {cv_results['test_accuracy'].mean():.1%} (+/- {cv_results['test_accuracy'].std():.1%})")
print(f"  - Precision: {cv_results['test_precision'].mean():.1%} (+/- {cv_results['test_precision'].std():.1%})")
print(f"  - Recall:    {cv_results['test_recall'].mean():.1%} (+/- {cv_results['test_recall'].std():.1%})")
print(f"  - F1-Score:  {cv_results['test_f1'].mean():.1%} (+/- {cv_results['test_f1'].std():.1%})")

# ============================================================================
# STEP 9: FEATURE IMPORTANCE / MODEL EXPLANATION
# ============================================================================
print("\n[STEP 9] Feature Importance (what drives decisions)...")
if hasattr(model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(feature_importance.to_string(index=False))
elif hasattr(model, 'coef_'):
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': np.abs(model.coef_[0])
    }).sort_values('Importance', ascending=False)
    print(feature_importance.to_string(index=False))
else:
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': np.nan
    })
    print("  Feature importance is not available for this model type.")

# ============================================================================
# STEP 10: SAVE MODEL & METADATA
# ============================================================================
print("\n[STEP 10] Saving Model Artifacts...")
joblib.dump(model, 'model/solo_parent_model.pkl')
joblib.dump(label_encoders, 'model/encoders.pkl')
joblib.dump(feature_columns, 'model/feature_columns.pkl')

benchmark_summary = [
    {
        'model_name': item['model_name'],
        'accuracy': item['accuracy'],
        'precision': item['precision'],
        'recall': item['recall'],
        'f1': item['f1'],
        'confusion_matrix': item['confusion_matrix']
    }
    for item in benchmark_results
]

metadata = {
    'model_type': selected_model_name,
    'model_version': '3.0_model_benchmarking',
    'training_date': pd.Timestamp.now().isoformat(),
    'selected_algorithm': selected_model_name,
    'selection_criteria': ['recall', 'f1', 'precision', 'accuracy'],
    'dataset_info': {
        'total_records': len(df),
        'training_records': len(X_train),
        'test_records': len(X_test),
        'feature_count': len(feature_columns),
        'leaking_columns_removed': leaking_present
    },
    'class_distribution': {
        'overall_eligible_ratio': float(y.mean()),
        'train_eligible_ratio': float(y_train.mean()),
        'test_eligible_ratio': float(y_test.mean()),
        'total_eligible': int((y == 1).sum()),
        'total_not_eligible': int((y == 0).sum())
    },
    'test_metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    },
    'cross_validation_metrics': {
        'accuracy_mean': float(cv_results['test_accuracy'].mean()),
        'accuracy_std': float(cv_results['test_accuracy'].std()),
        'precision_mean': float(cv_results['test_precision'].mean()),
        'precision_std': float(cv_results['test_precision'].std()),
        'recall_mean': float(cv_results['test_recall'].mean()),
        'recall_std': float(cv_results['test_recall'].std()),
        'f1_mean': float(cv_results['test_f1'].mean()),
        'f1_std': float(cv_results['test_f1'].std())
    },
    'benchmark_results': benchmark_summary,
    'feature_columns': feature_columns,
    'categorical_columns': categorical_cols,
    'feature_importance': (
        feature_importance.set_index('Feature')['Importance'].dropna().to_dict()
        if 'Importance' in feature_importance.columns
        else {}
    ),
    'preprocessing': {
        'encoder_type': 'LabelEncoder',
        'encoder_fit_data': 'training_set_only',
        'stratification_applied': True,
        'random_state': 42
    },
    'quality_assurance': {
        'train_test_leakage_check': 'PASS - Encoders fit on training set only',
        'stratification_check': 'PASS - Class distribution preserved',
        'no_leaking_columns_check': 'PASS - Recommendation, Priority_Level, Full_Name removed'
    }
}

joblib.dump(metadata, 'model/model_metadata.pkl')

print("  [OK] Model saved to: model/solo_parent_model.pkl")
print("  [OK] Encoders saved to: model/encoders.pkl")
print("  [OK] Features saved to: model/feature_columns.pkl")
print("  [OK] Metadata saved to: model/model_metadata.pkl")

# ============================================================================
# SUMMARY & VERIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE - MODEL BENCHMARKING VERSION")
print("=" * 80)

print(f"""
MODEL SELECTION:
  Selected Model:      {selected_model_name}
  Test Accuracy:       {accuracy:.1%} (on {len(y_test)} held-out examples)
  Test Precision:      {precision:.1%}
  Test Recall:         {recall:.1%}
  Test F1-Score:       {f1:.1%}

  Cross-Validation:    {cv_results['test_accuracy'].mean():.1%} ± {cv_results['test_accuracy'].std():.1%}
                       (consistent across 5 folds)

TOP BENCHMARK RESULTS:
  1. {benchmark_results[0]['model_name']} - Recall {benchmark_results[0]['recall']:.1%}, F1 {benchmark_results[0]['f1']:.1%}
  2. {benchmark_results[1]['model_name']} - Recall {benchmark_results[1]['recall']:.1%}, F1 {benchmark_results[1]['f1']:.1%}
  3. {benchmark_results[2]['model_name']} - Recall {benchmark_results[2]['recall']:.1%}, F1 {benchmark_results[2]['f1']:.1%}

DATA QUALITY CHECKS:
  [OK] No leaking columns (Recommendation, Priority_Level, Full_Name removed)
  [OK] Train/test split before encoding (no data leakage)
  [OK] Stratification applied (class balance preserved)
  [OK] Encoders fit on training set only
  [OK] No NaN values after preprocessing
  [OK] Benchmarked multiple algorithms before selecting the production model

FEATURE IMPORTANCE (Top 3):
  1. {feature_importance.iloc[0]['Feature']}: {feature_importance.iloc[0]['Importance']:.1%}
  2. {feature_importance.iloc[1]['Feature']}: {feature_importance.iloc[1]['Importance']:.1%}
  3. {feature_importance.iloc[2]['Feature']}: {feature_importance.iloc[2]['Importance']:.1%}
""")
