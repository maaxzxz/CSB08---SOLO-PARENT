"""
Solo Parent DSS - ML Model Training (Primary Comparison: DT / RF / LogReg)

This is the PRIMARY training script, matching the methodology described in
Chapter 3 ("Algorithm Evaluation Procedure" and "Phase 5: Model Building"):
Decision Tree, Random Forest, and Logistic Regression are trained and tuned
on the same dataset using modular preprocessing pipelines, evaluated head to
head, and the best-performing model is calibrated and saved as the
production pipeline.

For the supplementary comparison against GradientBoostingClassifier,
ExtraTreesClassifier, and SVC (outside this study's documented algorithm
scope), see appendix_algorithm_comparison.py. That script is NOT wired into
app.py and exists only to document that the additional candidates were
considered and did not outperform the three algorithms actually used here.
"""


import warnings
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

print("=" * 80)
print("SOLO PARENT DSS - ML MODEL TRAINING (PRIMARY: DT / RF / LOGREG)")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA & DATA QUALITY REPORT
# ============================================================================
print("\n[STEP 1] Loading Dataset & Data Quality Check...")
df = pd.read_csv('data/solo_parent_dataset.csv')
num_rows, num_cols = df.shape
print(f"  Dataset shape: {num_rows} rows, {num_cols} columns")

# Class Balance
class_counts = df['Eligibility'].value_counts()
class_ratios = df['Eligibility'].value_counts(normalize=True)
print("  Class Distribution:")
for cls in class_counts.index:
    print(f"    - {cls}: {class_counts[cls]} ({class_ratios[cls]:.1%})")

# Missing Values
missing_val = df.isnull().sum()
print("  Missing Values per Column:")
for col in missing_val.index:
    if missing_val[col] > 0:
        print(f"    - {col}: {missing_val[col]}")
if missing_val.sum() == 0:
    print("    - None (0 missing values)")

# Duplicate Rows
num_duplicates = df.duplicated().sum()
print(f"  Duplicate Rows: {num_duplicates}")

# Outliers in Monthly_Income
q1 = df['Monthly_Income'].quantile(0.25)
q3 = df['Monthly_Income'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = df[(df['Monthly_Income'] < lower_bound) | (df['Monthly_Income'] > upper_bound)]
print(f"  Monthly_Income Outliers (IQR Method): {len(outliers)} records out of range [{lower_bound:.1f}, {upper_bound:.1f}]")

# Dataset Size Warning
if num_rows < 1000:
    print("  [WARNING] Dataset has less than 1000 rows. This is a primary accuracy bottleneck.")
    print("            Model capacity and complexity must be regulated to avoid overfitting.")
else:
    print("  [OK] Dataset size is sufficient (>1000 rows).")

# ============================================================================
# STEP 2: REMOVE LEAKING COLUMNS
# ============================================================================
print("\n[STEP 2] Removing Truly Leaking Columns...")
# Recommendation & Priority_Level are generated as outcomes of assessment.
# Full_Name & Barangay are identifiers.
# Civil_Status and Sex are NOT leaks; they are predictive demographic characteristics.
leaking_columns = ['Recommendation', 'Priority_Level', 'Full_Name', 'Barangay']
leaking_present = [col for col in leaking_columns if col in df.columns]
df_clean = df.drop(columns=leaking_present, errors='ignore')

print(f"  Removed leaking columns: {leaking_present}")
print(f"  Retained columns: {df_clean.columns.tolist()}")

# ============================================================================
# STEP 3: FEATURE ENGINEERING & SPLIT
# ============================================================================
print("\n[STEP 3] Feature Engineering...")

# Map 'Gender' column to 'Gender' and format
if 'Gender' in df_clean.columns:
    df_clean['Gender'] = df_clean['Gender'].astype(str).str.strip().str.upper()

feature_columns = [
    'Age',
    'Gender',
    'Civil_Status',
    'Educational_Attainment',
    'Employment_Status',
    'Monthly_Income',
    'Number_of_Dependents',
    'With_Minor',
    'With_PWD',
    'Type_of_Solo_Parent'
]

X = df_clean[feature_columns].copy()
y = (df_clean['Eligibility'] == 'Eligible').astype(int)

# Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"  Features selected: {feature_columns}")
print(f"  Train set: {len(X_train)} records, Test set: {len(X_test)} records")

# ============================================================================
# STEP 4: PREPROCESSING & PIPELINE DESIGN
# ============================================================================
print("\n[STEP 4] Defining Modular Preprocessing Pipelines...")

categorical_cols = [
    'Gender',
    'Civil_Status',
    'Educational_Attainment',
    'Employment_Status',
    'With_Minor',
    'With_PWD',
    'Type_of_Solo_Parent'
]
numerical_cols = ['Age', 'Monthly_Income', 'Number_of_Dependents']

# 1. Tree Preprocessor (uses OrdinalEncoder)
tree_preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ]
)

# 2. Linear/SVM Preprocessor (uses OneHotEncoder + StandardScaler)
linear_preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ]
)

# ============================================================================
# STEP 5: HYPERPARAMETER TUNING & BENCHMARK
# ============================================================================
print("\n[STEP 5] Grid Search Hyperparameter Tuning & Candidate Comparison...")

candidate_configs = {
    'RandomForestClassifier': (
        Pipeline([
            ('preprocessor', tree_preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5]
        }
    ),
    'DecisionTreeClassifier': (
        Pipeline([
            ('preprocessor', tree_preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=42, class_weight='balanced'))
        ]),
        {
            'classifier__max_depth': [3, 5, 8, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 5, 10]
        }
    ),
    'LogisticRegression': (
        Pipeline([
            ('preprocessor', linear_preprocessor),
            ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
        ]),
        {
            'classifier__C': [0.1, 1, 10]
        }
    )
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
benchmark_results = []

for name, (pipeline, param_grid) in candidate_configs.items():
    print(f"  [TUNE & EVALUATE] {name}...")
    
    # We optimize F1-score during grid search to keep precision/recall balanced
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=skf,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Evaluate best estimator on test set
    best_estimator = grid_search.best_estimator_
    y_pred = best_estimator.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    benchmark_results.append({
        'model_name': name,
        'best_params': grid_search.best_params_,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'pipeline': best_estimator
    })
    
    print(f"    - Best Params: {grid_search.best_params_}")
    print(f"    - Accuracy:  {accuracy:.1%}")
    print(f"    - Precision: {precision:.1%}")
    print(f"    - Recall:    {recall:.1%}")
    print(f"    - F1-Score:  {f1:.1%}")

# Rank based on selection criteria: recall -> f1 -> precision -> accuracy
ranked_results = sorted(
    benchmark_results,
    key=lambda item: (item['recall'], item['f1'], item['precision'], item['accuracy']),
    reverse=True
)

# ============================================================================
# STEP 5b: INCOME-USAGE SAFETY CHECK
# ============================================================================
# A model can score well on accuracy/precision/recall/F1 while structurally
# never using Monthly_Income at all — e.g. a shallow Decision Tree whose
# first few splits on Number_of_Dependents/Type_of_Solo_Parent/Civil_Status
# already separate the training data well enough that income is never
# reached. That model would then treat a minimum-wage applicant and a
# millionaire identically, which is a real problem for an income-tested
# government benefits screener even if the headline metrics look fine.
# This step verifies each candidate actually relies on Monthly_Income to
# some non-trivial degree before it's eligible to be selected.
print("\n[STEP 5b] Income-Usage Safety Check...")
print("  Verifying each candidate actually uses Monthly_Income before it can be")
print("  selected for production, regardless of how well it scores otherwise...")

MIN_INCOME_IMPORTANCE = 0.01  # require >=1% relative importance/weight on income


def get_income_importance(pipeline):
    """Returns how much a fitted pipeline relies on Monthly_Income, as a
    fraction of total feature importance (tree models) or total absolute
    coefficient weight (linear models). Returns None if the classifier
    type isn't recognized, so the caller can flag it for manual review
    rather than silently assuming it's fine."""
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    feature_names = list(preprocessor.get_feature_names_out())
    income_indices = [i for i, name in enumerate(feature_names) if 'Monthly_Income' in name]
    if not income_indices:
        return 0.0

    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        total = float(importances.sum())
        if total == 0:
            return 0.0
        return float(sum(importances[i] for i in income_indices)) / total
    elif hasattr(classifier, 'coef_'):
        coefs = classifier.coef_[0]
        total_abs = float(sum(abs(c) for c in coefs))
        if total_abs == 0:
            return 0.0
        return float(sum(abs(coefs[i]) for i in income_indices)) / total_abs
    else:
        return None


for item in benchmark_results:
    income_importance = get_income_importance(item['pipeline'])
    item['income_importance'] = income_importance
    item['income_check_passed'] = (
        income_importance is not None and income_importance >= MIN_INCOME_IMPORTANCE
    )
    status = "PASS" if item['income_check_passed'] else "FAIL"
    imp_str = f"{income_importance:.1%}" if income_importance is not None else "UNKNOWN"
    print(f"    [{status}] {item['model_name']}: Monthly_Income importance = {imp_str}")

# Select the best-ranked model that ALSO passes the income-usage check —
# not just the top of the raw performance ranking.
selected_candidate = next(
    (item for item in ranked_results if item['income_check_passed']),
    None
)

if selected_candidate is None:
    print("  [WARNING] No candidate meaningfully uses Monthly_Income. Falling back to the")
    print("            top-ranked model anyway, but this should be investigated further —")
    print("            income-based screening is a real requirement under RA 11861 Sec. 33.")
    selected_candidate = ranked_results[0]
elif selected_candidate is not ranked_results[0]:
    skipped = ranked_results[0]
    print(f"  [OVERRIDE] {skipped['model_name']} ranked highest on accuracy/F1 "
          f"(income importance={skipped.get('income_importance', 0) or 0:.1%}) but failed the "
          f"income-usage check.")
    print(f"             Selecting {selected_candidate['model_name']} instead — it passes both "
          f"the performance ranking and the income-usage safety check.")
else:
    print(f"  [OK] Top-ranked model also passes the income-usage check — no override needed.")

best_config = selected_candidate
best_pipeline = best_config['pipeline']
selected_model_name = best_config['model_name']

print(f"\n  [SELECTED] {selected_model_name} as the production model candidate.")

# ============================================================================

# STEP 6: PROBABILITY CALIBRATION
# ============================================================================
print("\n[STEP 6] Calibrating Probabilities for Selected Model...")

# DIAGNOSTIC: probability spread BEFORE calibration
raw_proba = best_pipeline.predict_proba(X_test)[:, 1]
print(f"  [DIAGNOSTIC] Raw (uncalibrated) test probabilities: min={raw_proba.min():.3f}, "
      f"max={raw_proba.max():.3f}, mean={raw_proba.mean():.3f}, std={raw_proba.std():.3f}")
print(f"  [DIAGNOSTIC] Fraction of raw probs within 0.40-0.60 band: "
      f"{((raw_proba > 0.40) & (raw_proba < 0.60)).mean():.1%}")

# Calibrate the selected best pipeline on training data
calibrated_model = CalibratedClassifierCV(
    estimator=best_pipeline,
    method='sigmoid',
    cv=5
)
calibrated_model.fit(X_train, y_train)

# DIAGNOSTIC: probability spread AFTER calibration
cal_proba_diag = calibrated_model.predict_proba(X_test)[:, 1]
print(f"  [DIAGNOSTIC] Calibrated test probabilities: min={cal_proba_diag.min():.3f}, "
      f"max={cal_proba_diag.max():.3f}, mean={cal_proba_diag.mean():.3f}, std={cal_proba_diag.std():.3f}")
print(f"  [DIAGNOSTIC] Fraction of calibrated probs within 0.40-0.60 band: "
      f"{((cal_proba_diag > 0.40) & (cal_proba_diag < 0.60)).mean():.1%}")

# Evaluate calibrated model on test set
y_cal_pred = calibrated_model.predict(X_test)
accuracy = accuracy_score(y_test, y_cal_pred)
precision = precision_score(y_test, y_cal_pred, zero_division=0)
recall = recall_score(y_test, y_cal_pred, zero_division=0)
f1 = f1_score(y_test, y_cal_pred, zero_division=0)
cm = confusion_matrix(y_test, y_cal_pred)

print(f"  Calibrated {selected_model_name} Test Metrics:")
print(f"    - Accuracy:  {accuracy:.1%}")
print(f"    - Precision: {precision:.1%}")
print(f"    - Recall:    {recall:.1%}")
print(f"    - F1-Score:  {f1:.1%}")

# ============================================================================
# STEP 7: SAVE MODEL & METADATA
# ============================================================================
print("\n[STEP 7] Saving Model Artifacts...")
os.makedirs('model', exist_ok=True)
joblib.dump(calibrated_model, 'model/solo_parent_model.pkl')
joblib.dump({}, 'model/encoders.pkl')  # Saved empty dict for app.py startup backward compatibility
joblib.dump(feature_columns, 'model/feature_columns.pkl')

# Save model metadata
metadata = {
    'model_type': selected_model_name,
    'model_version': '5.0_primary_dt_rf_logreg',
    'training_date': pd.Timestamp.now().isoformat(),
    'selected_algorithm': selected_model_name,
    'feature_columns': feature_columns,
    'categorical_columns': categorical_cols,
    'numerical_columns': numerical_cols,
    'test_metrics': {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    },
    'benchmark_results': [
        {
            'model_name': item['model_name'],
            'best_params': item['best_params'],
            'accuracy': item['accuracy'],
            'precision': item['precision'],
            'recall': item['recall'],
            'f1': item['f1'],
            'confusion_matrix': item['confusion_matrix'],
            'income_importance': item.get('income_importance'),
            'income_check_passed': item.get('income_check_passed')
        }
        for item in ranked_results
    ],
    'income_usage_safety_check': {
        'min_income_importance_required': MIN_INCOME_IMPORTANCE,
        'selected_model_income_importance': best_config.get('income_importance'),
        'top_ranked_model_by_metrics_alone': ranked_results[0]['model_name'],
        'override_applied': selected_candidate is not ranked_results[0]
    }
}
joblib.dump(metadata, 'model/model_metadata.pkl')

print("  [OK] Calibrated Pipeline saved to: model/solo_parent_model.pkl")
print("  [OK] Dummy Encoders saved to: model/encoders.pkl")
print("  [OK] Features list saved to: model/feature_columns.pkl")
print("  [OK] Metadata saved to: model/model_metadata.pkl")
print("\n" + "=" * 80)
print("TRAINING & TUNING COMPLETED SUCCESSFULLY")
print("=" * 80)
