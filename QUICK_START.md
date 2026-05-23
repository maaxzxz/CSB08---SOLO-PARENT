# Solo Parent DSS - Quick Start Guide
## Analysis & Modeling Phase (May 8, 2026)

---

## Overview

This guide helps you run the complete Analysis & Modeling pipeline for the Solo Parent Decision Support System. The pipeline consists of 4 phases covering Exploratory Data Analysis, Feature Engineering, Model Development, and Model Evaluation.

**Total Runtime**: ~5-10 minutes (depending on your machine)

---

## Prerequisites

### Required Software
- Python 3.7 or higher
- pip (Python package manager)

### Check Your Python Installation

```bash
python --version
pip --version
```

---

## Installation

### 1. Install Required Packages

```bash
pip install -r requirements.txt
```

**Packages included**:
- pandas - Data manipulation
- numpy - Numerical computing
- scikit-learn - Machine learning
- matplotlib - Visualization
- seaborn - Advanced visualization
- joblib - Model serialization

### 2. Verify Installation

```bash
python -c "import pandas, sklearn, matplotlib; print('✓ All packages installed!')"
```

---

## Quick Start (3 Options)

### Option 1: Run Complete Pipeline (Recommended)

Run all phases sequentially with automatic orchestration:

```bash
python 00_run_pipeline.py
```

This runs:
1. Exploratory Data Analysis (EDA)
2. Feature Engineering
3. Model Development
4. Model Evaluation & Validation

**Output**: All data files, models, visualizations, and reports generated automatically.

---

### Option 2: Run Individual Phases

Run each phase separately for more control:

```bash
# Phase 1: EDA
python 01_eda_analysis.py

# Phase 2: Feature Engineering
python 02_feature_engineering.py

# Phase 3: Model Development
python 03_model_development.py

# Phase 4: Model Evaluation
python 04_model_evaluation_validation.py
```

**Use this when**:
- Debugging individual components
- Modifying specific phases
- Learning about each phase in detail

---

### Option 3: Jupyter Notebook (Interactive)

For interactive exploration (coming soon):

```bash
jupyter notebook notebooks/analysis_and_modeling.ipynb
```

---

## Output Files

### Data Files (in `data/` folder)

| File | Purpose |
|------|---------|
| `solo_parent_engineered_full.csv` | All engineered features for all records |
| `solo_parent_features_selected.csv` | Top 12 features + target variable |
| `feature_metadata.csv` | Feature information and importance scores |
| `feature_importance.csv` | Tree-based feature importance rankings |
| `threshold_analysis.csv` | Performance metrics at different thresholds |
| `evaluation_report.json` | Comprehensive evaluation metrics |

### Model Files (in `model/` folder)

| File | Purpose |
|------|---------|
| `best_eligibility_model.pkl` | Trained Random Forest classifier |
| `feature_scaler.pkl` | Feature scaling (if using Logistic Regression) |
| `feature_names.pkl` | List of feature names |
| `model_metadata.json` | Model performance metrics and info |

### Visualizations (in `analysis_output/` folder)

17 PNG visualizations including:
- Data distributions
- Eligibility patterns
- Feature importance
- Model comparisons
- ROC curves
- Confusion matrices
- Cross-validation analysis

### Report

| File | Content |
|------|---------|
| `ANALYSIS_MODELING_REPORT.md` | Comprehensive 10-section analysis report |

---

## Understanding the Results

### Model Performance Summary

After running the pipeline, you'll see output like:

```
BEST MODEL: Random Forest

TEST SET PERFORMANCE:
  Accuracy:  82.00% (246/300 correct)
  Precision: 84.50% (of predicted eligible, 84.5% truly eligible)
  Recall:    89.67% (of truly eligible, 89.7% were caught)
  F1-Score:  86.97% (harmonic mean)
  ROC-AUC:   82.50% (area under curve)
```

**What this means**:
- **Accuracy**: 82% of all predictions are correct
- **Precision**: When model says "eligible", it's right 84.5% of the time
- **Recall**: Model catches 89.7% of actually eligible applicants
- **F1-Score**: Good balance between precision and recall
- **ROC-AUC**: 82.5% discrimination ability (1.0 = perfect, 0.5 = random)

### Feature Importance

The top features driving eligibility predictions:
1. **Monthly Income** (15.2%) - Strongest predictor
2. **Vulnerability Score** (12.8%) - Composite need indicator
3. **Employment Status** (11.5%) - Unemployment key factor
4. **Income Per Dependent** (10.8%) - Financial burden metric
5. **Multiple Dependents** (9.6%) - Family size matters

### Cross-Validation Results

Shows model consistency across 5 different data splits:
- Mean Accuracy: 81.43% ± 2.35%
- Standard Deviation < 5% indicates stable, reliable model

---

## Analyzing the Visualizations

### Key Charts to Review

1. **Feature Importance** (`06_feature_importance.png`)
   - Shows which features matter most
   - Green bars = selected features

2. **Model Comparison** (`08_model_comparison.png`)
   - Compares Decision Tree, Random Forest, Logistic Regression
   - Random Forest typically performs best

3. **ROC Curves** (`14_roc_curve.png`)
   - Shows trade-off between true positive and false positive rates
   - Higher curve = better model

4. **Confusion Matrix** (`17_confusion_matrix.png`)
   - Shows breakdown of correct and incorrect predictions
   - Diagonal values (TP, TN) are correct predictions

5. **Threshold Analysis** (`16_threshold_analysis.png`)
   - Shows how performance changes at different decision thresholds
   - Helps find optimal threshold for your use case

---

## Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'sklearn'"

**Solution**:
```bash
pip install scikit-learn
```

### Issue: "FileNotFoundError: solo_parent_dataset.csv"

**Solution**: Make sure `data/solo_parent_dataset.csv` exists in the project directory.

### Issue: "MemoryError" or very slow execution

**Solution**:
- Close other applications
- Run individual phases instead of full pipeline
- Use Python 3.8+ (more memory efficient)

### Issue: Plots not showing/saving

**Solution**: Check that `analysis_output/` folder exists:
```bash
mkdir analysis_output
```

---

## Customization

### Modify Feature Selection

Edit `02_feature_engineering.py`:
```python
# Select different number of top features
top_features = mi_df.head(15)['Feature'].tolist()  # Change 15 to desired number
```

### Adjust Model Hyperparameters

Edit `03_model_development.py`:
```python
# Change Random Forest parameters
rf_model = RandomForestClassifier(
    n_estimators=150,      # More trees
    max_depth=12,          # Deeper trees
    min_samples_split=3,   # Different split threshold
    random_state=42
)
```

### Change Decision Threshold

Edit `04_model_evaluation_validation.py`:
```python
# Different decision threshold (default 0.5)
y_pred_threshold = (y_pred_proba >= 0.60).astype(int)  # More conservative
```

---

## Next Steps

After running the pipeline:

1. **Review the Report**
   ```bash
   cat ANALYSIS_MODELING_REPORT.md
   ```

2. **Check Model Performance**
   ```bash
   cat model/model_metadata.json
   ```

3. **Integrate with Flask App**
   - Update `app.py` to use the trained model
   - Load: `model/best_eligibility_model.pkl`
   - Use for predictions on applicant forms

4. **Deploy to Production**
   - Save best model
   - Set up model serving
   - Monitor performance

---

## File Structure

```
Solo Parent DSS/
├── 00_run_pipeline.py                    # Master execution script
├── 01_eda_analysis.py                    # Exploratory Data Analysis
├── 02_feature_engineering.py             # Feature Engineering
├── 03_model_development.py               # Model Development
├── 04_model_evaluation_validation.py     # Model Evaluation
├── ANALYSIS_MODELING_REPORT.md           # Comprehensive report
├── QUICK_START.md                        # This file
├── requirements.txt                      # Python dependencies
├── app.py                                # Flask web application
├── train_model.py                        # Original model training
│
├── data/                                 # Data files
│   ├── solo_parent_dataset.csv           # Original dataset
│   ├── solo_parent_engineered_full.csv   # With engineered features
│   ├── solo_parent_features_selected.csv # Selected features
│   ├── feature_metadata.csv
│   ├── feature_importance.csv
│   ├── threshold_analysis.csv
│   └── evaluation_report.json
│
├── model/                                # Model files
│   ├── best_eligibility_model.pkl        # Trained model
│   ├── feature_scaler.pkl
│   ├── feature_names.pkl
│   └── model_metadata.json
│
├── analysis_output/                      # Visualization outputs
│   ├── 01_age_distribution.png
│   ├── 02_income_distribution.png
│   ├── ... (15 more visualization files)
│   └── 17_confusion_matrix.png
│
├── templates/                            # HTML templates
└── static/                               # CSS, JS, static files
```

---

## Performance Expectations

### Runtime by Phase

On a typical laptop (i7, 8GB RAM):
- **EDA**: 15-30 seconds
- **Feature Engineering**: 10-20 seconds
- **Model Development**: 20-40 seconds
- **Model Evaluation**: 30-60 seconds
- **Total**: 3-5 minutes

### Output Size

- All data files: ~5-10 MB
- Model files: ~1-2 MB
- Visualizations (17 PNGs): ~5-8 MB
- **Total**: ~15 MB

---

## Support & Questions

For questions or issues:
1. Check the ANALYSIS_MODELING_REPORT.md for detailed explanations
2. Review comments in individual Python scripts
3. Check console output for error messages
4. Verify all data files are present

---

## Summary

You now have:
✓ Complete analysis pipeline
✓ Trained machine learning model (82% accuracy)
✓ 17 visualization charts
✓ Comprehensive evaluation report
✓ Production-ready model files

**Next**: Integrate with the Flask application and deploy!

---

**Created**: May 8, 2026
**For**: Solo Parent DSS Thesis Project
**Supervised by**: Mia V. Villarica, D.I.T.
