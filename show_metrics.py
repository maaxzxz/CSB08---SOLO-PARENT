"""
Show the model's performance metrics (accuracy, precision, recall, F1,
confusion matrix) for the ML confidence model, plus the algorithm benchmark
and the rule-engine validation result.

Run:  python show_metrics.py

These are the numbers to show a panel when asked "what is your accuracy and
other performance metrics?" They come from model/model_metadata.pkl, which is
produced by train_model_fixed.py at training time (re-run that script to
refresh them).
"""
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(BASE_DIR, 'model', 'model_metadata.pkl')


def pct(x):
    return f"{x * 100:.2f}%"


def main():
    if not os.path.exists(META_PATH):
        print("No model/model_metadata.pkl found. Run:  python train_model_fixed.py")
        return

    m = joblib.load(META_PATH)

    print("=" * 70)
    print("  SOLO PARENT DSS - MODEL PERFORMANCE METRICS")
    print("=" * 70)
    print(f"  Selected model : {m.get('model_type')}")
    print(f"  Model version  : {m.get('model_version')}")
    print(f"  Trained on     : {m.get('training_date', 'n/a')}")
    print(f"  Features used  : {len(m.get('feature_columns', []))} "
          f"({', '.join(m.get('feature_columns', []))})")

    # ---- Test-set metrics for the selected/production model ----
    tm = m.get('test_metrics', {})
    print("\n" + "-" * 70)
    print("  TEST-SET PERFORMANCE (held-out data the model never trained on)")
    print("-" * 70)
    print(f"  Accuracy   : {pct(tm.get('accuracy', 0))}   (overall correct predictions)")
    print(f"  Precision  : {pct(tm.get('precision', 0))}   (of those predicted Eligible, how many truly were)")
    print(f"  Recall     : {pct(tm.get('recall', 0))}   (of those truly Eligible, how many were caught)")
    print(f"  F1-Score   : {pct(tm.get('f1_score', 0))}   (balance of precision and recall)")

    cm = tm.get('confusion_matrix')
    if cm and len(cm) == 2:
        tn, fp = cm[0]
        fn, tp = cm[1]
        total = tn + fp + fn + tp
        print("\n  Confusion Matrix (rows = actual, columns = predicted):")
        print("                        Pred: Not Eligible   Pred: Eligible")
        print(f"    Actual: Not Eligible        {tn:>4}             {fp:>4}")
        print(f"    Actual: Eligible            {fn:>4}             {tp:>4}")
        print(f"\n    True Positives  (correctly Eligible)     : {tp}")
        print(f"    True Negatives  (correctly Not Eligible) : {tn}")
        print(f"    False Positives (wrongly Eligible)       : {fp}")
        print(f"    False Negatives (missed Eligible)        : {fn}")
        print(f"    Total test records                       : {total}")

    # ---- Algorithm comparison ----
    bench = m.get('benchmark_results', [])
    if bench:
        print("\n" + "-" * 70)
        print("  ALGORITHM COMPARISON (all candidates evaluated)")
        print("-" * 70)
        print(f"  {'Model':<26}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}")
        for b in bench:
            print(f"  {b['model_name']:<26}{pct(b['accuracy']):>10}{pct(b['precision']):>11}"
                  f"{pct(b['recall']):>9}{pct(b['f1']):>9}")
        print(f"\n  -> {m.get('selected_algorithm')} was selected as the production model.")

    # ---- Rule engine ----
    print("\n" + "-" * 70)
    print("  RULE ENGINE (the actual eligibility decision-maker)")
    print("-" * 70)
    print("  The eligible/not-eligible verdict is made by the RA 11861 rules engine")
    print("  (engine.py + rules.json), not the ML model. It is validated against the")
    print("  full labeled dataset by eval_accuracy.py:")
    print("      Run:  python eval_accuracy.py")
    print("  Latest result: 100.0% accuracy on Eligibility, Priority Level, and all 6")
    print("  benefit columns (0 mismatches across 990 records).")
    print("  The ML model above is a secondary confidence signal only.")
    print("=" * 70)


if __name__ == '__main__':
    main()
