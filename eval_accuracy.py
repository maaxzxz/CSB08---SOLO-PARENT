"""
Compares engine.evaluate() output against the labeled dataset row-by-row.
Reports per-field accuracy so a drift in ONE rule (e.g. VAT discount) is
visible instead of hiding inside one overall accuracy number.

Usage: python eval_accuracy.py
"""
import pandas as pd
from engine import Applicant, evaluate

BENEFIT_COL_TO_ID = {
    "Cash_Subsidy_Sec15a": "cash_subsidy_sec15a",
    "VAT_Discount_Sec15b": "vat_discount_sec15b",
    "PhilHealth_Coverage_Sec15c": "philhealth_coverage_sec15c",
    "Livelihood_Priority_Sec15d": "livelihood_priority_sec15d",
    "Housing_Priority_Sec15e": "housing_priority_sec15e",
    "Educational_Support_Sec9": "educational_support_sec9",
}


def row_to_applicant(row) -> Applicant:
    yc = row["Youngest_Child_Age"]
    yc = None if pd.isna(yc) or yc == "" else int(yc)
    return Applicant(
        type_of_solo_parent=row["Type_of_Solo_Parent"],
        duration_months=int(row["Duration_Months"]),
        is_pregnant=bool(row["Is_Pregnant"]),
        number_of_dependents=int(row["Number_of_Dependents"]),
        monthly_income=float(row["Monthly_Income"]),
        employment_status=row["Employment_Status"],
        with_pwd=(row["With_PWD"] == "Yes"),
        youngest_child_age=yc,
        receiving_other_govt_cash_aid=(row["Receiving_Other_Govt_Cash_Aid"] == "Yes"),
        formal_philhealth_member=(row["Formal_PhilHealth_Member"] == "Yes"),
        dependent_currently_studying=(row["Dependent_Currently_Studying"] == "Yes"),
    )


def main():
    df = pd.read_csv("data/solo_parent_dataset.csv")

    results = {"Eligibility": [0, 0], "Priority_Level": [0, 0]}
    for col in BENEFIT_COL_TO_ID:
        results[col] = [0, 0]
    mismatches = []

    for idx, row in df.iterrows():
        applicant = row_to_applicant(row)
        out = evaluate(applicant)

        # Eligibility
        correct = out["eligibility"] == row["Eligibility"]
        results["Eligibility"][correct] += 1
        if not correct:
            mismatches.append((idx, "Eligibility", row["Eligibility"], out["eligibility"]))

        if row["Eligibility"] == "Eligible":
            # Priority level
            correct = out["priority_level"] == row["Priority_Level"]
            results["Priority_Level"][correct] += 1
            if not correct:
                mismatches.append((idx, "Priority_Level", row["Priority_Level"], out["priority_level"]))

            # Each benefit
            for col, bid in BENEFIT_COL_TO_ID.items():
                expected = row[col] == "Yes"
                actual = out["benefits"].get(bid, False)
                correct = expected == actual
                results[col][correct] += 1
                if not correct:
                    mismatches.append((idx, col, expected, actual))

    print(f"{'Field':30s} {'Accuracy':>10s} {'N':>6s}")
    print("-" * 50)
    for field_name, (wrong, right) in results.items():
        n = wrong + right
        if n == 0:
            continue
        acc = right / n
        print(f"{field_name:30s} {acc:>9.1%} {n:>6d}")

    print(f"\n{len(mismatches)} total field-level mismatches.")
    if mismatches:
        print("\nFirst 10 mismatches (row, field, expected, actual):")
        for m in mismatches[:10]:
            print(" ", m)


if __name__ == "__main__":
    main()
