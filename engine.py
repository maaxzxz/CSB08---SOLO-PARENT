"""
RA 11861 Solo Parent eligibility & benefits engine.

This module reads rules.json as its ONLY source of truth. It contains
no hardcoded thresholds — change rules.json and behavior updates without
touching this file. app.py calls evaluate() for every assessment; eval_accuracy.py
validates it against the labeled dataset.
"""
import json
from pathlib import Path
from dataclasses import dataclass

RULES_PATH = Path(__file__).parent / "rules.json"
with open(RULES_PATH) as f:
    RULES = json.load(f)


@dataclass
class Applicant:
    type_of_solo_parent: str
    duration_months: int
    is_pregnant: bool
    number_of_dependents: int
    monthly_income: float
    employment_status: str
    with_pwd: bool
    youngest_child_age: int | None       # -1 = unborn, None = no children
    receiving_other_govt_cash_aid: bool
    formal_philhealth_member: bool
    dependent_currently_studying: bool


def _resolve(condition_node: dict, applicant: dict, thresholds: dict) -> bool:
    """Evaluate a single condition node (supports all/any/leaf)."""
    if "all" in condition_node:
        return all(_resolve(c, applicant, thresholds) for c in condition_node["all"])
    if "any" in condition_node:
        return any(_resolve(c, applicant, thresholds) for c in condition_node["any"])

    field_val = applicant.get(condition_node["field"])
    op = condition_node["op"]

    if op == "between":
        if field_val is None:
            return False
        # Resolve lower bound: explicit "min" wins, else "min_ref" looks it
        # up in thresholds, else default to negative infinity (no lower
        # bound). Using explicit membership checks avoids the truthiness
        # trap where a legitimate min/threshold value of 0 would be
        # discarded as falsy.
        if "min" in condition_node:
            lo = condition_node["min"]
        elif "min_ref" in condition_node:
            lo = thresholds[condition_node["min_ref"]]
        else:
            lo = float("-inf")
        if "max" in condition_node:
            hi = condition_node["max"]
        elif "max_ref" in condition_node:
            hi = thresholds[condition_node["max_ref"]]
        else:
            hi = float("inf")
        return lo <= field_val <= hi

    target = condition_node.get("value")
    if "value_ref" in condition_node:
        target = thresholds[condition_node["value_ref"]]

    if op == "==":
        return field_val == target
    if op == "!=":
        return field_val != target
    if op == "<=":
        return field_val is not None and field_val <= target
    if op == ">=":
        return field_val is not None and field_val >= target
    if op == "<":
        return field_val is not None and field_val < target
    if op == ">":
        return field_val is not None and field_val > target
    if op == "in":
        return field_val in target
    raise ValueError(f"Unknown op: {op}")


def evaluate(applicant: Applicant) -> dict:
    """
    Returns:
        {
          "eligibility": "Eligible" | "Not Eligible",
          "remarks": str,
          "benefits": {benefit_id: bool, ...},
          "recommendation": [ "Cash Subsidy (Sec. 15a)", ... ],
          "priority_level": "High" | "Medium" | "Low" | None
        }
    """
    cat = RULES["categories"].get(applicant.type_of_solo_parent)
    if cat is None:
        return {"eligibility": "Not Eligible", "remarks": "Unrecognized solo parent category.",
                "benefits": {}, "recommendation": [], "priority_level": None}

    # --- eligibility ---
    if cat["is_pregnancy_category"]:
        if applicant.is_pregnant:
            eligibility, remarks = "Eligible", ""
        else:
            eligibility, remarks = "Not Eligible", "Claimed pregnancy category but not marked pregnant."
    elif applicant.duration_months < cat["min_duration_months"]:
        eligibility = "Not Eligible"
        remarks = (f"Claimed '{applicant.type_of_solo_parent}' but duration is only "
                   f"{applicant.duration_months} month(s); RA 11861 requires at least "
                   f"{cat['min_duration_months']} month(s) for this category.")
    elif applicant.number_of_dependents < 1:
        eligibility = "Not Eligible"
        remarks = ("No qualifying dependent child under the applicant's care; RA 11861 "
                   "requires sole parental care and support of a child.")
    else:
        eligibility, remarks = "Eligible", ""

    if eligibility == "Not Eligible":
        return {"eligibility": eligibility, "remarks": remarks, "benefits": {},
                "recommendation": [], "priority_level": None}

    # --- benefits (each gated independently from rules.json) ---
    applicant_dict = {
        "monthly_income": applicant.monthly_income,
        "receiving_other_govt_cash_aid": applicant.receiving_other_govt_cash_aid,
        "youngest_child_age": applicant.youngest_child_age,
        "formal_philhealth_member": applicant.formal_philhealth_member,
        "employment_status": applicant.employment_status,
        "number_of_dependents": applicant.number_of_dependents,
        "dependent_currently_studying": applicant.dependent_currently_studying,
    }
    thresholds = RULES["thresholds"]

    benefits = {}
    recommendation = []
    for b in RULES["benefits"]:
        granted = _resolve(b["condition"], applicant_dict, thresholds)
        benefits[b["id"]] = granted
        if granted:
            recommendation.append(f"{b['label']} ({b['section']})")

    if not any(benefits.values()):
        fallback_id = RULES["fallback_benefit_if_none_qualify"]
        benefits[fallback_id] = True
        fb = next(b for b in RULES["benefits"] if b["id"] == fallback_id)
        recommendation.append(f"{fb['label']} ({fb['section']})")

    # --- priority score ---
    ps = RULES["priority_scoring"]
    n_benefits = sum(1 for v in benefits.values() if v)
    score = (n_benefits * ps["weight_per_benefit"]
             + (ps["weight_with_pwd"] if applicant.with_pwd else 0)
             + (ps["weight_is_pregnant"] if applicant.is_pregnant else 0)
             + min(applicant.number_of_dependents, ps["dependent_pair_cap"]) // ps["dependent_pair_divisor"]
               * ps["weight_per_dependent_pair"])

    if score >= ps["high_threshold"]:
        priority_level = "High"
    elif score >= ps["medium_threshold"]:
        priority_level = "Medium"
    else:
        priority_level = "Low"

    return {
        "eligibility": eligibility,
        "remarks": remarks,
        "benefits": benefits,
        "recommendation": recommendation,
        "priority_level": priority_level,
    }
