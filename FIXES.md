# Codebase Review & Fixes Log

Review of the Solo Parent DSS (RA 11861) codebase for logic errors, bugs, gaps, and
structural problems. Each entry: what was wrong, what changed, why, and a rough
running completeness score toward "logically sound and complete."

Baseline before fixes: `py_compile` clean, `eval_accuracy.py` 100% / 0 mismatches,
`test_eligibility.py` passing. So the core rules engine was already correct; the issues
below are in the app/presentation layer and edge-case handling around it.

---

## Fixed

### 1. Income-outlier verification masked a genuine "Not Eligible" verdict — HIGH
**File:** `app.py` (`submit_assessment`)
**Problem:** `income_outlier` unconditionally set `needs_verification = True`, even for
applicants the rules engine had *definitively* rejected for a non-income reason (e.g.
category duration below the legal minimum). The result template's
`{% elif result.needs_verification %}` branch then rendered a yellow "Needs Verification"
card *instead of* the red "Currently Not Eligible" card — hiding the real rejection reason.
Reproduced: spouse-detained 1 month (< 3 required) + ₱5,000,000 income showed
"Needs Verification", not "Not Eligible".
**Why it's wrong:** Eligibility itself is not income-gated (income only gates *which
benefits* apply). An applicant rejected for duration/dependents is definitively not
eligible regardless of income, so an income-range caveat adds no information there and
actively suppresses the real reason.
**Fix:** Only let `income_outlier` drive `needs_verification` when the applicant is
otherwise eligible. The `income_outlier_flagged` metadata is still recorded either way.
_Completeness: ~40%_

### 2. Homepage advertised the wrong cash-subsidy amount — HIGH (legal accuracy)
**File:** `templates/index.html`
**Problem:** The landing page listed "**₱1,500** Monthly Subsidy - Financial assistance for
children's daily needs." RA 11861 Sec. 15(a) is **₱1,000/month per solo parent** (not
₱1,500, and not per child). This is the exact error already corrected in `app.py`,
`result.html`, and the PDF — the homepage was the last place still showing it.
**Fix:** Corrected to "₱1,000 Monthly Cash Subsidy" with accurate eligibility framing, and
tightened the other three bullets to match the actual Sec. 15/Sec. 9 benefit scope.
_Completeness: ~50%_

### 3. Stale "Needs Verification for Subsidy/Discounts" wording in the PDF — MEDIUM
**File:** `app.py` (`generate_pdf` → `status_text`)
**Problem:** The PDF's system-assessment line read "ELIGIBLE (Needs Verification for
Subsidy/Discounts)". That phrasing dates from an older income-tier design that no longer
exists; verification is now triggered by an ML/rules conflict or an income outlier, not by
subsidy/discount tiering.
**Fix:** Reworded to "ELIGIBLE (Flagged for MSWDO/DSWD Verification)".
_Completeness: ~55%_

### 4. Misleading ML "Technical Details" panel — MEDIUM
**File:** `templates/_result_technical_details.html`
**Problems:** (a) The card was titled "ML Assessment Result" and its ELIGIBLE/NOT-ELIGIBLE
badge was driven by `result.eligible`, which is the **rules-engine** decision, not the ML
prediction — implying the ML made the call. (b) "Model Status" was hardcoded to a green
"Success" badge even when the model errored or was unavailable. (c) When the model was
unavailable/errored, `prob_eligible`/`prob_not_eligible`/`confidence` are 0/1/0 placeholders,
but they were still displayed as if real (e.g. an ELIGIBLE result showing "0.0% confidence,
100% Not-Eligible probability").
**Fix:** Rewrote the panel: added a lead sentence clarifying the rules engine decides and
the ML is a secondary signal; the badge/probabilities now render only when
`model_status == 'success'`; an informational note replaces the fake numbers when the model
is unavailable; and the "disagreement" bullet now explains it flags an unusual case for
review rather than implying the result is wrong.
_Completeness: ~65%_

### 5. engine.py: unused import + stale docstring — LOW (cleanup)
**File:** `engine.py`
**Problem:** `field` was imported from `dataclasses` but never used, and the module
docstring referenced a non-existent `api.py`.
**Fix:** Dropped the unused import; updated the docstring to describe the actual callers
(`app.py`, `eval_accuracy.py`).
_Completeness: ~68%_

### 6. engine.py: fragile `between` bound resolution — LOW (hardening)
**File:** `engine.py` (`_resolve`)
**Problem:** The lower bound used
`condition_node.get("min", condition_node.get("min_ref") and thresholds[...])`. This relies
on truthiness: a legitimate `min` / threshold value of `0` would be treated as falsy and the
bound silently dropped, and an absent bound produced `None`, which would raise on the
`None <= field_val` comparison. Not currently triggered by rules.json (all `between` rules
use an explicit `min`), but a latent trap for future rules.
**Fix:** Resolve `min`/`max` via explicit `in` membership checks, defaulting to
`-inf`/`+inf` when a bound is absent instead of `None`. Verified `eval_accuracy.py` still
100%.
_Completeness: ~70%_

### 7. Stale fixtures in the PDF dev harness — LOW (accuracy/consistency)
**File:** `test_pdf.py`
**Problem:** The three sample results still used the pre-rewrite benefit set and the wrong
"PHP 1,500 cash assistance per child" descriptions, plus benefits (Flexible Work Schedule,
Comprehensive Health Services) that no longer exist in `rules.json`. Running the harness
produced PDFs that misrepresented the current benefit model.
**Fix:** Updated all three fixtures to the current rules.json benefit set and correct
Sec. 15/Sec. 9 wording, and made the "borderline"/"high income" cases reflect the real
income-gated benefit subset. Harness still generates all three PDFs successfully.
_Completeness: ~80%_

---

## Validation after fixes

- `python -m py_compile app.py engine.py` — clean
- `python eval_accuracy.py` — **100%** on every field, 0 mismatches (unchanged)
- `python test_eligibility.py` — passing
- `python test_pdf.py` — all 3 PDFs generated
- Live smoke test: all **12** solo-parent categories + empty-optional-field edge cases
  return HTTP 200 (no 500s)
- ML-unavailable path: technical-details template renders the informational note instead of
  fake 0% probabilities
- Income-outlier: eligible+extreme income still flags verification; not-eligible+extreme
  income now correctly shows "Not Eligible" (fix #1)

Estimated overall completeness toward logically sound & complete: **~82%**. The core rules
engine was already correct (100% dataset validation); the fixes above resolved the
app/presentation-layer bugs and inaccuracies layered on top of it.

---

## Needs your input

These are genuine design decisions, not clear-cut bugs — flagging rather than guessing:

### A. Pregnancy category vs. the "Number of Dependent Children ≥ 1" form rule
**Where:** `templates/assessment_form.html` (`number_of_dependent_children` has `min="1"
required`) vs. `engine.py` pregnancy branch (`is_pregnancy_category`, `needs_dependent:
false`).
**Issue:** A first-time pregnant applicant (unborn child, no children born yet) genuinely has
**0** dependent children, but the form forces a minimum of 1. The engine grants pregnancy
eligibility regardless of the count, but the falsely-entered "1" then feeds priority scoring
and the educational-support benefit condition. Practical impact is small (educational support
also requires `dependent_currently_studying = Yes`), but the recorded data is inaccurate.
**Options:** (a) make the dependents minimum dynamic — `0` when the pregnancy category is
selected, `1` otherwise; (b) leave as-is and treat the unborn child as "1 dependent" by
convention. Needs your call on how pregnancy-with-no-born-children should be modeled.

### B. `With_Minor` ML feature can diverge from the dependent count
**Where:** `app.py` `has_minor_dependents()` derives the ML `With_Minor` feature from the
family-composition table's ages, while eligibility uses the separate
`number_of_dependent_children` field. These two independent inputs can disagree (e.g. 2
dependent children declared, but none listed as minors in the household table). Only the
secondary ML feature is affected, not the authoritative rules decision — but it's a latent
inconsistency. Whether to reconcile them (and how) depends on what the household table is
meant to represent vs. the dependent-children count.

### C. Bare `except:` clauses (code quality, not a live bug)
**Where:** ~8 spots in `app.py` (e.g. `calculate_age`, numeric parsing fallbacks). They
behave correctly today but a bare `except:` would also swallow unrelated programming errors
(NameError, etc.) introduced by future refactors. Low priority; left as-is to avoid churn,
noting it here for a future hygiene pass.
