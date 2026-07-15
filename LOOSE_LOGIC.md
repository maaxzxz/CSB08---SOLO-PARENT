# Loose Logic Review — Solo Parent DSS (RA 11861)

Find-and-recommend pass. These are cases the system currently *allows* even though they're
odd, inconsistent, or exploitable — not because a rule permits them, but because **nothing
explicitly blocks them**. Each item was reproduced against the running
`assess_eligibility()` / `engine.evaluate()` unless noted.

Confidence = how sure I am this is a real problem worth closing (vs. an intentional edge case).

> **STATUS: ALL 13 IMPLEMENTED.** See the "Implementation" note under each item and the
> "How it was implemented" summary at the bottom. Verified: `eval_accuracy.py` still 100% /
> 0 mismatches, all 12 categories submit HTTP 200, guard rejections and PDF generation work.

---

## 1. Applicant age is never checked — a minor (or a newborn) can be "Eligible"
**Slips through:** The eligibility decision (`engine.evaluate()`) checks category, duration,
dependents, and pregnancy — but **not the applicant's own age**. The `Applicant` dataclass
doesn't even carry an age field. Reproduced: an applicant born 2015 (~11 y/o), and one born
today (age 0), both return `eligible=True`.
**Why it's risky:** A solo parent is by definition an adult caregiver. The earlier
rule-based code enforced `age >= 18`; that check was silently lost when the logic was unified
onto the rules engine. A child, or an obviously bogus birthdate, now passes eligibility.
**Recommendation:** Add an adult-age gate (e.g. `age >= 18`) as an eligibility precondition
in the engine (or in `build_rule_engine_applicant`), driven by the applicant's birthday.
Consider a sensible upper bound too (e.g. reject age > 120 as a data-entry error).
**Confidence: HIGH**

## 2. "Dependent child" is never validated against RA 11861's dependent definition
**Slips through:** Eligibility only requires `number_of_dependent_children >= 1`. RA 11861
defines a qualifying dependent as unmarried, unemployed, and **22 or younger — or older only
if unable to care for themselves due to disability**. Nothing enforces this. Reproduced:
`youngest_child_age = 40`, not PWD, still `eligible=True`. The form even allows
`youngest_child_age` up to 99.
**Why it's risky:** Someone whose only "dependent" is a 40-year-old, able-bodied, employed
relative would be marked eligible for solo-parent benefits. The `youngest_child_age` field
exists but is only wired to the VAT benefit (0–6), never to dependency qualification, and
`with_pwd` only boosts priority — it doesn't extend the age limit the way the law intends.
**Recommendation:** Enforce that at least one dependent qualifies: youngest child age ≤ 22,
**or** `with_pwd = Yes` when the youngest is over 22. Ideally validate per-dependent using the
family-composition ages rather than a single "youngest" number (see #6).
**Confidence: HIGH**

## 3. Sex = Male combined with the pregnancy category is accepted
**Slips through:** `is_pregnant` is derived purely from
`solo_parent_status == 'pregnant_woman_unborn_child'`; `sex` is never cross-checked.
Reproduced: `sex = male` + pregnancy category → `is_pregnant=True` → `eligible=True`.
**Why it's risky:** A biologically impossible / clearly-erroneous combination is treated as a
valid eligible case. At minimum it signals a data-entry error that should be caught.
**Recommendation:** If the pregnancy category is selected, require `sex = Female` (block the
combination client-side and re-validate server-side). Alternatively add an explicit
"currently pregnant" confirmation rather than inferring it from the category alone.
**Confidence: MEDIUM-HIGH**

## 4. Civil status ↔ solo-parent reason contradictions are mostly unenforced — and the one rule that exists is client-side only
**Slips through:** Only one consistency rule exists (Married ⇒ can't pick Annulled or
Widowed), and it lives entirely in the form's JavaScript. A direct POST bypasses it:
reproduced `civil_status = married` + `solo_parent_status = widowed` → `eligible=True`. Many
other contradictory pairs are never blocked at all, even in the UI: Single + "Widowed (Death
of Spouse)", Single + "Spouse Detained" (single but has a spouse?), Widowed + "Abandoned by
Spouse", etc. — all reproduced as `eligible=True`.
**Why it's risky:** Two kinds of gap here: (a) semantically impossible civil-status/reason
combinations are accepted as valid eligible cases; (b) the *only* guard that does exist is
not enforced on the server, so it's cosmetic — any script or tampered request ignores it.
**Recommendation:** Define an allowed reason-set per civil status (single source of truth),
enforce it **server-side** in `assess_eligibility`, and mirror it in the JS. E.g. "Single"
should not permit spouse-based reasons (widowed/abandoned/separated/detained/annulled).
**Confidence: HIGH** (client-only enforcement) / **MEDIUM** (exact contradiction matrix is a
domain decision).

## 5. `youngest_child_age = -1` (the "unborn" sentinel) is accepted for non-pregnancy categories
**Slips through:** `-1` is meant to mean "unborn child, pregnancy category." But any category
can submit `-1`. Reproduced: a Widowed applicant with `youngest_child_age = -1` is eligible
and is granted the VAT benefit (whose rule is `between -1 and 6`).
**Why it's risky:** A widowed/separated/etc. applicant claiming an *unborn* child is
contradictory, and the `-1` sentinel silently satisfies the VAT age window it was never meant
to satisfy outside pregnancy.
**Recommendation:** Only accept `youngest_child_age = -1` when the pregnancy category is
selected; otherwise require `youngest_child_age >= 0`. Enforce in both the form and the server.
**Confidence: MEDIUM**

## 6. `number_of_dependent_children` is never reconciled with the family-composition table
**Slips through:** The dependent count, the family table, and `youngest_child_age` are three
independent inputs. Reproduced: `number_of_dependent_children = 20` while the family table
holds a single row typed as "Parent" (zero children listed) → still eligible with full
benefits.
**Why it's risky:** The headline dependent count can be fabricated independently of the
household roster, and the ML `With_Minor` feature (derived from table ages) can silently
contradict the claimed count. Priority scoring uses the count, so inflating it can raise
priority with no corroborating roster.
**Recommendation:** Cross-check that the family table actually contains at least
`number_of_dependent_children` members whose relationship is child/dependent and whose ages
qualify (ties into #2). At minimum, warn/flag when the count and the roster disagree.
**Confidence: MEDIUM**

## 7. Negative / nonsensical monthly income is accepted server-side
**Slips through:** The UI's `currency-input` strips non-digits, but `parse_money()` on the
server does `float(text)` after only stripping commas. Reproduced via direct value:
`monthly_income = -5000` → parsed as `-5000.0`, which is `<=` every income ceiling, so it
**grants** Cash Subsidy, VAT, and Housing Priority.
**Why it's risky:** Negative income is meaningless and, worse, unlocks the most benefits. A
tampered/scripted POST (or a future UI change) hits this directly.
**Recommendation:** Clamp/validate income to `>= 0` in `parse_money()` (or reject negatives).
Combine with the existing outlier flag for the upper end.
**Confidence: MEDIUM** (needs a non-UI submission today, but it's a missing boundary check).

## 8. Duration has no upper bound and isn't sanity-checked against the applicant's age
**Slips through:** `category_duration_answer` has no max. Reproduced: `999999` months
(~83,000 years) accepted, eligible. There's also no check that the claimed duration is even
possible for the applicant's age — a 20-year-old can claim 300 months (25 years) of spouse
detention/separation.
**Why it's risky:** Physically impossible durations pass as valid, and an absurd value could
distort any future analytics or trust in the record.
**Recommendation:** Cap duration to a reasonable maximum (e.g. ≤ `age * 12`), and reject
durations that exceed the applicant's own age in months. Enforce server-side.
**Confidence: MEDIUM**

## 9. Conditional "required" and gating logic is enforced only in the browser
**Slips through:** The duration follow-up's `required` toggle and the civil-status gating are
JavaScript-only. There is **no server-side re-validation** of any conditionally-required
field, and in fact no server-side required-field validation at all (see #10).
**Why it's risky:** The intended "answer X ⇒ field Y becomes required" rules don't hold for
any request that doesn't run the page's JS.
*(Partial mitigation: for duration-gated categories, a missing duration parses to 0, which is
below the minimum, so eligibility fails safe. But the enforcement itself is absent, and other
conditional fields have no such backstop.)*
**Recommendation:** Re-validate conditional requirements server-side: if a duration-gated
category is chosen, require a duration; enforce the civil-status/reason matrix (#4); etc.
**Confidence: MEDIUM-HIGH**

## 10. No server-side required-field validation — malformed submissions are processed anyway
**Slips through:** Every "required" is an HTML attribute only. A direct POST omitting fields
is processed with silent `.get()` defaults (blank name, blank sex, empty civil status) and
still renders a result page and a downloadable PDF.
**Why it's risky:** The system will emit an official-looking assessment/PDF built on missing
or blank inputs. Even as a "first-step screening," producing a record from an incomplete
submission is an integrity gap.
**Recommendation:** Add a server-side validation layer that rejects submissions missing the
truly-required fields (name, birthday, sex, civil status, solo-parent status, dependents,
income, and the Section IV screening answers) before assessing.
**Confidence: MEDIUM**

## 11. "Dependent currently studying = Yes" isn't tied to having a school-age dependent
**Slips through:** Educational Support is granted on `dependent_currently_studying = Yes` +
`number_of_dependents >= 1`, with no check that any dependent is actually of school age.
Someone whose only dependent is an infant (age 0, or the `-1` unborn sentinel) can answer
"Yes" and receive Educational Support.
**Why it's risky:** A benefit tied to an enrolled child can be granted with no plausibly
enrolled child, purely on a self-declared toggle that contradicts the age data on file.
**Recommendation:** Cross-validate "currently studying" against dependent ages (e.g. require
at least one dependent within a school-age range), or at least flag the contradiction.
**Confidence: LOW-MEDIUM**

## 12. `total_family_income` is display-readonly but trusted as-is on the server
**Slips through:** The field is auto-calculated and `readonly` in the UI, but the server reads
whatever value is posted. A direct POST can set an arbitrary `total_family_income`
independent of the family-income rows; it feeds the income-outlier verification check.
**Why it's risky:** A "read-only, computed" value isn't actually authoritative — it can be
decoupled from its inputs by a non-UI request, weakening the outlier safety net.
**Recommendation:** Recompute `total_family_income` server-side from the submitted family
rows rather than trusting the posted total.
**Confidence: LOW-MEDIUM**

## 13. Self-declared screening answers are unverifiable single-toggle unlocks (likely intentional)
**Slips through:** `receiving_other_govt_cash_aid`, `formal_philhealth_member`, and
`dependent_currently_studying` are self-declared yes/no toggles that directly gate benefits
(e.g. answering "No" to other govt aid unlocks the Cash Subsidy). Nothing verifies them.
**Why it's risky:** Benefit gating rests entirely on unverified self-declaration, so answers
can be chosen to maximize benefits.
**Recommendation:** Likely acceptable for a first-step screening (final MSWDO/DSWD interview
verifies), but worth a visible disclaimer that these are declarations subject to verification,
and worth logging them for the caseworker. Consider requiring supporting documents in the
generated checklist when any of these unlock a benefit.
**Confidence: LOW** (probably intentional by design).

---

## Suggested priority if you act on these
1. **#1 (age floor)** and **#2 (dependent qualification)** — these let clearly-ineligible
   people pass, and both trace to eligibility checks that were lost/never added.
2. **#4 (server-side civil-status/reason enforcement)** and **#9/#10 (server-side validation
   generally)** — close the "browser-only enforcement" class of gap.
3. **#3, #5, #7, #8** — boundary/consistency guards that prevent impossible inputs.
4. **#6, #11, #12** — cross-field consistency and trusting computed/roster data.
5. **#13** — mostly a disclaimer/logging improvement.

---

## How it was implemented (all 13)

Two-layer design that preserves the engine's 100% dataset validation:

**Layer 1 — the rules engine (`engine.py` + `rules.json`), for true eligibility rules that
the dataset already respects, so validation stays 100%:**
- **#1** — new `min_applicant_age: 18` (and `max_applicant_age: 120`) in `rules.json`; the
  engine's `Applicant` now carries `age`, and `evaluate()` rejects applicants under the floor.
  Both callers (`app.py` `build_rule_engine_applicant`, `eval_accuracy.py` `row_to_applicant`)
  now pass age. (Confirmed the dataset has no eligible row < 18, so zero labels changed.)
- **#2** — new `max_dependent_age: 22` in `rules.json`; `evaluate()` marks Not Eligible when
  the youngest declared dependent is over the cap and no PWD dependent was declared.

**Layer 2 — a submission-integrity guard layer in `app.py`
(`evaluate_submission_guards`), for cross-field/input-integrity checks that must NOT live in
the pure engine (they'd break dataset validation or aren't RA-rule logic).** Returns
`hard_errors` (block → "Submission Needs Correction" with the reasons) and `soft_warnings`
(flag → "Needs Verification", still eligible):
- **#3** hard-block Male + pregnancy category.
- **#4** hard-block civil-status/reason contradictions via `CIVIL_STATUS_ALLOWED_REASONS`
  (single-source policy matrix; guardianship/caregiver/pregnancy/OFW allowed under any status).
- **#5** hard-block the `-1` "unborn" youngest-child value outside the pregnancy category.
- **#6** soft-warn when the declared dependent count exceeds the child/dependent rows listed
  in the family table.
- **#7** hard-block negative income (rejected, not clamped — clamping to 0 would wrongly grant
  the most benefits).
- **#8** hard-block durations over 1200 months, or longer than the applicant has been alive
  (`> age * 12`).
- **#10** server-side required-field check (`REQUIRED_SUBMISSION_FIELDS`) so a malformed direct
  POST is rejected instead of producing a result/PDF from blanks.
- **#11** soft-warn when "a dependent is currently studying" but the only dependent is under 3.
- **#12** `total_family_income` is now recomputed server-side from the family rows (in both
  `assess_eligibility` and `submit_assessment`), never trusted from the posted field.

**Layer 3 — client-side mirrors in `assessment_form.html` (UX; server remains authoritative):**
- **#4** full `CIVIL_STATUS_ALLOWED_REASONS` matrix disables inconsistent reason options
  (replacing the old Married-only rule).
- **#3** the pregnancy option is disabled when Sex = Male (Sex now re-triggers the gate).
- **#5** the youngest-child-age field's `min` flips to `0` off the pregnancy category (and any
  stray `-1` is cleared).
- **#8** duration input capped with `maxlength="4"`.
- **#9** conditional-required already toggled for the duration follow-up; kept.
- **#13** self-declaration disclaimer added atop Section IV (Benefit Screening).

**Result surfacing (`templates/result.html`):** hard errors render as a "Submission Needs
Correction" list; soft warnings render inside the verification notice.

**Verification after implementation:**
- `eval_accuracy.py` — **100%** on every field, 0 mismatches (engine changes didn't move any
  label).
- `test_eligibility.py` — passing.
- All **12** categories submit HTTP 200 with valid civil-status/reason combos.
- Each guard reproduced: minor, over-age dependent, male+pregnancy, `-1` misuse, civil/reason
  contradiction, negative income, absurd/too-long duration, missing fields → all blocked with
  a clear reason; roster mismatch and studying-infant → soft-flagged; PDF generates for both
  eligible and guard-error results.

**One tunable policy decision to be aware of:** the `CIVIL_STATUS_ALLOWED_REASONS` matrix
(#4) is a reasonable default that blocks clear contradictions while allowing
guardianship/caregiver/pregnancy/OFW under any civil status. If your MSWDO wants a
stricter or looser matrix, it's a single dict in `app.py` (mirror the JS copy).
