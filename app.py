from flask import Flask, render_template, request, send_file
from datetime import datetime
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from io import BytesIO
import hashlib
import hmac
import json
import os
import re
import textwrap
import joblib
import pandas as pd
from engine import Applicant as RuleApplicant, evaluate as rule_engine_evaluate, RULES as ENGINE_RULES

app = Flask(__name__)
# Also used to HMAC-sign the assessment payload embedded in result.html (see
# _sign_result_payload / download_pdf). Set SECRET_KEY in the environment for
# any real deployment — the fallback is a development-only value.
app.secret_key = os.environ.get('SECRET_KEY', 'solo-parent-dss-key')


def _sign_result_payload(payload: str) -> str:
    """HMAC-SHA256 signature over the result JSON that submit_assessment()
    embeds in result.html. /download-pdf only renders a payload whose
    signature verifies — without this, anyone could POST arbitrary JSON and
    mint an official-looking "ELIGIBLE" assessment PDF."""
    return hmac.new(app.secret_key.encode('utf-8'),
                    payload.encode('utf-8'), hashlib.sha256).hexdigest()

# Load ML Model and Encoders at startup using absolute paths relative to script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    ml_model = joblib.load(os.path.join(BASE_DIR, 'model', 'solo_parent_model.pkl'))
    ml_encoders = joblib.load(os.path.join(BASE_DIR, 'model', 'encoders.pkl'))
    ml_feature_columns = joblib.load(os.path.join(BASE_DIR, 'model', 'feature_columns.pkl'))
    ml_model_loaded = True
except Exception as e:
    print(f"Warning: Could not load ML model: {e}. Using rule-based assessment only.")
    ml_model_loaded = False
    ml_model = None
    ml_encoders = None
    ml_feature_columns = None

try:
    ml_model_metadata = joblib.load(os.path.join(BASE_DIR, 'model', 'model_metadata.pkl'))
except Exception:
    ml_model_metadata = {}

# Eligibility and benefit thresholds (minimum wage, VAT-exemption income
# cap, category duration minimums, etc.) all live in rules.json now — see
# engine.py's evaluate(), which app.py's assess_eligibility() calls into.
# Nothing in app.py should hardcode a legal threshold; if a number needs to
# change, change it in rules.json.

# Independent income-outlier safety net — separate from both the ML model
# and the rules engine. Neither was trained/validated on incomes far outside
# the training dataset's actual range (data/solo_parent_dataset.csv tops out
# at PHP 84,072/month), so their verdicts can't be trusted as "confidently
# decided" for applicants far beyond that, even if the model and the rules
# happen to agree. Update TRAINING_DATA_MAX_MONTHLY_INCOME if the training
# dataset is regenerated with a different income range.
TRAINING_DATA_MAX_MONTHLY_INCOME = 84072
INCOME_OUTLIER_MULTIPLE = 3
INCOME_OUTLIER_THRESHOLD = TRAINING_DATA_MAX_MONTHLY_INCOME * INCOME_OUTLIER_MULTIPLE  # ~252,216

# Presentation-only explanatory text for rules.json's benefits, keyed by
# each benefit's "id" field. This does NOT affect which benefits are
# granted — that's decided entirely by engine.evaluate() from rules.json.
# If rules.json gains a new benefit with no entry here, it still displays
# fine with a generic "RA 11861 Sec. X" fallback (see assess_eligibility()).
BENEFIT_DESCRIPTIONS = {
    'cash_subsidy_sec15a': 'PHP 1,000 monthly cash subsidy per solo parent (not per child), subject to fund availability.',
    'vat_discount_sec15b': "10% discount and VAT exemption on baby's milk, food, micronutrient supplements, sanitary diapers, and prescribed medicines/vaccines for children aged 0-6.",
    'philhealth_coverage_sec15c': 'Automatic PhilHealth/NHIP coverage for solo parents not already formally covered.',
    'livelihood_priority_sec15d': 'Priority access to livelihood, self-employment, and skills training programs.',
    'housing_priority_sec15e': 'Priority allocation and liberal payment terms in government socialized housing projects.',
    'educational_support_sec9': 'Full scholarship for one child, plus priority in UniFAST/tertiary education programs for other dependents.',
}

SOLO_PARENT_STATUS_LABELS = {
    'widowed': 'Widowed (Death of Spouse)',
    'abandoned': 'Abandoned by Spouse',
    'separated_divorced': 'Legally Separated',
    'single_parent_unmarried': 'Single Parent (Unmarried)',
    'spouse_detained_convicted': 'Spouse Detained / Convicted',
    'spouse_incapacitated_medical': 'Spouse Incapacitated (Medical)',
    'annulled_nullified_marriage': 'Annulled / Nullified Marriage',
    'guardian_adoptive_foster_parent': 'Guardian / Adoptive / Foster Parent',
    'relative_caregiver_4th_degree': 'Relative Caregiver (Up to 4th Degree)',
    'solo_grandparent_caregiver': 'Solo Grandparent Caregiver',
    'pregnant_woman_unborn_child': 'Pregnant Woman (Unborn Child)',
    'ofw_related_guardian': 'OFW-Related Guardian'
}

SOLO_PARENT_STATUS_DOCUMENTS = {
    'widowed': [
        'PSA/Local Civil Registry death certificate of spouse'
    ],
    'abandoned': [
        'Barangay certification, police blotter, or affidavit showing abandonment'
    ],
    'separated_divorced': [
        'Court decree/order of legal separation or divorce recognition, if available',
        'If no court order, barangay certification or sworn affidavit of de facto separation'
    ],
    'single_parent_unmarried': [
        'Affidavit of sole parental care and support',
        'Certificate of No Marriage (CENOMAR), when required by LGU'
    ],
    'spouse_detained_convicted': [
        'Certification of detention/commitment or prison record of spouse'
    ],
    'spouse_incapacitated_medical': [
        'Medical certificate proving physical/mental incapacity of spouse'
    ],
    'annulled_nullified_marriage': [
        'Court decision/decree of nullity or annulment (or recognized divorce decree)'
    ],
    'guardian_adoptive_foster_parent': [
        'Court/legal document proving guardianship, adoption, or foster care authority'
    ],
    'relative_caregiver_4th_degree': [
        'Proof of relationship within 4th civil degree and affidavit of assumption of care'
    ],
    'solo_grandparent_caregiver': [
        'Birth certificates/proof of filiation and affidavit showing grandparent as sole caregiver'
    ],
    'pregnant_woman_unborn_child': [
        'Medical certificate confirming pregnancy and affidavit of sole responsibility'
    ],
    'ofw_related_guardian': [
        'OFW contract/certification and affidavit/certification of sole caregiving by applicant'
    ]
}

COMMON_REQUIRED_DOCUMENTS = [
    'Duly accomplished Solo Parent application form',
    'Barangay certificate of residency and certification as solo parent applicant',
    'Birth certificate/s of dependent child/children (or proof of dependency)',
    'Valid government-issued ID of applicant',
    'Proof of income (payslip/employment certificate/ITR or barangay indigency certificate, when applicable)'
]


def format_display_label(value):
    text = str(value or '').strip()
    if not text:
        return ''
    return ' '.join(part.capitalize() for part in text.split())


def get_solo_parent_status_label(status_key):
    key = str(status_key or '').strip().lower()
    return SOLO_PARENT_STATUS_LABELS.get(key, format_display_label(key.replace('_', ' ')))


def get_required_documents_for_status(status_key):
    key = str(status_key or '').strip().lower()
    docs = list(COMMON_REQUIRED_DOCUMENTS)
    docs.extend(SOLO_PARENT_STATUS_DOCUMENTS.get(key, []))
    docs.append('2 pcs. 1x1 recent ID photo (if required by LGU/MSWDO)')
    return docs

SOLO_PARENT_FOLLOW_UP_QUESTIONS = {
    'abandoned': 'How long has your spouse been absent?',
    'separated_divorced': 'How long have you been separated from your spouse?',
    'ofw_related_guardian': 'How many months has the OFW been continuously abroad?',
    'relative_caregiver_4th_degree': 'How long have the parents been absent or unable to care for the child?',
    'solo_grandparent_caregiver': 'How long have you been the primary caregiver for your grandchild (i.e. how long have the parents been absent or unable to care for them)?',
    'spouse_detained_convicted': 'How many months has your spouse been detained or serving sentence?'
}

# Maps this form's solo_parent_status keys to the exact Type_of_Solo_Parent
# category strings used by rules.json / engine.py (and by the training
# dataset) — this is the single translation layer both the rules engine and
# the ML feature extractor rely on, so it lives here once instead of being
# duplicated. 'solo_grandparent_caregiver' intentionally maps to the same
# category as 'relative_caregiver_4th_degree': under RA 11861 a grandparent
# is a relative within the 4th civil degree, not a separate legal category,
# and rules.json only defines the one entry (with its 6-month duration
# minimum applying to both).
TYPE_OF_SOLO_PARENT_MAP = {
    'single_parent_unmarried': 'Unmarried parent who keeps and rears the child',
    'widowed': 'Death of spouse',
    'separated_divorced': 'Legal or de facto separation',
    'abandoned': 'Abandonment by spouse',
    'ofw_related_guardian': 'Spouse/family member/guardian of an OFW',
    'annulled_nullified_marriage': 'Declaration of nullity, annulment, or divorce',
    'spouse_detained_convicted': 'Detention of spouse (3+ months) or serving sentence',
    'spouse_incapacitated_medical': 'Physical or mental incapacity of spouse',
    'pregnant_woman_unborn_child': 'Pregnant woman solely responsible for unborn child',
    'guardian_adoptive_foster_parent': 'Legal guardian, adoptive, or foster parent',
    'relative_caregiver_4th_degree': 'Relative within 4th civil degree assuming care',
    'solo_grandparent_caregiver': 'Relative within 4th civil degree assuming care',
    # RA 11861 and rules.json also define this category, but the current
    # form has no corresponding option:
    # 'rape': 'Birth of child due to rape'
}

# ---------------------------------------------------------------------------
# Submission integrity guards (see LOOSE_LOGIC.md)
# ---------------------------------------------------------------------------
# Core fields a real assessment cannot be produced without. The HTML form
# marks these required, but a direct POST bypasses that — this list is the
# server-side backstop so a malformed submission is rejected instead of
# silently producing a result/PDF from blanks. (#10)
REQUIRED_SUBMISSION_FIELDS = [
    ('first_name', 'First name'),
    ('surname', 'Surname'),
    ('birthday', 'Birthday'),
    ('sex', 'Sex'),
    ('civil_status', 'Civil status'),
    ('monthly_income', 'Monthly income'),
    ('contact_no', 'Contact number'),
    ('solo_parent_status', 'Reason for being a solo parent'),
    ('with_pwd', 'PWD registration answer'),
    ('receiving_other_govt_cash_aid', 'Other government cash aid answer'),
    ('formal_philhealth_member', 'PhilHealth membership answer'),
    ('dependent_currently_studying', 'Dependent currently studying answer'),
    # number_of_dependent_children and youngest_child_age are intentionally
    # NOT here: they're derived from the family composition
    # (derive_dependent_children), not entered by the applicant.
]

# Which solo-parent reasons are logically consistent with each civil status
# (#4). Reasons tied to the applicant's OWN marriage are gated; guardianship,
# caregiver, grandparent, pregnancy, and OFW-related reasons are not about the
# applicant's marital status and are allowed under any civil status. This is a
# policy default meant to block clear contradictions (e.g. "Single" + a
# spouse-based reason) without over-blocking plausible cases — tune here.
_CIVIL_STATUS_NEUTRAL_REASONS = {
    'guardian_adoptive_foster_parent',
    'relative_caregiver_4th_degree',
    'solo_grandparent_caregiver',
    'pregnant_woman_unborn_child',
    'ofw_related_guardian',
}
CIVIL_STATUS_ALLOWED_REASONS = {
    'single':    {'single_parent_unmarried'} | _CIVIL_STATUS_NEUTRAL_REASONS,
    'married':   {'abandoned', 'spouse_detained_convicted', 'spouse_incapacitated_medical'} | _CIVIL_STATUS_NEUTRAL_REASONS,
    'separated': {'separated_divorced', 'abandoned', 'spouse_detained_convicted', 'spouse_incapacitated_medical'} | _CIVIL_STATUS_NEUTRAL_REASONS,
    'annulled':  {'annulled_nullified_marriage'} | _CIVIL_STATUS_NEUTRAL_REASONS,
    'divorced':  {'annulled_nullified_marriage'} | _CIVIL_STATUS_NEUTRAL_REASONS,
    'widowed':   {'widowed'} | _CIVIL_STATUS_NEUTRAL_REASONS,
}

# Enum whitelists. The consistency guards above are keyed on these values, so
# an unrecognized value must FAIL validation rather than silently skip the
# checks keyed on it (e.g. civil_status="widow" previously bypassed the whole
# civil-status/reason matrix, and sex="m" slipped past the male+pregnancy
# guard). Valid reasons are TYPE_OF_SOLO_PARENT_MAP's keys; valid civil
# statuses are CIVIL_STATUS_ALLOWED_REASONS's keys.
VALID_SEX_VALUES = {'male', 'female'}

# Absolute ceiling on a claimed circumstance duration, independent of age
# (#8). 100 years in months — anything beyond is a data-entry error.
MAX_DURATION_MONTHS = 1200

# Legal minimum age to marry in the Philippines (Family Code). A household
# member below this cannot legitimately be recorded as Married or Widowed.
MINIMUM_MARRIAGE_AGE = 18

# Minimum plausible age to have reached each educational attainment level.
# Used to catch impossible family-member rows (e.g. a 5-year-old "College").
# 'Pre-school' / blank have no minimum.
EDUCATION_MINIMUM_AGE = {
    'elementary': 5,
    'high school': 12,
    'vocational': 15,
    'college': 16,
    'postgraduate': 21,
}

# Minimum plausible age gap between a parent and their child. Used to catch
# impossible household rows such as a 26-year-old listed as the applicant's
# "Child" when the applicant is not old enough to be their parent.
MIN_PARENT_CHILD_AGE_GAP = 12


# Relationships that can represent a dependent child, counting toward
# "Number of Dependent Children" and "Age of Youngest Dependent Child".
# Both "Dependent" and "Child" rows are considered — whether a row actually
# COUNTS is decided by RA 11861's qualification criteria (unmarried,
# unemployed, within the age cap), so a grown/married/employed child is
# excluded by those filters, not by the relationship label. "Child" used to
# be informational-only, but that silently rejected any parent who naturally
# listed their kid as "Child" instead of "Dependent".
DEPENDENT_RELATIONSHIPS = {'dependent', 'child'}

# Minimum lawful working age in the Philippines (RA 9231 permits light work
# from age 15). A member younger than this is never counted as "employed"
# from their occupation text — parents often type things like "Grade 1",
# "Pupil", or "Daycare" in a young child's occupation column, none of which
# should disqualify the child as a dependent. A declared income still counts
# as earning regardless of age.
MINIMUM_EMPLOYMENT_AGE = 15

# Occupation values (beyond the applicant-level NO_OCCUPATION_SYNONYMS, which
# is defined later and resolved at call time) that mean a household member is
# NOT employed and so can still be a dependent.
_EXTRA_NOT_EMPLOYED_OCCUPATIONS = {
    'student', 'studying', 'schooling', 'minor', 'baby', 'infant', 'toddler',
    'child', 'kid', 'preschool', 'pre-school'
}


def _member_is_employed(member):
    # A declared income makes the member employed/earning regardless of what
    # the occupation text says — otherwise a "Dependent" with ₱30,000/month
    # income and a blank occupation would still count as an unemployed
    # dependent, sidestepping RA 11861's UNEMPLOYED requirement while the
    # form openly holds the contradicting evidence.
    try:
        if float(member.get('income') or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    # Below the minimum working age, occupation text can't mean employment.
    try:
        if int(str(member.get('age', '')).strip()) < MINIMUM_EMPLOYMENT_AGE:
            return False
    except (TypeError, ValueError):
        pass
    occ = str(member.get('occupation', '')).strip().lower().strip('.')
    if not occ or occ in NO_OCCUPATION_SYNONYMS or occ in _EXTRA_NOT_EMPLOYED_OCCUPATIONS:
        return False
    return True


def _member_is_pwd(member):
    return str(member.get('pwd', '')).strip().lower() in ('yes', 'true', '1')


# Education levels that satisfy the Sec. 9 educational-support scholarship.
# RA 11861 Sec. 9 grants a scholarship for basic/higher/technical-vocational
# education, but this system scopes it specifically to High School or College
# per policy decision — Pre-school, Elementary, Vocational, and Postgraduate
# are excluded.
SCHOLARSHIP_ELIGIBLE_EDUCATION_LEVELS = {'high school', 'college'}


def derive_dependent_children(family_members, is_pregnancy=False):
    """Work out the dependent-child figures from the family composition, per
    RA 11861's statutory definition of "children or dependents":

        "...those living with and dependent upon the solo parent for support
        who are UNMARRIED, UNEMPLOYED and twenty-two (22) years old or below,
        or those over twenty-two (22) years old but who are unable to fully
        take care or protect themselves ... because of a physical or mental
        disability or condition ... this definition shall only apply for
        purposes of availing the benefits under this Act."

    So a household member counts as a DEPENDENT CHILD only if their
    relationship is "Dependent" or "Child" AND they are unmarried AND
    UNEMPLOYED AND within the dependent age cap (22 or younger, or older
    only if that member is a PWD). The qualification criteria — not the
    relationship label — do the filtering: a grown, married, or employed
    child is excluded by them automatically. An EMPLOYED member does NOT
    qualify as a dependent at all — there is no exception for this in the
    law. If employment removes the applicant's only dependent, the count
    drops to zero and the applicant is Not Eligible (see assess_eligibility,
    which surfaces the specific reason).

    Returns a dict:
      count             -> Number of Dependent Children (incl. the unborn for pregnancy)
      youngest          -> Age of Youngest Dependent Child (-1 for an unborn)
      excluded_employed -> names of "Dependent" rows disqualified for being
                            employed (or for having a declared income)
      any_pwd           -> whether any counted dependent is a PWD
      school_level_ok   -> whether any counted dependent's education is High
                            School or College (gates the Sec. 9 scholarship)
    """
    max_dependent_age = ENGINE_RULES['thresholds'].get('max_dependent_age', 22)
    ages = []
    excluded_employed = []
    any_pwd = False
    school_level_ok = False
    for m in family_members:
        if str(m.get('relationship', '')).strip().lower() not in DEPENDENT_RELATIONSHIPS:
            continue
        if str(m.get('civil_status', '')).strip().lower() in ('married', 'widowed'):
            continue  # a dependent child is unmarried
        try:
            age = int(str(m.get('age', '')).strip())
        except (TypeError, ValueError):
            continue
        if age < 0:
            continue
        member_pwd = _member_is_pwd(m)
        if age > max_dependent_age and not member_pwd:
            # RA 11861's over-22 exception is for a dependent unable to care
            # for THEMSELVES due to disability — it must be the DEPENDENT
            # who is a PWD. The applicant's own PWD status raises priority
            # scoring but cannot make an able-bodied over-22 member a
            # qualifying dependent.
            continue
        if _member_is_employed(m):
            # RA 11861's dependent definition requires UNEMPLOYED with no
            # exception — an employed member is not a dependent under this
            # Act, no matter their age or how the household actually relies
            # on them.
            excluded_employed.append(str(m.get('name', '')).strip() or 'a dependent')
            continue
        ages.append(age)
        if member_pwd:
            any_pwd = True
        if str(m.get('education', '')).strip().lower() in SCHOLARSHIP_ELIGIBLE_EDUCATION_LEVELS:
            school_level_ok = True

    count = len(ages)
    youngest = min(ages) if ages else None
    if is_pregnancy:
        youngest = -1               # unborn child is the youngest
        count += 1                  # count the unborn child
    return {
        'count': count,
        'youngest': youngest,
        'excluded_employed': excluded_employed,
        'any_pwd': any_pwd,
        'school_level_ok': school_level_ok,
    }


def validate_family_members(family_members, applicant_age=None):
    """Cross-field sanity checks on each family-composition row: an age must
    be consistent with the member's civil status, educational attainment, and
    (relative to the applicant) their relationship. Catches impossible rows
    such as a 5-year-old marked Married, a toddler with a College education,
    or a 26-year-old listed as the applicant's Child. Returns a list of hard
    error strings."""
    errors = []
    for m in family_members:
        name = str(m.get('name', '')).strip() or 'A family member'
        try:
            age = int(str(m.get('age', '')).strip())
        except (TypeError, ValueError):
            continue  # no usable age -> nothing to cross-check

        if age < 0:
            errors.append(f"{name}: age cannot be negative.")
            continue

        # The form derives each row's age from its birthday client-side, so
        # the two can only disagree via a tampered/scripted POST or a stale
        # value — reconcile them here. ±1 year of slack covers a birthday
        # passing between page load and submission.
        birthday_text = str(m.get('birthday', '')).strip()
        if birthday_text:
            try:
                born = datetime.strptime(birthday_text, '%Y-%m-%d')
            except ValueError:
                born = None
            if born is not None:
                today = datetime.today()
                derived_age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
                if derived_age >= 0 and abs(derived_age - age) > 1:
                    errors.append(
                        f"{name}: age {age} does not match their birthday "
                        f"({birthday_text} makes them about {derived_age})."
                    )

        civil = str(m.get('civil_status', '')).strip().lower()
        relationship = str(m.get('relationship', '')).strip().lower()

        if civil in ('married', 'widowed') and age < MINIMUM_MARRIAGE_AGE:
            errors.append(
                f"{name}: age {age} cannot have a civil status of "
                f"{m.get('civil_status')} (minimum marriage age is {MINIMUM_MARRIAGE_AGE})."
            )

        # A "Dependent" IS a dependent child by definition (see
        # derive_dependent_children), and a dependent child must be
        # unmarried. Letting someone be entered as both "Dependent" and
        # "Married"/"Widowed" is a direct contradiction that was previously
        # accepted silently (the row just got excluded from the count with
        # no explanation) — now it's flagged instead, so the applicant knows
        # to either fix the civil status or change the relationship to
        # "Child" if this member is genuinely not a dependent.
        if relationship == 'dependent' and civil in ('married', 'widowed'):
            errors.append(
                f'{name}: cannot be marked as "Dependent" while their civil status is '
                f'{m.get("civil_status")}; a dependent child must be unmarried.'
            )

        education = str(m.get('education', '')).strip().lower()
        min_edu_age = EDUCATION_MINIMUM_AGE.get(education)
        if min_edu_age is not None and age < min_edu_age:
            errors.append(
                f"{name}: age {age} is too young for an educational attainment of "
                f"{m.get('education')} (typically age {min_edu_age}+)."
            )

        # Relationship vs. applicant's age: a Child must be younger than the
        # applicant by at least a plausible parenthood gap; a Parent must be
        # correspondingly older.
        if applicant_age:
            if relationship == 'child' and (applicant_age - age) < MIN_PARENT_CHILD_AGE_GAP:
                errors.append(
                    f"{name}: age {age} is not consistent with being the applicant's child "
                    f"(the applicant is {applicant_age}; a parent is normally at least "
                    f"{MIN_PARENT_CHILD_AGE_GAP} years older than their child)."
                )
            elif relationship == 'parent' and (age - applicant_age) < MIN_PARENT_CHILD_AGE_GAP:
                errors.append(
                    f"{name}: age {age} is not consistent with being the applicant's parent "
                    f"(the applicant is {applicant_age}; a parent is normally at least "
                    f"{MIN_PARENT_CHILD_AGE_GAP} years older)."
                )

    return errors


def evaluate_submission_guards(data, *, solo_parent_status, civil_status, sex,
                               monthly_income, youngest_child_age_raw,
                               duration_months, number_of_children, age,
                               family_members, dependent_currently_studying):
    """Cross-field integrity checks that live OUTSIDE the rules engine (the
    engine stays a pure, dataset-validated RA 11861 evaluator). Returns
    (hard_errors, soft_warnings):
      - hard_errors  -> the submission is invalid/impossible; caller marks it
                        Not Eligible and shows the reasons.
      - soft_warnings-> plausible but suspicious; caller flags for MSWDO/DSWD
                        verification without blocking.
    """
    hard_errors = []
    soft_warnings = []

    # #10 required fields (server-side backstop for direct POSTs)
    for field_name, label in REQUIRED_SUBMISSION_FIELDS:
        if not str(data.get(field_name, '')).strip():
            hard_errors.append(f"Missing required field: {label}.")

    # Enum whitelists: an unrecognized sex/civil-status/reason value must
    # fail outright — previously it silently BYPASSED every consistency
    # check keyed on it (e.g. civil_status="widow" skipped the whole
    # civil-status/reason matrix, and sex="m" slipped past the pregnancy
    # guard). Blank values are already reported by the required-field check.
    sex_key = sex.strip().lower()
    if sex_key and sex_key not in VALID_SEX_VALUES:
        hard_errors.append(f'Unrecognized sex value "{sex}".')
    civil_key = civil_status.strip().lower()
    if civil_key and civil_key not in CIVIL_STATUS_ALLOWED_REASONS:
        hard_errors.append(f'Unrecognized civil status "{civil_status}".')
    if solo_parent_status and solo_parent_status not in TYPE_OF_SOLO_PARENT_MAP:
        hard_errors.append(f'Unrecognized reason for being a solo parent "{solo_parent_status}".')

    # #7 income cannot be negative (rejecting, not clamping — clamping a
    # negative to 0 would wrongly unlock the lowest, most generous tier)
    if monthly_income < 0:
        hard_errors.append("Monthly income cannot be negative.")

    # Non-numeric money text is rejected for the same reason negatives are:
    # parse_money() maps it to 0.0, and ₱0 unlocks the most generous benefit
    # tier — so "N/A" income would silently be treated as the most favorable
    # possible answer.
    if invalid_money_text(data.get('monthly_income')):
        hard_errors.append("Monthly income must be a numeric amount (e.g. 15000).")
    for key in data:
        if key.startswith('family_income_') and invalid_money_text(data.get(key)):
            hard_errors.append("Family member income must be a numeric amount.")
            break

    # Occupation vs. declared income consistency (soft — informal/irregular
    # income and freshly-started jobs are legitimate, so these flag for
    # MSWDO/DSWD verification rather than block). The inconsistent pair
    # matters because benefits gate on the two fields separately: employment
    # status gates Sec. 15(d), income gates 15(a)/(b)/(e).
    employment_status = infer_employment_status(data.get('occupation', ''))
    min_wage = ENGINE_RULES['thresholds'].get('min_wage_monthly', 16000)
    if employment_status == 'Unemployed' and monthly_income > min_wage:
        soft_warnings.append(
            f"Occupation indicates unemployed, but declared monthly income is "
            f"PHP {monthly_income:,.0f} (above the PHP {min_wage:,} minimum wage)."
        )
    elif employment_status in ('Employed', 'Self-Employed', 'Part-Time') and monthly_income == 0:
        soft_warnings.append(
            f"Occupation indicates {employment_status.lower()}, but declared monthly income is PHP 0."
        )

    # #3 pregnancy category requires female applicant
    if solo_parent_status == 'pregnant_woman_unborn_child' and sex.strip().lower() == 'male':
        hard_errors.append("The pregnancy category cannot be selected with sex recorded as Male.")

    # #5 the -1 "unborn" sentinel is only meaningful for the pregnancy category
    youngest_raw = str(youngest_child_age_raw).strip()
    if youngest_raw == '-1' and solo_parent_status != 'pregnant_woman_unborn_child':
        hard_errors.append("Youngest child age -1 (unborn) is only valid for the pregnant-woman category.")

    # #4 civil status vs. reason consistency
    allowed = CIVIL_STATUS_ALLOWED_REASONS.get(civil_status.strip().lower())
    if allowed is not None and solo_parent_status and solo_parent_status not in allowed:
        hard_errors.append(
            f'"{get_solo_parent_status_label(solo_parent_status)}" is not consistent with a '
            f'civil status of "{format_civil_status(civil_status)}".'
        )

    # #8 duration sanity: absolute cap, and not longer than the applicant has
    # been alive
    if duration_months > MAX_DURATION_MONTHS:
        hard_errors.append(f"Reported duration ({duration_months} months) is implausibly large.")
    elif age and duration_months > age * 12:
        hard_errors.append(
            f"Reported duration ({duration_months} months) exceeds the applicant's age "
            f"({age} years) — the circumstance cannot predate the applicant."
        )

    # #6 is now structurally impossible: the dependent count is DERIVED from
    # the qualifying family-composition rows (derive_dependent_children), so it
    # can no longer disagree with the roster.

    # #11 "a dependent is currently studying" but the only dependent is too
    # young to be enrolled (soft — older siblings may exist if >1 child)
    try:
        youngest_int = int(youngest_raw)
    except (TypeError, ValueError):
        youngest_int = None
    if (dependent_currently_studying and number_of_children == 1
            and youngest_int is not None and 0 <= youngest_int < 3):
        soft_warnings.append(
            "Marked a dependent as currently studying, but the only dependent is under 3 years old."
        )

    # Family-composition row consistency (age vs civil status / education /
    # relationship-to-applicant)
    hard_errors.extend(validate_family_members(family_members, applicant_age=age))

    # Suggestion #4 (revised): duplicate family members. The key is name AND
    # birthday, not name alone — a father and son sharing a name (Sr./Jr.)
    # are a legitimate household, not a duplicate.
    seen = {}
    for m in family_members:
        name_key = str(m.get('name', '')).strip().lower()
        if not name_key:
            continue
        key = (name_key, str(m.get('birthday', '')).strip())
        seen[key] = seen.get(key, 0) + 1
    for (name_key, _birthday), n in seen.items():
        if n > 1:
            hard_errors.append(
                f'"{name_key.title()}" is listed {n} times in the family composition with the '
                f'same birthday; remove the duplicate.'
            )

    # Suggestion #5: contact number must be a valid PH mobile number.
    contact_error = validate_ph_contact(data.get('contact_no', ''))
    if contact_error:
        hard_errors.append(contact_error)

    return hard_errors, soft_warnings


def validate_ph_contact(contact_no):
    """Validate a Philippine mobile number. Accepts 11 digits starting 09
    (09XXXXXXXXX) or the +63 form (639XXXXXXXXX). Returns an error string, or
    None if valid/blank."""
    digits = ''.join(ch for ch in str(contact_no or '') if ch.isdigit())
    if not digits:
        return None  # emptiness is handled by the required-field check
    if digits.startswith('63'):
        digits = '0' + digits[2:]  # normalize +63 / 63 prefix to 0
    if len(digits) == 11 and digits.startswith('09'):
        return None
    return "Contact number must be a valid Philippine mobile number (e.g. 09XXXXXXXXX)."


def format_civil_status(value):
    status_map = {
        'single': 'Single',
        'married': 'Married',
        'separated': 'Separated',
        'divorced': 'Divorced',
        'annulled': 'Annulled',
        'widowed': 'Widowed'
    }
    key = str(value or '').strip().lower()
    return status_map.get(key, format_display_label(value))


def calculate_age(birthdate_str):
    """Calculate age from birthdate string YYYY-MM-DD format"""
    try:
        birthdate = datetime.strptime(birthdate_str, '%Y-%m-%d')
        today = datetime.today()
        age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
        return age
    except:
        return 0


def parse_money(value):
    """Parses a currency-style form value into a float, stripping thousands
    separators first (e.g. "5,000,000" -> 5000000.0). Without this, a bare
    float(value) call raises on any comma-formatted input and gets caught
    by a bare except, silently defaulting to 0 — which previously made a
    ₱5,000,000 income submission behave as if it were ₱0, incorrectly
    skipping both the income-outlier check and the benefit-tier income
    gating. Returns 0.0 for blank or otherwise unparseable input."""
    if value is None:
        return 0.0
    text = str(value).replace(',', '').strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def invalid_money_text(value):
    """True if the value is non-blank but not parseable as a number — the
    case parse_money() silently maps to 0.0. Callers reject such input as a
    hard error (see evaluate_submission_guards): treating "N/A" as ₱0 would
    grant the lowest-income, most generous benefit tier. Blank stays valid
    here; required-ness is the required-field check's job."""
    text = str(value or '').replace(',', '').strip()
    if not text:
        return False
    try:
        float(text)
        return False
    except ValueError:
        return True


def parse_family_members(form_data):
    """Parse family members safely, even if rows are deleted"""
    family_members = []

    indexes = []
    for key in form_data.keys():
        if key.startswith('family_name_'):
            try:
                indexes.append(int(key.replace('family_name_', '')))
            except:
                pass

    for index in sorted(indexes):
        name = form_data.get(f'family_name_{index}', '').strip()

        if name:
            income = parse_money(form_data.get(f'family_income_{index}', 0))

            family_members.append({
                'name': name,
                'relationship': form_data.get(f'family_relationship_{index}', ''),
                'birthday': form_data.get(f'family_birthday_{index}', ''),
                'age': form_data.get(f'family_age_{index}', '0'),
                'civil_status': form_data.get(f'family_civil_status_{index}', ''),
                'education': form_data.get(f'family_educ_{index}', ''),
                'occupation': form_data.get(f'family_occupation_{index}', ''),
                'pwd': form_data.get(f'family_pwd_{index}', 'No'),
                'income': income
            })

    return family_members


# Free-text "no occupation" synonyms applicants commonly write instead of
# the literal word "unemployed" — previously only 'unemployed' and 'none'
# were recognized, so e.g. "N/A" silently fell through to the 'Employed'
# default below, which is exactly backwards for an unrecognized answer.
NO_OCCUPATION_SYNONYMS = {
    'unemployed', 'none', 'n/a', 'na', 'not applicable', 'nil', 'n a',
    'wala', 'walang trabaho', 'walang hanapbuhay', 'no occupation',
    'no work', 'no job', '-', '--', 'none.'
}


# Whole-word patterns only. The previous substring test (`word in occ`)
# misclassified ordinary occupations — "Receptionist" and "Sculptor" contain
# "pt" and became Part-Time (which grants the Sec. 15(d) livelihood benefit),
# "Town Clerk" contains "own" and became Self-Employed.
_SELF_EMPLOYED_RE = re.compile(r'\b(own|owner|self|business|freelance|freelancer|vendor)\b')
_PART_TIME_RE = re.compile(r'\b(part[\s-]?time|pt)\b')


def infer_employment_status(occupation_text):
    """Infer employment status from occupation text field"""
    if not occupation_text:
        return 'Unemployed'

    occ = occupation_text.lower().strip().strip('.')

    if occ in NO_OCCUPATION_SYNONYMS or len(occ) == 0:
        return 'Unemployed'
    elif _SELF_EMPLOYED_RE.search(occ):
        return 'Self-Employed'
    elif _PART_TIME_RE.search(occ):
        return 'Part-Time'
    else:
        return 'Employed'


def has_minor_dependents(family_members):
    """Check if there are any minor dependents (age < 18)"""
    for member in family_members:
        try:
            age = int(member.get('age', 0))
            if age > 0 and age < 18:
                return 'Yes'
        except:
            pass
    return 'No'


def extract_ml_features(form_data):
    """Extract and transform form data to ML model features"""
    family_members = parse_family_members(form_data)

    try:
        birthday = form_data.get('birthday', '')
        age = calculate_age(birthday)
    except:
        age = 0

    # IMPORTANT: these values must match the exact Type_of_Solo_Parent strings
    # used in data/solo_parent_dataset.csv (and rules.json's category keys),
    # or the encoder / rules engine will treat the submission as an unseen
    # category. See TYPE_OF_SOLO_PARENT_MAP for the shared mapping.
    solo_parent_type = form_data.get('solo_parent_status', '').strip().lower()
    # Fallback for a value the form didn't expect: use the most common
    # legitimate category rather than silently guessing. This should be rare
    # once the form's option list matches TYPE_OF_SOLO_PARENT_MAP's keys.
    mapped_type = TYPE_OF_SOLO_PARENT_MAP.get(solo_parent_type, 'Unmarried parent who keeps and rears the child')

    gender_map = {'male': 'M', 'female': 'F'}
    mapped_gender = gender_map.get(form_data.get('sex', '').strip().lower(), 'F')

    # Civil_Status is trained as 5 distinct categories — 'Married' and
    # 'Separated' are NOT the same category, so they must not be collapsed
    # into one mapped value.
    civil_status_map = {
        'single': 'Single',
        'married': 'Married',
        'separated': 'Separated',
        'divorced': 'Annulled',
        'annulled': 'Annulled',
        'widowed': 'Widow'
    }
    mapped_civil_status = civil_status_map.get(form_data.get('civil_status', '').strip().lower(), 'Single')

    features = {
        'Age': age,
        'Gender': mapped_gender,
        'Civil_Status': mapped_civil_status,
        'Educational_Attainment': form_data.get('educational_attainment', '').strip() or 'Elementary',
        'Employment_Status': infer_employment_status(form_data.get('occupation', '')),
        'Monthly_Income': parse_money(form_data.get('monthly_income', 0)),
        'Number_of_Dependents': int(form_data.get('number_of_dependent_children', 0) or 0),
        'With_Minor': has_minor_dependents(family_members),
        'With_PWD': form_data.get('with_pwd', 'No'),
        'Type_of_Solo_Parent': mapped_type
    }

    return features


def assess_eligibility_ml(form_data):
    """ML-based eligibility assessment using pre-trained Pipeline model"""
    if not ml_model_loaded:
        return {
            'eligible': False,
            'confidence': 0.0,
            'model_status': 'unavailable'
        }

    try:
        features = extract_ml_features(form_data)

        feature_df = pd.DataFrame([features])

        # Align and reorder the features to match the training feature list.
        feature_df = feature_df[ml_feature_columns]

        # Two model formats are supported here:
        #
        # 1) Self-contained sklearn Pipeline (train_model_fixed.py v4.0+):
        #    the model's own ColumnTransformer does encoding internally and
        #    expects RAW text/category values. encoders.pkl is an empty
        #    dict {} in this case — nothing to do here, pass features
        #    straight through.
        #
        # 2) Legacy flat estimator (v3.0 and earlier): trained on
        #    LabelEncoded columns. encoders.pkl holds one fitted
        #    LabelEncoder per categorical column, and every value must be
        #    manually encoded, then the whole row cast to float, before
        #    calling predict().
        #
        # Previously this function unconditionally ran the v3.0-style
        # manual encoding + astype(float) path. With a v4.0 pipeline model
        # (empty encoders.pkl), no columns ever matched `col in ml_encoders`,
        # so nothing got encoded, and astype(float) then raised
        # "could not convert string to float" on every single request —
        # silently caught below, always returning 0% confidence / error.
        if ml_encoders:
            categorical_cols = [
                'Educational_Attainment', 'Employment_Status',
                'With_Minor', 'With_PWD', 'Type_of_Solo_Parent'
            ]
            unseen_category_warnings = []
            for col in categorical_cols:
                if col not in feature_df.columns or col not in ml_encoders:
                    continue
                encoder = ml_encoders[col]
                raw_value = feature_df.at[0, col]
                if raw_value in encoder.classes_:
                    feature_df.at[0, col] = encoder.transform([raw_value])[0]
                else:
                    # Unseen category: fall back to the encoder's first known
                    # class so prediction can still run, but flag it — this
                    # should be rare if TYPE_OF_SOLO_PARENT_MAP is kept in
                    # sync with the training data's category strings.
                    unseen_category_warnings.append(f"{col}='{raw_value}'")
                    feature_df.at[0, col] = encoder.transform([encoder.classes_[0]])[0]

            if unseen_category_warnings:
                print(f"ML feature warning — unseen categories, using fallback encoding: {', '.join(unseen_category_warnings)}")

            feature_df = feature_df.astype(float)
        # else: self-contained pipeline model — leave feature_df as raw
        # text/category values; the pipeline's own preprocessor handles it.

        prediction = ml_model.predict(feature_df)[0]
        probabilities = ml_model.predict_proba(feature_df)[0]

        classes = list(ml_model.classes_)

        if len(classes) == 1:
            prob_pred = 1.0
        else:
            try:
                idx_pred = classes.index(prediction)
                prob_pred = float(probabilities[idx_pred])
            except ValueError:
                prob_pred = float(max(probabilities))

        if 1 in classes:
            prob_eligible = float(probabilities[classes.index(1)])
        else:
            prob_eligible = prob_pred if prediction == 1 else (1.0 - prob_pred)

        prob_not_eligible = 1.0 - prob_eligible

        result = {
            'eligible': bool(prediction == 1),
            'confidence': prob_pred,
            'prob_eligible': prob_eligible,
            'prob_not_eligible': prob_not_eligible,
            'model_status': 'success'
        }

        return result
    except Exception as e:
        print(f"ML prediction error: {e}")
        return {
            'eligible': False,
            'confidence': 0.0,
            'model_status': 'error',
            'error': str(e),
            'prob_eligible': 0.0,
            'prob_not_eligible': 1.0
        }





def assess_eligibility(data):
    """
    Rule-based eligibility assessment based on RA 11861.
    Returns eligibility status and qualified benefits.
    """
    result = {
        'eligible': False,
        'needs_verification': False,
        'benefits': [],
        'applicant_info': {},
        'family_members': []
    }

    # Extract personal information
    first_name = data.get('first_name', '').strip()
    middle_name = data.get('middle_name', '').strip()
    surname = data.get('surname', '').strip()
    birthday = data.get('birthday', '')
    sex = data.get('sex', '')
    address = data.get('address', '').strip()
    contact_no = data.get('contact_no', '').strip()
    civil_status = data.get('civil_status', '')
    occupation = data.get('occupation', '').strip()

    monthly_income = parse_money(data.get('monthly_income', 0))

    solo_parent_status = data.get('solo_parent_status', '').strip().lower()
    status_specific_answer = data.get('category_duration_answer', '').strip()
    solo_parent_reason = data.get('solo_parent_reason', '').strip()

    # Calculate applicant age
    age = calculate_age(birthday)

    # Parse family members
    family_members = parse_family_members(data)

    # The dependent-children count and the youngest dependent's age are DERIVED
    # authoritatively from the family composition — only household members who
    # genuinely qualify as dependent children (a Child/Dependent who is
    # unmarried and within the dependent age cap) count. A grown or married
    # "Child" row does not. The posted number_of_dependent_children /
    # youngest_child_age fields (read-only, auto-filled client-side) are not
    # trusted here.
    is_pregnancy = solo_parent_status == 'pregnant_woman_unborn_child'
    applicant_with_pwd = data.get('with_pwd', 'No') == 'Yes'
    dep_info = derive_dependent_children(family_members, is_pregnancy=is_pregnancy)
    number_of_children = dep_info['count']
    youngest_child_age = dep_info['youngest']
    # The engine's PWD flag (priority scoring + the over-22 dependent
    # exception) is true if the applicant OR any counted dependent is a PWD.
    with_pwd = applicant_with_pwd or dep_info['any_pwd']
    # Sec. 9 scholarship: only if the applicant declared a dependent is
    # currently studying AND that qualifying dependent's education level is
    # High School or College (see SCHOLARSHIP_ELIGIBLE_EDUCATION_LEVELS).
    dependent_currently_studying = (
        (data.get('dependent_currently_studying', 'No') == 'Yes')
        and dep_info['school_level_ok']
    )

    # #12: Total family income is recomputed authoritatively from the family
    # composition rows here, rather than trusting the (read-only, but still
    # POST-able) total_family_income field the client submits.
    total_family_income = sum(float(m.get('income', 0) or 0) for m in family_members)

    # Store applicant info
    result['applicant_info'] = {
        'first_name': first_name,
        'middle_name': middle_name,
        'surname': surname,
        'name': f"{first_name} {middle_name} {surname}".replace('  ', ' ').strip(),

        'birthday': birthday,
        'birth_place': data.get('birthplace', '').strip(),
        'age': age,
        'sex': sex.title(),

        'address': address,
        'contact': contact_no,
        'landline': data.get('landline', '').strip(),

        'civil_status': format_civil_status(civil_status),
        'education': data.get('educational_attainment', '').strip(),
        'religion': data.get('religion', '').strip(),

        'occupation': occupation,
        'employer': data.get('employer_name', '').strip(),
        'company_address': data.get('company_address', '').strip(),
        'office_contact': data.get('office_contact', '').strip(),

        'monthly_income': monthly_income,
        'total_family_income': total_family_income,

        'solo_parent_status': get_solo_parent_status_label(solo_parent_status),
        'solo_parent_reason': solo_parent_reason,
        'status_specific_question': SOLO_PARENT_FOLLOW_UP_QUESTIONS.get(solo_parent_status, ''),
        'status_specific_answer': status_specific_answer,
        'number_of_children': number_of_children
    }

    result['required_documents'] = get_required_documents_for_status(solo_parent_status)

    result['family_members'] = family_members

    # Eligibility and benefits are decided entirely by engine.evaluate(),
    # which reads rules.json — the single source of truth shared with the
    # client-side app in client/. This function's job is just translating
    # this form's field names/values into the Applicant shape the engine
    # expects; no eligibility thresholds or category lists live here.
    rule_applicant = build_rule_engine_applicant(data, occupation=occupation,
                                                  solo_parent_status=solo_parent_status,
                                                  status_specific_answer=status_specific_answer,
                                                  number_of_children=number_of_children,
                                                  monthly_income=monthly_income,
                                                  age=age,
                                                  youngest_child_age=youngest_child_age,
                                                  with_pwd=with_pwd,
                                                  dependent_currently_studying=dependent_currently_studying)
    engine_result = rule_engine_evaluate(rule_applicant)

    result['eligible'] = engine_result['eligibility'] == 'Eligible'
    result['remarks'] = engine_result['remarks']
    result['priority_level'] = engine_result['priority_level']

    # If the engine rejected this applicant specifically for having no
    # qualifying dependent, AND that's because their only "Dependent" row(s)
    # are employed, make the reason explicit rather than the engine's generic
    # remark — RA 11861's dependent definition requires "unemployed" with no
    # exception, so an employed dependent does not count for benefit-availing
    # purposes. Guarded on the engine's own remark so this doesn't
    # misattribute an unrelated rejection reason (e.g. duration too short)
    # to employment just because an employed row also happens to be present.
    if (not result['eligible'] and number_of_children == 0
            and 'No qualifying dependent' in engine_result.get('remarks', '')
            and dep_info['excluded_employed']):
        names = ', '.join(dep_info['excluded_employed'])
        result['remarks'] = (
            f"No qualifying dependent: {names} is employed or has a declared income. Under RA 11861, a dependent must "
            f"be unmarried, UNEMPLOYED, and 22 years old or below (or older only if unable to "
            f"care for themselves due to disability) to count as a dependent for purposes of "
            f"availing benefits under this Act; an employed child does not qualify."
        )
    # The old borderline-income "pending verification" tier doesn't exist
    # in rules.json's model (each benefit is a clean granted/not-granted
    # boolean); needs_verification is now only set by the ML-conflict and
    # income-outlier safety nets in submit_assessment().
    result['needs_verification'] = False

    # Submission integrity guards (see LOOSE_LOGIC.md / evaluate_submission_guards).
    # These live outside the rules engine so the engine stays a pure,
    # dataset-validated RA 11861 evaluator.
    try:
        rule_duration_months = int(status_specific_answer)
    except (TypeError, ValueError):
        rule_duration_months = 0
    hard_errors, soft_warnings = evaluate_submission_guards(
        data,
        solo_parent_status=solo_parent_status,
        civil_status=civil_status,
        sex=sex,
        monthly_income=monthly_income,
        youngest_child_age_raw=('' if youngest_child_age is None else str(youngest_child_age)),
        duration_months=rule_duration_months,
        number_of_children=number_of_children,
        age=age,
        family_members=family_members,
        dependent_currently_studying=(data.get('dependent_currently_studying', 'No') == 'Yes'),
    )
    result['guard_errors'] = hard_errors
    result['guard_warnings'] = soft_warnings

    # A hard integrity error overrides any "Eligible" verdict: the submission
    # is inconsistent/impossible, so no benefits are extended and the reasons
    # are surfaced to the applicant.
    if hard_errors:
        result['eligible'] = False
        result['priority_level'] = None
        result['remarks'] = ' '.join(hard_errors)
        engine_result = {**engine_result, 'benefits': {}}

    result['benefits'] = []
    if result['eligible']:
        for b in ENGINE_RULES['benefits']:
            if engine_result['benefits'].get(b['id']):
                result['benefits'].append({
                    'name': b['label'],
                    'description': BENEFIT_DESCRIPTIONS.get(b['id'], f"RA 11861 {b['section']}")
                })

    # Informational notes for the applicant (not blockers). RA 11861's
    # dependent definition requires UNEMPLOYED with no exception, so an
    # employed "Dependent" row does not count at all (see
    # derive_dependent_children). If the applicant is still eligible — i.e.
    # they have at least one OTHER qualifying dependent besides the employed
    # one(s) — explain why the employed member specifically wasn't counted,
    # so the exclusion isn't silent.
    result['dependent_notes'] = []
    if result['eligible'] and dep_info['excluded_employed']:
        names = ', '.join(dep_info['excluded_employed'])
        result['dependent_notes'].append(
            f"{names} was not counted as a dependent because they are employed or have a declared income. Under RA "
            f"11861, a dependent must be unmarried, unemployed, and 22 years old or below "
            f"(or older only if unable to care for themselves due to disability); an "
            f"employed household member does not qualify, even if you support them. This "
            f"did not affect your eligibility because you have at least one other "
            f"qualifying dependent."
        )

    return result


def build_rule_engine_applicant(data, *, occupation, solo_parent_status,
                                 status_specific_answer, number_of_children,
                                 monthly_income, age=None, youngest_child_age=None,
                                 with_pwd=None, dependent_currently_studying=None):
    """Translates this form's raw field names/values into the Applicant
    shape engine.evaluate() expects. Pure field-name translation only — no
    eligibility thresholds or category lists belong here; those live in
    rules.json. number_of_children and youngest_child_age are the values
    DERIVED from the family composition (see derive_dependent_children), not
    the raw posted fields. dependent_currently_studying, if provided, is
    likewise the derived value (self-declared answer AND a qualifying
    dependent's education level is High School or College — see
    SCHOLARSHIP_ELIGIBLE_EDUCATION_LEVELS)."""
    try:
        duration_months = int(status_specific_answer)
    except (TypeError, ValueError):
        # Categories with no duration follow-up question (see
        # SOLO_PARENT_FOLLOW_UP_QUESTIONS) never populate this field, and
        # their rules.json min_duration_months is 0, so 0 is the correct
        # default rather than a sentinel failure value.
        duration_months = 0

    if dependent_currently_studying is None:
        dependent_currently_studying = (data.get('dependent_currently_studying', 'No') == 'Yes')

    return RuleApplicant(
        type_of_solo_parent=TYPE_OF_SOLO_PARENT_MAP.get(solo_parent_status),
        duration_months=duration_months,
        is_pregnant=(solo_parent_status == 'pregnant_woman_unborn_child'),
        number_of_dependents=number_of_children,
        monthly_income=monthly_income,
        employment_status=infer_employment_status(occupation),
        with_pwd=((data.get('with_pwd', 'No') == 'Yes') if with_pwd is None else with_pwd),
        youngest_child_age=youngest_child_age,
        receiving_other_govt_cash_aid=(data.get('receiving_other_govt_cash_aid', 'No') == 'Yes'),
        formal_philhealth_member=(data.get('formal_philhealth_member', 'No') == 'Yes'),
        dependent_currently_studying=dependent_currently_studying,
        age=age,
    )


@app.route('/')
def index():
    """Homepage"""
    return render_template('index.html')


@app.route('/assessment')
def assessment_form():
    """Display application form"""
    return render_template('assessment_form.html')


@app.route('/submit-assessment', methods=['POST'])
def submit_assessment():
    """Process form submission and show results"""
    form_data = request.form.to_dict()

    ml_result = assess_eligibility_ml(form_data)
    model_type = ml_model_metadata.get('model_type', 'ML Model')
    model_version = ml_model_metadata.get('model_version', '1.0')
    decision_source = f"{model_type} v{model_version}"

    family_members = parse_family_members(form_data)
    try:
        birthday = form_data.get('birthday', '')
        age = calculate_age(birthday)
    except:
        age = 0

    monthly_income = parse_money(form_data.get('monthly_income', 0))
    # #12: recomputed from the family rows, not trusted from the posted total.
    total_family_income = sum(float(m.get('income', 0) or 0) for m in family_members)

    # Dependent-children count derived authoritatively from the qualifying
    # family-composition rows (see derive_dependent_children) — for display
    # consistency with assess_eligibility.
    _is_pregnancy = form_data.get('solo_parent_status', '').strip().lower() == 'pregnant_woman_unborn_child'
    number_of_children = derive_dependent_children(
        family_members, is_pregnancy=_is_pregnancy)['count']

    first_name = form_data.get('first_name', '').strip()
    middle_name = form_data.get('middle_name', '').strip()
    surname = form_data.get('surname', '').strip()

    result = {
        'eligible': ml_result['eligible'],
        'confidence': ml_result['confidence'],
        'needs_verification': False,
        'decision_source': decision_source,
        'ml_metadata': {
            'confidence_score': ml_result['confidence'],
            'prob_eligible': ml_result.get('prob_eligible', 0),
            'prob_not_eligible': ml_result.get('prob_not_eligible', 1 - ml_result['confidence']),
            'model_type': model_type,
            'model_version': model_version,
            'training_date': ml_model_metadata.get('training_date', ''),
            'model_status': ml_result.get('model_status', 'unknown')
        },
        'benefits': [],
        'applicant_info': {
            'first_name': first_name,
            'middle_name': middle_name,
            'surname': surname,
            'name': f"{first_name} {middle_name} {surname}".replace('  ', ' ').strip(),
            'birthday': form_data.get('birthday', ''),
            'birth_place': form_data.get('birthplace', '').strip(),
            'age': age,
            'sex': form_data.get('sex', '').title(),
            'address': form_data.get('address', '').strip(),
            'contact': form_data.get('contact_no', '').strip(),
            'landline': form_data.get('landline', '').strip(),
            'civil_status': format_civil_status(form_data.get('civil_status', '')),
            'education': form_data.get('educational_attainment', '').strip(),
            'religion': form_data.get('religion', '').strip(),
            'occupation': form_data.get('occupation', '').strip(),
            'employer': form_data.get('employer_name', '').strip(),
            'company_address': form_data.get('company_address', '').strip(),
            'office_contact': form_data.get('office_contact', '').strip(),
            'monthly_income': monthly_income,
            'total_family_income': total_family_income,
            'solo_parent_status': get_solo_parent_status_label(form_data.get('solo_parent_status', '')),
            'solo_parent_reason': form_data.get('solo_parent_reason', '').strip(),
            'status_specific_question': SOLO_PARENT_FOLLOW_UP_QUESTIONS.get(form_data.get('solo_parent_status', '').strip().lower(), ''),
            'status_specific_answer': form_data.get('category_duration_answer', '').strip(),
            'number_of_children': number_of_children
        },
        'family_members': family_members,
        'required_documents': get_required_documents_for_status(form_data.get('solo_parent_status', '').strip().lower())
    }

    # Run the rule-based assessment
    try:
        rule_result = assess_eligibility(form_data)
    except Exception as e:
        print(f"Rule-based override check failed: {e}")
        rule_result = {'eligible': False, 'needs_verification': False, 'benefits': []}

    is_eligible = rule_result['eligible']
    needs_verification = rule_result.get('needs_verification', False)
    override = False
    rule_conflict = False
    result['remarks'] = rule_result.get('remarks', '')
    result['priority_level'] = rule_result.get('priority_level')
    result['guard_errors'] = rule_result.get('guard_errors', [])
    result['guard_warnings'] = rule_result.get('guard_warnings', [])
    result['dependent_notes'] = rule_result.get('dependent_notes', [])

    # Soft integrity warnings (roster mismatch, etc.) don't block eligibility
    # but flag an eligible case for MSWDO/DSWD verification.
    if result['guard_warnings'] and is_eligible:
        needs_verification = True

    if ml_result.get('model_status') == 'success':
        if ml_result['eligible'] != rule_result['eligible']:
            rule_conflict = True
            if is_eligible:
                needs_verification = True

    # Independent income-outlier safety net (see INCOME_OUTLIER_THRESHOLD
    # above). This check runs regardless of whether the ML model and the
    # rule-based logic agree — agreement between them doesn't mean much if
    # neither was ever validated on income values this extreme.
    #
    # The flag is only allowed to force "needs verification" when the
    # applicant is otherwise ELIGIBLE. Eligibility itself isn't income-gated
    # (income only gates which benefits apply), so a categorically
    # not-eligible applicant is definitively not eligible no matter how
    # large their reported income — turning that into "needs verification"
    # would just hide the real rejection reason behind an income caveat.
    income_outlier = max(monthly_income, total_family_income) > INCOME_OUTLIER_THRESHOLD
    if income_outlier and is_eligible:
        needs_verification = True

    decision_source = "Rule-Based Engine"
    if rule_conflict:
        decision_source += " (with ML Conflict Verification)"
    if income_outlier:
        decision_source += " (Income Outside Training Range — Flagged for Review)"

    result['eligible'] = is_eligible
    result['decision_source'] = decision_source
    result['needs_verification'] = needs_verification

    # Align the displayed confidence with the FINAL (rule-based) decision,
    # not the ML model's raw predicted-class confidence. Without this, a
    # rule/ML disagreement (rule_conflict) could show e.g. a green
    # "ELIGIBLE" banner next to the model's confidence in "Not Eligible".
    prob_eligible = ml_result.get('prob_eligible', 0.0)
    prob_not_eligible = ml_result.get('prob_not_eligible', 1.0 - prob_eligible)
    decision_confidence = prob_eligible if is_eligible else prob_not_eligible
    result['confidence'] = decision_confidence
    result['ml_metadata']['confidence_score'] = decision_confidence

    result['ml_metadata']['override'] = override
    result['ml_metadata']['rule_conflict'] = rule_conflict
    result['ml_metadata']['income_outlier_flagged'] = income_outlier
    result['ml_metadata']['income_outlier_threshold'] = INCOME_OUTLIER_THRESHOLD

    # assess_eligibility() already computed the benefit list via
    # engine.evaluate() / rules.json — the single source of truth shared
    # with client/. Nothing gets recomputed here.
    result['benefits'] = rule_result.get('benefits', [])

    # The PDF-download form embeds this exact JSON string plus its HMAC
    # signature; /download-pdf refuses any payload whose signature doesn't
    # verify, so the client can't alter the assessment between the result
    # page and the PDF.
    result_json = json.dumps(result, sort_keys=True, separators=(',', ':'))
    result_sig = _sign_result_payload(result_json)

    return render_template('result.html', result=result,
                           result_json=result_json, result_sig=result_sig)


def generate_pdf(result):
    """Generate PDF in Solo Parent Application Form style.

    The form is laid out in an 11x17 "design" coordinate space (DESIGN_PAGE)
    and rendered onto Legal-size paper (8.5x14) by applying a single uniform
    fit-to-width scale. This keeps every existing coordinate intact while the
    whole document shrinks proportionally to fit Legal without distortion.
    Scaling down to ~77% also gains vertical room, so the content fits on one
    Legal page with margin to spare.
    """
    buffer = BytesIO()

    # Output paper: Philippine "Legal"/long bond, 8.5in x 14in.
    LEGAL_PAPER = (8.5 * inch, 14 * inch)
    # Original design canvas the coordinates below were built for (11x17).
    DESIGN_PAGE = (11 * inch, 17 * inch)
    # Uniform scale that fits the design width into the Legal width.
    FIT_SCALE = LEGAL_PAPER[0] / DESIGN_PAGE[0]

    c = canvas.Canvas(buffer, pagesize=LEGAL_PAPER)
    c.scale(FIT_SCALE, FIT_SCALE)
    # All drawing below uses design (11x17) coordinates; the scale maps them
    # onto the Legal sheet.
    width, height = DESIGN_PAGE

    def new_page():
        """Start a new page and re-apply the fit scale (showPage resets the
        transform)."""
        c.showPage()
        c.scale(FIT_SCALE, FIT_SCALE)

    app_info = result.get('applicant_info', {})
    family_members = result.get('family_members', [])

    # ---------- Helper functions ----------
    def safe(value):
        if value is None:
            return ""
        return str(value)

    def money(value):
        try:
            return f"PHP {float(value):,.0f}"
        except:
            return ""

    def line(x1, y, x2):
        c.line(x1, y, x2, y)

    def label_value(label, value, x, y, line_start, line_end, font_size=9):
        c.setFont("Times-Roman", font_size)
        c.drawString(x, y, label)
        line(line_start, y - 2, line_end)
        value_text = safe(value)[:60]
        c.drawCentredString((line_start + line_end) / 2, y + 1, value_text)

    def checkbox(x, y, text, checked=False):
        c.rect(x, y - 2, 8, 8)

        if checked:
            c.setFont("Helvetica-Bold", 8)
            c.drawString(x + 1.5, y - 1.5, "X")

        c.setFont("Times-Roman", 9)
        c.drawString(x + 12, y - 1, text)

    def draw_text_in_cell(text, x, y, w, h, font_size=7, center=False):
        text = safe(text)
        c.setFont("Times-Roman", font_size)

        max_chars = max(8, int(w / 4))
        lines = textwrap.wrap(text, width=max_chars)

        start_y = y + h - 11

        for line_text in lines[:2]:
            if center:
                text_width = stringWidth(line_text, "Times-Roman", font_size)
                c.drawString(x + (w - text_width) / 2, start_y, line_text)
            else:
                c.drawString(x + 3, start_y, line_text)

            start_y -= 8

    def draw_wrapped_text(text, x, y, max_chars=110, line_height=11, font_name="Times-Roman", font_size=8):
        c.setFont(font_name, font_size)
        lines = textwrap.wrap(safe(text), width=max_chars)

        for text_line in lines:
            c.drawString(x, y, text_line)
            y -= line_height

        return y

    def status_text():
        if result.get('eligible'):
            confidence = result.get('confidence', 0)
            if result.get('needs_verification'):
                return f"ELIGIBLE (Flagged for MSWDO/DSWD Verification) ({confidence:.1%} confidence)"
            return f"ELIGIBLE ({confidence:.1%} confidence)"
        elif result.get('needs_verification'):
            return "NEEDS VERIFICATION"
        else:
            confidence = result.get('confidence', 0)
            return f"NOT ELIGIBLE ({confidence:.1%} confidence)"

    # ---------- Page settings ----------
    margin_left = 25
    margin_right = 25
    top = height - 35

    c.setStrokeColor(colors.black)
    c.setFillColor(colors.black)

    # ---------- Header ----------
    c.setFont("Times-Roman", 9)
    c.drawCentredString(width / 2, top, "Republic of the Philippines")
    c.drawCentredString(width / 2, top - 12, "Province of Laguna")

    c.setFont("Times-Italic", 8.5)
    c.drawCentredString(
        width / 2,
        top - 27,
        "For Solo Parent Eligibility Assessment under Republic Act No. 11861"
    )

    c.setFont("Times-Bold", 13)
    c.drawCentredString(width / 2, top - 50, "SOLO PARENT APPLICATION FORM")

    c.setFont("Times-Roman", 10)
    c.drawString(width / 2 - 100, top - 67, "Application Number:")
    line(width / 2, top - 68, width / 2 + 100)

    # Picture box
    pic_x = width - 115
    pic_y = top - 82
    c.rect(pic_x, pic_y, 70, 82)

    c.setFont("Times-Roman", 9)
    c.drawCentredString(pic_x + 35, pic_y + 58, "1x1 id pic")
    c.drawCentredString(pic_x + 35, pic_y + 38, "Solo Parent")
    c.drawCentredString(pic_x + 35, pic_y + 22, "Applicant")

    # ---------- I. Identifying Information ----------
    y = top - 112

    c.setFont("Times-Bold", 10)
    c.drawString(margin_left, y, "I.")
    c.drawString(margin_left + 35, y, "IDENTIFYING INFORMATION")

    y -= 24

    # Name line
    c.setFont("Times-Roman", 9)
    c.drawString(margin_left + 45, y, "Name")
    c.drawString(margin_left + 105, y, ":")
    line(margin_left + 150, y - 2, width - 150)

    first_start, first_end = margin_left + 150, margin_left + 275
    middle_start, middle_end = margin_left + 275, margin_left + 400
    sur_start, sur_end = margin_left + 400, width - 150
    c.drawCentredString((first_start + first_end) / 2, y + 1, safe(app_info.get('first_name', ''))[:16])
    c.drawCentredString((middle_start + middle_end) / 2, y + 1, safe(app_info.get('middle_name', ''))[:16])
    c.drawCentredString((sur_start + sur_end) / 2, y + 1, safe(app_info.get('surname', ''))[:16])

    c.setFont("Times-Roman", 8)
    c.drawString(margin_left + 185, y - 12, "(First Name)")
    c.drawString(margin_left + 300, y - 12, "(Middle Name)")
    c.drawString(margin_left + 425, y - 12, "(Surname)")

    y -= 28
    label_value("Birthday", safe(app_info.get('birthday', '')), margin_left + 45, y, margin_left + 150, margin_left + 250)
    label_value("Age", safe(app_info.get('age', '')), margin_left + 285, y, margin_left + 325, margin_left + 380)
    label_value("Sex", safe(app_info.get('sex', '')), margin_left + 420, y, margin_left + 455, width - 150)

    y -= 16
    label_value("Birth Place", safe(app_info.get('birth_place', '')), margin_left + 45, y, margin_left + 150, width - 150)

    y -= 16
    label_value("Address", safe(app_info.get('address', '')), margin_left + 45, y, margin_left + 150, width - 150)

    y -= 16
    c.setFont("Times-Roman", 9)
    c.drawString(margin_left + 45, y, "Civil Status")
    c.drawString(margin_left + 105, y, ":")

    civil = safe(app_info.get('civil_status', '')).lower()

    checkbox(margin_left + 150, y, "Single", "single" in civil)
    checkbox(margin_left + 210, y, "Married", "married" in civil)
    checkbox(margin_left + 285, y, "Separated", "separated" in civil)
    checkbox(margin_left + 370, y, "Divorced", "divorce" in civil or "divorced" in civil)
    checkbox(margin_left + 455, y, "Annulled", "annulled" in civil)
    checkbox(margin_left + 530, y, "Widow", "widow" in civil or "widowed" in civil)

    y -= 16
    label_value("Educ. Attainment", safe(app_info.get('education', '')), margin_left + 45, y, margin_left + 150, margin_left + 280)
    label_value("Religion", safe(app_info.get('religion', '')), margin_left + 300, y, margin_left + 360, margin_left + 400)
    label_value("Monthly Income", money(app_info.get('monthly_income', 0)), margin_left + 410, y, margin_left + 490, width - 35)

    y -= 16
    label_value("Occupation", safe(app_info.get('occupation', '')), margin_left + 45, y, margin_left + 150, width - 150)

    y -= 16
    label_value("Name of Employer", safe(app_info.get('employer', '')), margin_left + 45, y, margin_left + 150, width - 150)

    y -= 16
    label_value("Company Address", safe(app_info.get('company_address', '')), margin_left + 45, y, margin_left + 150, width - 150)

    y -= 16
    label_value("Office Contact No.", safe(app_info.get('office_contact', '')), margin_left + 45, y, margin_left + 150, margin_left + 315)
    label_value("Contact No.", safe(app_info.get('contact', '')), margin_left + 330, y, margin_left + 400, width - 150)

    y -= 16
    label_value("Landline No.", safe(app_info.get('landline', '')), margin_left + 330, y, margin_left + 400, width - 150)

    # ---------- II. Family Composition ----------
    y -= 26

    c.setFont("Times-Bold", 10)
    c.drawString(margin_left, y, "II.")
    c.drawString(margin_left + 35, y, "FAMILY COMPOSITION")

    table_x = margin_left
    table_w = width - (margin_left + margin_right)

    header_h = 32
    row_h = 22
    rows = max(1, len(family_members)) # Dynamically count rows, minimum 1
    
    table_height = header_h + (rows * row_h)
    table_y = y - table_height - 8

    col_widths = [
        130,
        75,
        70,
        32,
        65,
        75,
        table_w - (130 + 75 + 70 + 32 + 65 + 75)
    ]

    headers = [
        "NAMES",
        "RELATIONSHIP",
        "BIRTHDAY",
        "AGE",
        "STATUS",
        "EDUC.\nATTNMENT",
        "OCCUPATION/\nMONTHLY\nINCOME"
    ]

    # Outer table rectangle
    c.rect(table_x, table_y, table_w, table_height)

    # Header separator line
    header_bottom_y = table_y + (rows * row_h)
    c.line(table_x, header_bottom_y, table_x + table_w, header_bottom_y)

    # Internal horizontal row lines (only draw between data rows)
    for i in range(1, rows):
        y_line = table_y + (i * row_h)
        c.line(table_x, y_line, table_x + table_w, y_line)

    # Column vertical lines
    current_x = table_x
    for w in col_widths[:-1]:
        current_x += w
        c.line(current_x, table_y, current_x, table_y + table_height)

    # Headers Text
    current_x = table_x
    for i, header in enumerate(headers):
        c.setFont("Times-Bold", 7.5)

        header_lines = header.split("\n")
        line_y = header_bottom_y + 20

        for header_line in header_lines:
            text_width = stringWidth(header_line, "Times-Bold", 7.5)
            c.drawString(current_x + (col_widths[i] - text_width) / 2, line_y, header_line)
            line_y -= 8

        current_x += col_widths[i]

    # Family member rows Text
    for row_index in range(rows):
        member = family_members[row_index] if row_index < len(family_members) else {}
        
        row_bottom = table_y + ((rows - 1 - row_index) * row_h)

        member_occupation = member.get('occupation', '')
        member_income = member.get('income', 0) or 0
        occupation_income = f"{member_occupation} / {money(member_income)}" if member_income else member_occupation

        values = [
            member.get('name', ''),
            member.get('relationship', ''),
            member.get('birthday', ''),
            member.get('age', ''),
            member.get('civil_status', ''),
            member.get('education', ''),
            occupation_income
        ]

        current_x = table_x

        for col_index, value in enumerate(values):
            draw_text_in_cell(
                value,
                current_x,
                row_bottom,
                col_widths[col_index],
                row_h,
                font_size=7,
                center=True
            )
            current_x += col_widths[col_index]

    # Total family income row
    income_y_position = table_y - 15

    c.setFont("Times-Bold", 8)
    c.drawRightString(table_x + table_w - 80, income_y_position, "TOTAL FAMILY INCOME:")

    c.setFont("Times-Roman", 8)
    c.drawString(table_x + table_w - 75, income_y_position, money(app_info.get('total_family_income', 0)))

    c.setFont("Times-Italic", 8)
    c.drawString(
        table_x, 
        income_y_position,
        "* Please include other members of the household aside from family members"
    )

    # ---------- III. Circumstances ----------
    y = table_y - 45 

    # Page Break Check
    if y < 150:
        new_page()
        y = height - 50

    c.setFont("Times-Bold", 10)
    c.drawString(margin_left, y, "III.")
    c.drawString(margin_left + 35, y, "CIRCUMSTANCES / REASONS OF BEING A SOLO PARENT")

    y -= 24

    reason = safe(app_info.get('solo_parent_reason', ''))
    reason_lines = textwrap.wrap(reason, width=110)

    for i in range(4):
        line_y = y - (i * 18)
        line(margin_left + 10, line_y - 2, width - 35)

        if i < len(reason_lines):
            c.setFont("Times-Roman", 9)
            c.drawString(margin_left + 15, line_y + 1, reason_lines[i])

    # ---------- Consent paragraph ----------
    # Page Break Check
    if y < 200:
        new_page()
        y = height - 40

    consent_y = y - 88

    consent_text = (
        "By signing this document, I/we hereby grant my/our free, voluntary and unconditional consent "
        "to the collection and processing of all Personal Data relating to me/us disclosed/transmitted "
        "by me/us in person or by my/our authorized representative to the information database system "
        "of the Municipal Social Welfare and Development Office and of the Municipal Government of "
        "Santa Cruz, Laguna by its authorized officials and employees, by whatever means in accordance "
        "with Republic Act 10173, otherwise known as the Data Privacy Act of 2012, including its "
        "Implementing Rules and Regulations as well as issuances by the National Privacy Commission."
    )

    c.setFont("Times-Italic", 7.5)
    wrapped_consent = textwrap.wrap(consent_text, width=145)

    for line_text in wrapped_consent:
        c.drawString(margin_left + 35, consent_y, line_text)
        consent_y -= 8

    # ---------- Signature and thumb mark ----------
    signature_y = consent_y - 30

    line(width / 2 - 90, signature_y, width / 2 + 20)
    c.setFont("Times-Roman", 8)
    c.drawCentredString(width / 2 - 35, signature_y - 10, "Signature over Printed Name")

    line(width / 2 + 45, signature_y, width / 2 + 120)
    c.drawCentredString(width / 2 + 82, signature_y - 10, "Date")

    thumb_x = width - 115
    thumb_y = signature_y - 25

    c.rect(thumb_x, thumb_y, 70, 60)
    c.drawCentredString(thumb_x + 35, thumb_y - 12, "Right Thumb Mark")

    # ---------- Small system assessment note ----------
    note_x = width - 345
    note_y = signature_y - 55

    c.setFont("Times-Bold", 8)
    c.drawString(note_x, note_y, "System Assessment Result:")

    c.setFont("Times-Roman", 8)
    c.drawString(note_x + 118, note_y, status_text())

    c.setFont("Times-Italic", 6.5)
    c.drawString(note_x, note_y - 13, "Final approval is still subject to MSWDO/DSWD verification.")

    # ---------- V. Eligibility Result, Requirements & Recommendations ----------
    y = signature_y - 95

    # Page Break Check for Section V
    if y < 320:
        new_page()
        y = height - 50

    c.setFont("Times-Bold", 11)
    c.drawString(margin_left, y, "V. REQUIREMENTS & RECOMMENDATIONS")

    y -= 22

    if result.get('eligible'):
        intro_text = (
            "You are ELIGIBLE for Solo Parent Benefits. Please prepare the following documents "
            "and visit your local DSWD/MSWDO office:"
        )
    elif result.get('needs_verification'):
        intro_text = (
            "Your case REQUIRES VERIFICATION. Please prepare the following documents and visit "
            "your local DSWD/MSWDO office for further assessment:"
        )
    else:
        intro_text = (
            "You are currently NOT ELIGIBLE based on the system assessment. However, you may still "
            "visit your local DSWD/MSWDO office for proper verification, guidance, or other available assistance programs."
        )

    c.setFont("Times-Bold", 8.5)
    if result.get('eligible'):
        c.drawString(margin_left, y, "You are ELIGIBLE for Solo Parent Benefits.")
        c.setFont("Times-Roman", 8.5)
        c.drawString(margin_left + 175, y, "Please prepare the following documents and visit your local DSWD/MSWDO office:")
        y -= 15
    else:
        y = draw_wrapped_text(intro_text, margin_left, y, max_chars=125, line_height=11, font_size=8.5)
        y -= 5

    c.setFont("Times-Bold", 8.5)
    c.drawString(margin_left, y, "Required Documents:")

    y -= 13

    required_documents = result.get('required_documents', [])

    c.setFont("Times-Roman", 8.3)
    for item in required_documents:
        c.drawString(margin_left + 12, y, "- " + item)
        y -= 11

    y -= 8

    c.setFont("Times-Bold", 8.5)
    c.drawString(margin_left, y, "Next Steps:")

    y -= 13

    next_steps = [
        "Prepare and gather all required documents.",
        "Visit your local DSWD/MSWDO office with this assessment report.",
        "File your formal Solo Parent registration.",
        "Wait for document verification and processing.",
        "Claim your Solo Parent ID once approved by the authorized office."
    ]

    c.setFont("Times-Roman", 8.3)
    for i, step in enumerate(next_steps, 1):
        c.drawString(margin_left + 12, y, f"{i}. {step}")
        y -= 11

    if result.get('eligible'):
        y -= 8

        c.setFont("Times-Bold", 8.5)
        c.drawString(margin_left, y, "Eligible Benefits Include:")

        y -= 13

        # result['benefits'] already holds exactly the benefits
        # engine.evaluate() granted (see assess_eligibility()) — listed
        # here directly so the PDF can never drift from the web page.
        for benefit in result.get('benefits', []):
            c.circle(margin_left + 15, y + 3, 3.5, stroke=1, fill=0)
            text = f"{benefit.get('name', '')}: {benefit.get('description', '')}"
            y = draw_wrapped_text(text, margin_left + 25, y, max_chars=115, line_height=10, font_size=8.3)
            y -= 1

    # ---------- Footer ----------
    y -= 20

    c.line(margin_left, y, width - margin_right, y)

    y -= 14

    c.setFont("Times-Bold", 7.5)
    c.drawString(margin_left, y, "Assessment Prepared By:")

    c.setFont("Times-Roman", 7.5)
    c.drawString(margin_left + 100, y, "Solo Parent Decision Support System")

    y -= 11

    c.setFont("Times-Bold", 7.5)
    c.drawString(margin_left, y, "Assessment Date:")

    c.setFont("Times-Roman", 7.5)
    c.drawString(margin_left + 75, y, datetime.now().strftime('%B %d, %Y at %I:%M %p'))

    y -= 12

    important_note = (
        "Important Note: This is an automated assessment based on the encoded information and RA 11861-related criteria. "
        "Final determination of eligibility is made by the authorized MSWDO/DSWD personnel upon submission of complete documents "
        "and verification interview."
    )

    c.setFont("Times-Italic", 6.8)
    draw_wrapped_text(important_note, margin_left, y, max_chars=180, line_height=9, font_size=6.8)

    c.showPage()
    c.save()

    buffer.seek(0)
    return buffer


@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    """Generate and download PDF report.

    Only accepts the result payload submit_assessment() signed and embedded
    in result.html — without the HMAC check, anyone could POST arbitrary
    JSON here and mint an official-looking "ELIGIBLE" assessment PDF with
    any name, confidence, and benefit list."""
    result_json = request.form.get('result_data', '')
    signature = request.form.get('result_sig', '')
    if not result_json or not hmac.compare_digest(
            _sign_result_payload(result_json), signature):
        return ("Invalid or missing assessment signature. Please download the PDF "
                "from the assessment result page.", 403)
    result = json.loads(result_json)

    pdf_buffer = generate_pdf(result)

    filename = f"Solo_Parent_Assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename
    )


@app.route('/faq')
def faq():
    """Frequently Asked Questions page"""
    return render_template('faq.html')


@app.route('/contact')
def contact():
    """Contact MSWDO page"""
    return render_template('contact.html')


@app.route('/privacy')
def privacy_policy():
    """Privacy Policy page"""
    return render_template('privacy_policy.html')


@app.route('/report')
def report():
    """Report an Inquiry page"""
    return render_template('report.html')


@app.route('/ra11861')
def ra11861():
    """RA 11861 Compliance page"""
    return render_template('ra11861.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))