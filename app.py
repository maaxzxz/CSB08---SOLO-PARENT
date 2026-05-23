from flask import Flask, render_template, request, send_file
from datetime import datetime
from reportlab.lib.pagesizes import A4, legal
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from io import BytesIO
import json
import os
import textwrap
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = 'solo-parent-dss-key'

# Load ML Model and Encoders at startup
try:
    ml_model = joblib.load('model/solo_parent_model.pkl')
    ml_encoders = joblib.load('model/encoders.pkl')
    ml_feature_columns = joblib.load('model/feature_columns.pkl')
    ml_model_loaded = True
except Exception as e:
    print(f"Warning: Could not load ML model: {e}. Using rule-based assessment only.")
    ml_model_loaded = False
    ml_model = None
    ml_encoders = None
    ml_feature_columns = None

# RA 11861 Defined Benefits
BENEFITS = {
    'monthly_subsidy': {
        'name': 'Monthly Subsidy',
        'description': 'PHP 1,500 cash assistance per child, subject to official guidelines'
    },
    'vat_exemption': {
        'name': 'VAT Exemption & Discounts',
        'description': 'VAT discount and VAT exemption on qualified goods and services'
    },
    'educational_support': {
        'name': 'Educational Support',
        'description': 'Assistance program for dependent children'
    },
    'priority_services': {
        'name': 'Priority Services',
        'description': 'Priority access to government services'
    }
}


def calculate_age(birthdate_str):
    """Calculate age from birthdate string YYYY-MM-DD format"""
    try:
        birthdate = datetime.strptime(birthdate_str, '%Y-%m-%d')
        today = datetime.today()
        age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
        return age
    except:
        return 0


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
            family_members.append({
                'name': name,
                'relationship': form_data.get(f'family_relationship_{index}', ''),
                'birthday': form_data.get(f'family_birthday_{index}', ''),
                'age': form_data.get(f'family_age_{index}', '0'),
                'civil_status': form_data.get(f'family_civil_status_{index}', ''),
                'education': form_data.get(f'family_educ_{index}', ''),
                'occupation': form_data.get(f'family_occupation_{index}', '')
            })

    return family_members


def infer_employment_status(occupation_text):
    """Infer employment status from occupation text field"""
    if not occupation_text:
        return 'Unemployed'

    occ = occupation_text.lower().strip()

    if occ == 'unemployed' or occ == 'none' or len(occ) == 0:
        return 'Unemployed'
    elif any(word in occ for word in ['own', 'self', 'business', 'freelance', 'vendor']):
        return 'Self-Employed'
    elif any(word in occ for word in ['part-time', 'pt', 'part time']):
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

    solo_parent_type = form_data.get('solo_parent_status', '').strip().lower()
    solo_parent_map = {
        'single': 'Unmarried',
        'widowed': 'Widowed',
        'separated': 'Separated',
        'abandoned': 'Abandoned',
        'ofw': 'OFW Spouse'
    }
    mapped_type = solo_parent_map.get(solo_parent_type, 'Unmarried')

    features = {
        'Age': age,
        'Educational_Attainment': form_data.get('educational_attainment', '').strip() or 'Elementary',
        'Employment_Status': infer_employment_status(form_data.get('occupation', '')),
        'Monthly_Income': float(form_data.get('monthly_income', 0)),
        'Number_of_Dependents': int(form_data.get('number_of_dependent_children', 0)),
        'With_Minor': has_minor_dependents(family_members),
        'With_PWD': form_data.get('with_pwd', 'No'),
        'Type_of_Solo_Parent': mapped_type
    }

    return features


def assess_eligibility_ml(form_data):
    """ML-based eligibility assessment using pre-trained Random Forest model"""
    if not ml_model_loaded:
        return {
            'eligible': False,
            'confidence': 0.0,
            'model_status': 'unavailable'
        }

    try:
        features = extract_ml_features(form_data)

        feature_df = pd.DataFrame([features])

        for col in ml_encoders:
            if col in feature_df.columns:
                try:
                    feature_df[col] = ml_encoders[col].transform(feature_df[col].astype(str))
                except:
                    feature_df[col] = 0

        feature_df = feature_df[ml_feature_columns]

        prediction = ml_model.predict(feature_df)[0]
        probabilities = ml_model.predict_proba(feature_df)[0]

        if len(ml_model.classes_) == 1:
            confident_class = ml_model.classes_[0]
            confident_prob = 1.0
            other_prob = 0.0
        else:
            confident_prob = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
            other_prob = 1.0 - confident_prob

        result = {
            'eligible': bool(prediction == 1),
            'confidence': confident_prob if prediction == 1 else other_prob,
            'prob_eligible': confident_prob if prediction == 1 else other_prob,
            'prob_not_eligible': other_prob if prediction == 1 else confident_prob,
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

    try:
        monthly_income = float(data.get('monthly_income', 0))
    except:
        monthly_income = 0

    try:
        total_family_income = float(data.get('total_family_income', 0))
    except:
        total_family_income = 0

    solo_parent_status = data.get('solo_parent_status', '')

    try:
        number_of_children = int(data.get('number_of_dependent_children', 0))
    except:
        number_of_children = 0

    solo_parent_reason = data.get('solo_parent_reason', '').strip()

    # Calculate applicant age
    age = calculate_age(birthday)

    # Parse family members
    family_members = parse_family_members(data)

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

        'civil_status': civil_status,
        'education': data.get('educational_attainment', '').strip(),
        'religion': data.get('religion', '').strip(),

        'occupation': occupation,
        'employer': data.get('employer_name', '').strip(),
        'company_address': data.get('company_address', '').strip(),
        'office_contact': data.get('office_contact', '').strip(),

        'monthly_income': monthly_income,
        'total_family_income': total_family_income,

        'solo_parent_status': solo_parent_status.title(),
        'solo_parent_reason': solo_parent_reason,
        'number_of_children': number_of_children
    }

    result['family_members'] = family_members

    # RA 11861 Eligibility Criteria
    has_valid_status = solo_parent_status in [
        'widowed',
        'abandoned',
        'separated',
        'single'
    ]

    # Income threshold
    income_threshold = 30000
    adjusted_threshold = income_threshold + (number_of_children * 5000)

    # Automatic eligible criteria
    if (
        has_valid_status and
        number_of_children > 0 and
        total_family_income < adjusted_threshold and
        age >= 18
    ):
        result['eligible'] = True
        result['benefits'] = [
            BENEFITS['monthly_subsidy'],
            BENEFITS['vat_exemption'],
            BENEFITS['educational_support'],
            BENEFITS['priority_services']
        ]

    # Needs verification criteria
    elif (
        has_valid_status and
        number_of_children > 0 and
        total_family_income < (adjusted_threshold + 10000) and
        age >= 18
    ):
        result['needs_verification'] = True
        result['eligible'] = False

    # Not eligible
    else:
        result['eligible'] = False
        result['needs_verification'] = False

    return result


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

    family_members = parse_family_members(form_data)
    try:
        birthday = form_data.get('birthday', '')
        age = calculate_age(birthday)
    except:
        age = 0

    try:
        monthly_income = float(form_data.get('monthly_income', 0))
    except:
        monthly_income = 0

    try:
        total_family_income = float(form_data.get('total_family_income', 0))
    except:
        total_family_income = 0

    try:
        number_of_children = int(form_data.get('number_of_dependent_children', 0))
    except:
        number_of_children = 0

    first_name = form_data.get('first_name', '').strip()
    middle_name = form_data.get('middle_name', '').strip()
    surname = form_data.get('surname', '').strip()

    result = {
        'eligible': ml_result['eligible'],
        'confidence': ml_result['confidence'],
        'needs_verification': False,
        'decision_source': 'ML Model v1.0',
        'ml_metadata': {
            'confidence_score': ml_result['confidence'],
            'prob_eligible': ml_result.get('prob_eligible', 0),
            'prob_not_eligible': ml_result.get('prob_not_eligible', 1 - ml_result['confidence']),
            'model_version': '1.0',
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
            'civil_status': form_data.get('civil_status', ''),
            'education': form_data.get('educational_attainment', '').strip(),
            'religion': form_data.get('religion', '').strip(),
            'occupation': form_data.get('occupation', '').strip(),
            'employer': form_data.get('employer_name', '').strip(),
            'company_address': form_data.get('company_address', '').strip(),
            'office_contact': form_data.get('office_contact', '').strip(),
            'monthly_income': monthly_income,
            'total_family_income': total_family_income,
            'solo_parent_status': form_data.get('solo_parent_status', '').title(),
            'solo_parent_reason': form_data.get('solo_parent_reason', '').strip(),
            'number_of_children': number_of_children
        },
        'family_members': family_members
    }

    if ml_result['eligible']:
        result['benefits'] = [
            {'name': 'Monthly Subsidy', 'description': 'PHP 1,500 cash assistance per child, subject to official guidelines'},
            {'name': 'VAT Exemption & Discounts', 'description': 'VAT discount and VAT exemption on qualified goods and services'},
            {'name': 'Educational Support', 'description': 'Assistance program for dependent children'},
            {'name': 'Priority Services', 'description': 'Priority access to government services'}
        ]

    return render_template('result.html', result=result)


def generate_pdf(result):
    """Generate PDF in Solo Parent Application Form style"""
    buffer = BytesIO()

    # Tabloid Paper: 11 inches wide x 17 inches long
    TABLOID_PAPER = (11 * inch, 17 * inch)

    c = canvas.Canvas(buffer, pagesize=TABLOID_PAPER)
    width, height = TABLOID_PAPER

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
        c.drawString(line_start + 3, y + 1, safe(value)[:60])

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

    c.drawString(margin_left + 165, y + 1, safe(app_info.get('first_name', ''))[:16])
    c.drawString(margin_left + 280, y + 1, safe(app_info.get('middle_name', ''))[:16])
    c.drawString(margin_left + 405, y + 1, safe(app_info.get('surname', ''))[:16])

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
    checkbox(margin_left + 215, y, "Married/Separated", "married" in civil or "separated" in civil)
    checkbox(margin_left + 345, y, "Annulled", "annulled" in civil)
    checkbox(margin_left + 430, y, "Widow", "widow" in civil or "widowed" in civil)

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

        values = [
            member.get('name', ''),
            member.get('relationship', ''),
            member.get('birthday', ''),
            member.get('age', ''),
            member.get('civil_status', ''),
            member.get('education', ''),
            member.get('occupation', '')
        ]

        current_x = table_x

        for col_index, value in enumerate(values):
            draw_text_in_cell(
                value,
                current_x,
                row_bottom,
                col_widths[col_index],
                row_h,
                font_size=7
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
        c.showPage()
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
        c.showPage()
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

    # ---------- Requirements ----------
    req_y_start = signature_y - 55
    req_y = req_y_start

    c.setFont("Times-Bold", 9)
    c.drawString(margin_left, req_y, "REQUIREMENTS:")

    requirements = [
        "Barangay Certificate",
        "Employment Certificate",
        "Affidavit of being a solo parent",
        "Birth Certificate (Minor Children/PWD)",
        "Death Certificate (If Widow)",
        "2 pcs. 1x1 Picture of the Applicant"
    ]

    c.setFont("Times-Roman", 8)
    req_y -= 13

    for item in requirements:
        c.drawString(margin_left + 18, req_y, "-")
        c.drawString(margin_left + 32, req_y, item)
        req_y -= 11

    # ---------- Small system assessment note ----------
    note_x = width - 345
    note_y = req_y_start

    c.setFont("Times-Bold", 8)
    c.drawString(note_x, note_y, "System Assessment Result:")

    c.setFont("Times-Roman", 8)
    c.drawString(note_x + 118, note_y, status_text())

    c.setFont("Times-Italic", 6.5)
    c.drawString(note_x, note_y - 13, "Final approval is still subject to MSWDO/DSWD verification.")

    # ---------- V. Eligibility Result, Requirements & Recommendations ----------
    y = req_y - 35

    # Page Break Check for Section V
    if y < 320:
        c.showPage()
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

    required_documents = [
        "Barangay Certificate confirming solo parent status",
        "Employment Certificate or Payslip, if employed",
        "Affidavit of Solo Parent Status, notarized",
        "Birth Certificate of all dependent children",
        "Death Certificate if widowed, or Court Orders if separated/annulled",
        "Valid ID and Proof of Residence",
        "Tax Identification Number (TIN), if applicable"
    ]

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

        eligible_benefits = [
            "PHP 1,500/month subsidy per dependent child, subject to official guidelines",
            "VAT exemption on qualified goods and services",
            "Educational support and scholarship assistance",
            "Priority access to government services",
            "Healthcare and wellness programs"
        ]

        c.setFont("Times-Roman", 8.3)
        for item in eligible_benefits:
            # Draw an empty circle
            c.circle(margin_left + 15, y + 3, 3.5, stroke=1, fill=0)
            c.drawString(margin_left + 25, y, item)
            y -= 11

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
    """Generate and download PDF report"""
    result_json = request.form.get('result_data', '{}')
    result = json.loads(result_json)

    pdf_buffer = generate_pdf(result)

    filename = f"Solo_Parent_Assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename
    )


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
