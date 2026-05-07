from flask import Flask, render_template, request, redirect, url_for, send_file
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from io import BytesIO
import json
import os

app = Flask(__name__)
app.secret_key = 'solo-parent-dss-key'

# RA 11861 Defined Benefits
BENEFITS = {
    'monthly_subsidy': {
        'name': 'Monthly Subsidy',
        'description': '₱1,500 cash assistance per child (benefit of 3 children)'
    },
    'vat_exemption': {
        'name': 'VAT Exemption & Discounts',
        'description': 'VAT discount and VAT exemption on goods/services'
    },
    'educational_support': {
        'name': 'Educational Support',
        'description': 'Assistance program for dependent children'
    },
    'priority_services': {
        'name': 'Priority Services',
        'description': 'Exclusive access to government services'
    }
}

def calculate_age(birthdate_str):
    """Calculate age from birthdate string (YYYY-MM-DD format)"""
    try:
        birthdate = datetime.strptime(birthdate_str, '%Y-%m-%d')
        today = datetime.today()
        age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
        return age
    except:
        return 0

def parse_family_members(form_data):
    """Parse family members from form data"""
    family_members = []
    index = 0
    while f'family_name_{index}' in form_data:
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
        index += 1
    return family_members

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
    surname = data.get('surname', '').strip()
    birthday = data.get('birthday', '')
    sex = data.get('sex', '')
    address = data.get('address', '').strip()
    contact_no = data.get('contact_no', '').strip()
    civil_status = data.get('civil_status', '')
    occupation = data.get('occupation', '').strip()
    monthly_income = float(data.get('monthly_income', 0))
    total_family_income = float(data.get('total_family_income', 0))
    solo_parent_status = data.get('solo_parent_status', '')
    number_of_children = int(data.get('number_of_dependent_children', 0))
    solo_parent_reason = data.get('solo_parent_reason', '').strip()

    # Calculate applicant age
    age = calculate_age(birthday)

    # Parse family members
    family_members = parse_family_members(data)

    # Store applicant info
    result['applicant_info'] = {
        'name': f"{first_name} {surname}",
        'age': age,
        'address': address,
        'contact': contact_no,
        'civil_status': civil_status,
        'occupation': occupation,
        'monthly_income': monthly_income,
        'solo_parent_status': solo_parent_status.title()
    }
    result['family_members'] = family_members

    # RA 11861 Eligibility Criteria:
    # 1. Must be a solo parent (widowed, abandoned, separated, single parent with child)
    # 2. Must have dependent children
    # 3. Income threshold (typically below or near poverty line)
    # 4. Must be at least 21 years old (or at least supporting a minor child)

    has_valid_status = solo_parent_status in [
        'widowed',
        'abandoned',
        'separated',
        'single'
    ]

    # Income threshold - baseline is ₱30,000/month
    # Can be adjusted based on family size and region
    income_threshold = 30000
    adjusted_threshold = income_threshold + (number_of_children * 5000)

    # ELIGIBILITY DECISION LOGIC

    # Automatic eligible criteria
    if (has_valid_status and
        number_of_children > 0 and
        total_family_income < adjusted_threshold and
        age >= 18):  # Can be 18+ if supporting minors

        result['eligible'] = True
        # All RA 11861 benefits apply
        result['benefits'] = [
            BENEFITS['monthly_subsidy'],
            BENEFITS['vat_exemption'],
            BENEFITS['educational_support'],
            BENEFITS['priority_services']
        ]

    # Needs verification criteria (borderline cases)
    elif (has_valid_status and
          number_of_children > 0 and
          total_family_income < (adjusted_threshold + 10000) and
          age >= 18):

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
    # Get form data
    form_data = request.form.to_dict()

    # Run assessment
    result = assess_eligibility(form_data)

    # Store result in session for PDF download
    request.session_data = result

    # Render result page
    return render_template('result.html', result=result)


def generate_pdf(result):
    """Generate PDF in questionnaire format"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=0.5*inch, leftMargin=0.5*inch,
                           topMargin=0.5*inch, bottomMargin=0.5*inch)

    # Container for PDF elements
    elements = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#002d5f'),
        spaceAfter=12,
        alignment=1,  # Center
        fontName='Helvetica-Bold'
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=11,
        textColor=colors.HexColor('#002d5f'),
        spaceAfter=10,
        fontName='Helvetica-Bold'
    )

    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=9,
        spaceAfter=4
    )

    app_info = result['applicant_info']

    # Header
    elements.append(Paragraph("SOLO PARENT ASSESSMENT & ELIGIBILITY REPORT", title_style))
    elements.append(Paragraph("Republic of the Philippines - Department of Social Welfare and Development", normal_style))
    elements.append(Paragraph(f"Assessment Date: {datetime.now().strftime('%B %d, %Y')}", normal_style))
    elements.append(Spacer(1, 0.2*inch))

    # SECTION I: IDENTIFYING INFORMATION
    elements.append(Paragraph("I. IDENTIFYING INFORMATION", heading_style))

    id_data = [
        ['Full Name:', f"{app_info['name']}"],
        ['Age / Sex:', f"{app_info['age']} years old"],
        ['Civil Status:', app_info['civil_status'].title()],
        ['Address:', app_info['address']],
        ['Contact No.:', app_info['contact']],
        ['Occupation:', app_info['occupation']],
        ['Monthly Income:', f"PHP {app_info['monthly_income']:,.2f}"],
        ['Solo Parent Status:', app_info['solo_parent_status']],
    ]

    id_table = Table(id_data, colWidths=[2*inch, 4*inch])
    id_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))

    elements.append(id_table)
    elements.append(Spacer(1, 0.15*inch))

    # SECTION II: FAMILY COMPOSITION
    elements.append(Paragraph("II. FAMILY COMPOSITION", heading_style))

    if result['family_members']:
        family_data = [['Name', 'Relationship', 'Age', 'Civil Status', 'Education', 'Occupation/Income']]
        for member in result['family_members']:
            family_data.append([
                member.get('name', ''),
                member.get('relationship', ''),
                str(member.get('age', '')),
                member.get('civil_status', ''),
                member.get('education', ''),
                member.get('occupation', '')
            ])

        family_table = Table(family_data, colWidths=[1.3*inch, 1.1*inch, 0.7*inch, 0.9*inch, 0.9*inch, 1.1*inch])
        family_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#002d5f')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
        ]))

        elements.append(family_table)
    elements.append(Spacer(1, 0.15*inch))

    # SECTION III: ELIGIBILITY ASSESSMENT RESULT
    elements.append(Paragraph("III. ELIGIBILITY ASSESSMENT RESULT", heading_style))

    if result['eligible']:
        status_text = "ELIGIBLE FOR SOLO PARENT BENEFITS"
        status_color = colors.HexColor('#28a745')
    elif result['needs_verification']:
        status_text = "NEEDS VERIFICATION"
        status_color = colors.HexColor('#ffc107')
    else:
        status_text = "NOT CURRENTLY ELIGIBLE"
        status_color = colors.HexColor('#dc3545')

    result_style = ParagraphStyle(
        'ResultStatus',
        parent=styles['Normal'],
        fontSize=11,
        textColor=status_color,
        fontName='Helvetica-Bold',
        spaceAfter=8
    )

    elements.append(Paragraph(f"Status: {status_text}", result_style))

    result_info = f"""
    This assessment is based on the applicant's information against Republic Act 11861 (Solo Parent Welfare Act) criteria.
    The evaluation considers family income, number of dependents, employment status, and solo parent classification.
    """
    elements.append(Paragraph(result_info, normal_style))
    elements.append(Spacer(1, 0.1*inch))

    # SECTION IV: QUALIFIED BENEFITS
    if result['eligible'] and result['benefits']:
        elements.append(Paragraph("IV. QUALIFIED BENEFITS UNDER RA 11861", heading_style))

        benefits_data = []
        for i, benefit in enumerate(result['benefits'], 1):
            benefits_data.append([
                f"{i}.",
                Paragraph(f"<b>{benefit['name']}</b><br/>{benefit['description']}", normal_style)
            ])

        benefits_table = Table(benefits_data, colWidths=[0.4*inch, 5.6*inch])
        benefits_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0f8ff')),
        ]))

        elements.append(benefits_table)
        elements.append(Spacer(1, 0.15*inch))

    # SECTION V: REQUIREMENTS & RECOMMENDATIONS
    elements.append(Paragraph("V. REQUIREMENTS & RECOMMENDATIONS", heading_style))

    if result['eligible']:
        rec_text = """
        <b>You are ELIGIBLE for Solo Parent Benefits.</b> Please prepare the following documents and visit your local DSWD office:
        <br/><br/>
        <b>Required Documents:</b><br/>
        • Barangay Certificate confirming solo parent status<br/>
        • Employment Certificate or Payslip (if employed)<br/>
        • Affidavit of Solo Parent Status (notarized)<br/>
        • Birth Certificate of all dependent children<br/>
        • Death Certificate (if widowed) or Court Orders (if separated/annulled)<br/>
        • Valid ID and Proof of Residence<br/>
        • Tax Identification Number (TIN) if applicable<br/>
        <br/>
        <b>Next Steps:</b><br/>
        1. Prepare and gather all required documents<br/>
        2. Visit your local DSWD office with this assessment report<br/>
        3. File your formal Solo Parent registration<br/>
        4. Wait for processing (typically 30-45 days)<br/>
        5. Claim your Solo Parent ID and start enjoying benefits<br/>
        <br/>
        <b>Eligible Benefits Include:</b><br/>
        • ₱1,500/month subsidy per dependent child<br/>
        • VAT exemption on goods and services<br/>
        • Educational support and scholarships<br/>
        • Priority access to government services<br/>
        • Healthcare and wellness programs<br/>
        """

    elif result['needs_verification']:
        rec_text = """
        <b>Your case REQUIRES VERIFICATION.</b> Additional documents are needed to complete your assessment.
        <br/><br/>
        <b>Documents for Verification:</b><br/>
        • Updated income certificate from employer or barangay<br/>
        • Proof of solo parent status (supporting documents)<br/>
        • Birth Certificates of all dependent children<br/>
        • Updated proof of residence<br/>
        • Any additional documents supporting your circumstances<br/>
        <br/>
        <b>What to Do:</b><br/>
        1. Gather the required verification documents<br/>
        2. Contact your local DSWD office to schedule a verification interview<br/>
        3. Bring this assessment report and all supporting documents<br/>
        4. DSWD staff will conduct verification and provide final assessment<br/>
        5. You will be notified of the results within 15 business days<br/>
        """

    else:
        rec_text = """
        <b>You are currently NOT ELIGIBLE</b> based on the provided information and RA 11861 criteria.
        <br/><br/>
        <b>You may reapply if:</b><br/>
        • Your income decreases significantly<br/>
        • You lose your current employment<br/>
        • You have additional dependent children<br/>
        • Your family circumstances change<br/>
        • You obtain additional supporting documents<br/>
        <br/>
        <b>Recommendations:</b><br/>
        1. Contact your barangay social worker for other assistance programs<br/>
        2. Explore other government support services you may qualify for<br/>
        3. Consider reapplication when circumstances change<br/>
        4. Keep this assessment for your records<br/>
        """

    elements.append(Paragraph(rec_text, normal_style))
    elements.append(Spacer(1, 0.2*inch))

    # Footer
    elements.append(Paragraph("_" * 80, normal_style))
    footer_text = f"""
    <b>Assessment Prepared By:</b> Solo Parent Decision Support System<br/>
    <b>Assessment Date:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>
    <b>Important Note:</b> This is an automated assessment based on RA 11861 criteria. Final determination of eligibility
    is made by DSWD upon submission of complete documents and verification interview.
    This report is valid for 12 months from the date of assessment.
    """
    elements.append(Paragraph(footer_text, ParagraphStyle('FooterStyle', parent=styles['Normal'], fontSize=7)))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer


@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    """Generate and download PDF report"""
    # Get result from request
    result_json = request.form.get('result_data', '{}')
    result = json.loads(result_json)

    # Generate PDF
    pdf_buffer = generate_pdf(result)

    # Return PDF
    filename = f"Solo_Parent_Assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename
    )


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))


