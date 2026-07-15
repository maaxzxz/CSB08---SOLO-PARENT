import sys
import os

from app import app, generate_pdf

def test_pdf_generation():
    print("=" * 80)
    print("TESTING PDF GENERATION FOR ELIGIBLE AND BORDERLINE/HIGH INCOME CASES")
    print("=" * 80)

    # Benefit fixtures below mirror the current rules.json benefit set and
    # RA 11861 Sec. 15/Sec. 9 wording (see BENEFIT_DESCRIPTIONS in app.py).

    # 1. Low income result — qualifies for the full benefit set
    res_low = {
        'eligible': True,
        'confidence': 0.95,
        'needs_verification': False,
        'benefits': [
            {'name': 'Cash Subsidy', 'description': 'PHP 1,000 monthly cash subsidy per solo parent, subject to fund availability.'},
            {'name': 'VAT Discount/Exemption', 'description': '10% discount and VAT exemption on select child-care goods for children aged 0-6.'},
            {'name': 'PhilHealth / NHIP Automatic Coverage', 'description': 'Automatic PhilHealth/NHIP coverage for solo parents not already formally covered.'},
            {'name': 'Livelihood / Employment Priority', 'description': 'Priority access to livelihood, self-employment, and skills training programs.'},
            {'name': 'Housing Priority', 'description': 'Priority allocation in government socialized housing projects.'},
            {'name': 'Educational Support', 'description': 'Scholarship and tuition assistance for a dependent currently studying.'}
        ],
        'applicant_info': {
            'name': 'Carmela Reyes',
            'age': 22,
            'civil_status': 'Single',
            'solo_parent_status': 'Single Parent (Unmarried)',
            'address': 'Labuin',
            'contact': '09123456789',
            'occupation': 'Unemployed',
            'monthly_income': 1508,
            'total_family_income': 1508,
            'number_of_children': 2
        },
        'family_members': []
    }

    # 2. Eligible but flagged for verification (income outlier / ML conflict)
    res_borderline = {
        'eligible': True,
        'confidence': 0.88,
        'needs_verification': True,
        'benefits': [
            {'name': 'PhilHealth / NHIP Automatic Coverage', 'description': 'Automatic PhilHealth/NHIP coverage for solo parents not already formally covered.'},
            {'name': 'Educational Support', 'description': 'Scholarship and tuition assistance for a dependent currently studying.'}
        ],
        'applicant_info': {
            'name': 'Maria Santos',
            'age': 35,
            'civil_status': 'Separated',
            'solo_parent_status': 'Legally Separated',
            'address': 'Calios',
            'contact': '09123456789',
            'occupation': 'Employed',
            'monthly_income': 42000,
            'total_family_income': 42000,
            'number_of_children': 2
        },
        'family_members': []
    }

    # 3. High income result — only the non-income-gated benefits apply
    res_high = {
        'eligible': True,
        'confidence': 0.92,
        'needs_verification': False,
        'benefits': [
            {'name': 'PhilHealth / NHIP Automatic Coverage', 'description': 'Automatic PhilHealth/NHIP coverage for solo parents not already formally covered.'},
            {'name': 'Educational Support', 'description': 'Scholarship and tuition assistance for a dependent currently studying.'}
        ],
        'applicant_info': {
            'name': 'Dominic Villanueva',
            'age': 55,
            'civil_status': 'Separated',
            'solo_parent_status': 'Legally Separated',
            'address': 'Labuin',
            'contact': '09123456789',
            'occupation': 'Self-Employed',
            'monthly_income': 60000,
            'total_family_income': 60000,
            'number_of_children': 2
        },
        'family_members': []
    }

    # Use project directory path for writing test PDFs
    project_dir = os.path.dirname(os.path.abspath(__file__))

    for name, res in [("Low Income", res_low), ("Borderline", res_borderline), ("High Income", res_high)]:
        try:
            print(f"Generating PDF for: {name}")
            pdf_buffer = generate_pdf(res)
            filename = f"test_{name.replace(' ', '_').lower()}_assessment.pdf"
            filepath = os.path.join(project_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(pdf_buffer.getvalue())
            print(f"  [SUCCESS] Written to {filepath}")
        except Exception as e:
            print(f"  [FAILED] PDF generation failed for {name}: {e}")

if __name__ == '__main__':
    test_pdf_generation()
