import sys
import os

from app import app, generate_pdf

def test_pdf_generation():
    print("=" * 80)
    print("TESTING PDF GENERATION FOR ELIGIBLE AND BORDERLINE/HIGH INCOME CASES")
    print("=" * 80)

    # 1. Low income result
    res_low = {
        'eligible': True,
        'confidence': 0.95,
        'needs_verification': False,
        'benefits': [
            {'name': 'Monthly Subsidy', 'description': 'PHP 1,500 cash assistance per child'},
            {'name': 'VAT Exemption & Discounts', 'description': '10% discount and VAT exemption'},
            {'name': 'Comprehensive Health Services', 'description': 'Discounts on medicines'},
            {'name': 'Educational Assistance', 'description': 'Scholarship programs'},
            {'name': 'Housing Benefits', 'description': 'Priority allocation'},
            {'name': 'Flexible Work Schedule', 'description': '7 days leave'}
        ],
        'applicant_info': {
            'name': 'Carmela Reyes',
            'age': 22,
            'civil_status': 'Single',
            'solo_parent_status': 'Single',
            'address': 'Labuin',
            'contact': '09123456789',
            'occupation': 'Unemployed',
            'monthly_income': 1508,
            'total_family_income': 1508,
            'number_of_children': 2
        },
        'family_members': []
    }

    # 2. Borderline income result
    res_borderline = {
        'eligible': True,
        'confidence': 0.88,
        'needs_verification': True,
        'benefits': [
            {'name': 'Monthly Subsidy (Pending Verification)', 'description': 'PHP 1,500 cash assistance per child, pending verification'},
            {'name': 'VAT Exemption & Discounts (Pending Verification)', 'description': 'VAT discount and exemption, pending verification'},
            {'name': 'Comprehensive Health Services', 'description': 'Discounts on medicines'},
            {'name': 'Educational Assistance', 'description': 'Scholarship programs'},
            {'name': 'Housing Benefits', 'description': 'Priority allocation'},
            {'name': 'Flexible Work Schedule', 'description': '7 days leave'}
        ],
        'applicant_info': {
            'name': 'Maria Santos',
            'age': 35,
            'civil_status': 'Separated',
            'solo_parent_status': 'Separated',
            'address': 'Calios',
            'contact': '09123456789',
            'occupation': 'Employed',
            'monthly_income': 42000,
            'total_family_income': 42000,
            'number_of_children': 2
        },
        'family_members': []
    }

    # 3. High income result
    res_high = {
        'eligible': True,
        'confidence': 0.92,
        'needs_verification': False,
        'benefits': [
            {'name': 'Comprehensive Health Services', 'description': 'Discounts on medicines'},
            {'name': 'Educational Assistance', 'description': 'Scholarship programs'},
            {'name': 'Housing Benefits', 'description': 'Priority allocation'},
            {'name': 'Flexible Work Schedule', 'description': '7 days leave'}
        ],
        'applicant_info': {
            'name': 'Dominic Villanueva',
            'age': 55,
            'civil_status': 'Separated',
            'solo_parent_status': 'Separated',
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
