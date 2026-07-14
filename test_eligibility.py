import sys
import os

from app import app, assess_eligibility

client = app.test_client()

def run_test_cases():
    print("=" * 80)
    print("RUNNING AUTOMATED ELIGIBILITY & DYNAMIC BENEFITS TESTS")
    print("=" * 80)

    # Test Case 1: Low-Income Solo Parent
    case_low = {
        'first_name': 'Carmela',
        'middle_name': 'R',
        'surname': 'Reyes',
        'birthday': '2004-02-20',
        'sex': 'female',
        'address': 'Labuin',
        'contact_no': '09123456789',
        'civil_status': 'single',
        'occupation': 'Unemployed',
        'monthly_income': '1508',
        'total_family_income': '1508',
        'solo_parent_status': 'single',
        'number_of_dependent_children': '2',
        'with_pwd': 'No'
    }

    # Test Case 2: Borderline Income Solo Parent
    # NOTE: income updated to PHP 18,000/month. The old value (42,000) no
    # longer lands in the "borderline/pending verification" tier now that
    # app.py uses the real RA 11861 thresholds (minimum wage ~15,860/month
    # and the PHP 250,000/year VAT-exemption cap, ~20,833/month) instead of
    # the old made-up ~30,000-45,000 range. 18,000 sits between the two real
    # thresholds, so it correctly triggers "needs_verification".
    case_borderline = {
        'first_name': 'Maria',
        'middle_name': 'B',
        'surname': 'Santos',
        'birthday': '1990-05-15',
        'sex': 'female',
        'address': 'Calios',
        'contact_no': '09123456789',
        'civil_status': 'separated',
        'occupation': 'Employed',
        'monthly_income': '18000',
        'total_family_income': '18000',
        'solo_parent_status': 'separated',
        'number_of_dependent_children': '2',
        'with_pwd': 'No'
    }

    # Test Case 3: High-Income Solo Parent
    case_high = {
        'first_name': 'Dominic',
        'middle_name': 'A',
        'surname': 'Villanueva',
        'birthday': '1980-10-22',
        'sex': 'male',
        'address': 'Labuin',
        'contact_no': '09123456789',
        'civil_status': 'separated',
        'occupation': 'Self-Employed',
        'monthly_income': '60000',
        'total_family_income': '60000',
        'solo_parent_status': 'separated',
        'number_of_dependent_children': '2',
        'with_pwd': 'No'
    }

    # Test Case 4: Non-Eligible Parent
    case_non_eligible = {
        'first_name': 'Teresa',
        'middle_name': 'M',
        'surname': 'Aquino',
        'birthday': '2005-03-03',
        'sex': 'female',
        'address': 'San Jose',
        'contact_no': '09123456789',
        'civil_status': 'single',
        'occupation': 'Employed',
        'monthly_income': '10947',
        'total_family_income': '10947',
        'solo_parent_status': 'single',
        'number_of_dependent_children': '0',
        'with_pwd': 'No'
    }

    test_cases = [
        ("Low-Income Case", case_low),
        ("Borderline Income Case", case_borderline),
        ("High-Income Case", case_high),
        ("Non-Eligible Case", case_non_eligible)
    ]

    for name, data in test_cases:
        print(f"\n--- Testing {name} ---")
        rule_res = assess_eligibility(data)
        print(f"Rule-Based - Eligible: {rule_res['eligible']}, Needs Verification: {rule_res['needs_verification']}")
        print("Rule-Based - Qualified Benefits:")
        for benefit in rule_res['benefits']:
            print(f"  - {benefit['name']}: {benefit['description']}")

        response = client.post('/submit-assessment', data=data)
        html = response.data.decode('utf-8')
        
        if "Eligible for Solo Parent Benefits" in html:
            print("Web Output: ELIGIBLE")
        elif "Needs Verification" in html:
            print("Web Output: NEEDS VERIFICATION")
        else:
            print("Web Output: NOT ELIGIBLE")

        print("Web Output - Benefits Rendered:")
        for benefit_name in ["Monthly Subsidy", "VAT Exemption", "Educational Assistance", "Housing Benefits", "Comprehensive Health Services", "Flexible Work Schedule"]:
            if benefit_name in html:
                print(f"  - [FOUND] {benefit_name}")
            else:
                print(f"  - [NOT FOUND] {benefit_name}")

if __name__ == '__main__':
    run_test_cases()