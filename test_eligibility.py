import sys
import os

from app import app, assess_eligibility

client = app.test_client()


def with_engine_fields(data, youngest_child_age='5'):
    """Fills in the Section IV fields engine.evaluate() requires
    (added when app.py was unified onto rules.json/engine.py), defaulted
    to values that don't themselves grant/deny anything, so each test case
    above stays focused on the field(s) it's actually exercising."""
    data = dict(data)
    data.setdefault('youngest_child_age', youngest_child_age)
    data.setdefault('receiving_other_govt_cash_aid', 'No')
    data.setdefault('formal_philhealth_member', 'No')
    data.setdefault('dependent_currently_studying', 'No')
    return data


def run_test_cases():
    print("=" * 80)
    print("RUNNING AUTOMATED ELIGIBILITY & DYNAMIC BENEFITS TESTS")
    print("=" * 80)

    # Test Case 1: Low-Income Solo Parent
    case_low = with_engine_fields({
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
        'solo_parent_status': 'single_parent_unmarried',
        'number_of_dependent_children': '2',
        'with_pwd': 'No'
    })

    # Test Case 2: Borderline Income Solo Parent (between the minimum-wage
    # subsidy cutoff and the PHP 250,000/year VAT-exemption cap defined in
    # rules.json) — under the current model there's no separate "pending
    # verification" tier; this just checks the cash subsidy correctly drops
    # out while VAT/other benefits remain.
    case_borderline = with_engine_fields({
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
        'solo_parent_status': 'separated_divorced',
        'category_duration_answer': '12',
        'number_of_dependent_children': '2',
        'with_pwd': 'No'
    })

    # Test Case 3: High-Income Solo Parent
    case_high = with_engine_fields({
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
        'solo_parent_status': 'separated_divorced',
        'category_duration_answer': '12',
        'number_of_dependent_children': '2',
        'with_pwd': 'No'
    })

    # Test Case 4: Non-Eligible Parent (zero dependents)
    case_non_eligible = with_engine_fields({
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
        'solo_parent_status': 'single_parent_unmarried',
        'number_of_dependent_children': '0',
        'with_pwd': 'No'
    }, youngest_child_age='')

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
        for benefit_name in ["Cash Subsidy", "VAT Discount/Exemption", "PhilHealth", "Livelihood", "Housing Priority", "Educational Support"]:
            if benefit_name in html:
                print(f"  - [FOUND] {benefit_name}")
            else:
                print(f"  - [NOT FOUND] {benefit_name}")

if __name__ == '__main__':
    run_test_cases()