import pandas as pd
import numpy as np

df = pd.read_csv('data/solo_parent_dataset.csv')
print("--- Type_of_Solo_Parent counts ---")
print(df['Type_of_Solo_Parent'].value_counts(dropna=False))

print("\n--- Civil_Status counts ---")
print(df['Civil_Status'].value_counts(dropna=False))

print("\n--- Eligibility counts ---")
print(df['Eligibility'].value_counts(dropna=False))

# Let's test custom rules mapping:
# HTML form inputs for solo_parent_status: single, widowed, separated, abandoned, ofw, annulled
# We can map them in the test.
mismatches_rule_vs_csv = 0
for idx, row in df.iterrows():
    # Map CSV Type_of_Solo_Parent to what the form would submit
    csv_type = str(row['Type_of_Solo_Parent']).strip().lower()
    
    # Map 'unmarried' to 'single' (as in form), 'ofw spouse' to 'ofw', others as is
    if csv_type == 'unmarried':
        form_status = 'single'
    elif csv_type == 'ofw spouse':
        form_status = 'ofw'
    else:
        form_status = csv_type # 'widowed', 'abandoned', 'separated', 'annulled'
        
    birthday = pd.to_datetime(row['Birthday']).strftime('%Y-%m-%d')
    
    # We will simulate rules using the proposed update:
    has_valid_status = form_status in ['widowed', 'abandoned', 'separated', 'single', 'ofw', 'annulled']
    
    age = row['Age']
    number_of_children = row['Number_of_Dependents']
    
    # Is eligible rule check
    rule_eligible = has_valid_status and (number_of_children > 0) and (age >= 18)
    csv_eligible = (row['Eligibility'] == 'Eligible')
    
    if rule_eligible != csv_eligible:
        mismatches_rule_vs_csv += 1
        print(f"Mismatch at index {idx}: Type={row['Type_of_Solo_Parent']}, Dependents={number_of_children}, Age={age}, CSV_Eligible={csv_eligible}, Rule_Eligible={rule_eligible}")

print(f"\nRemaining mismatches with updated rule list: {mismatches_rule_vs_csv}")
