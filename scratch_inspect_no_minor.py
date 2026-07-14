import pandas as pd
df = pd.read_csv('data/solo_parent_dataset.csv')
no_minor = df[df['With_Minor'] == 'No']
print(f"Total records with With_Minor == 'No': {len(no_minor)}")
print("\n--- Eligible records with With_Minor == 'No' ---")
print(no_minor[no_minor['Eligibility'] == 'Eligible'][['Full_Name', 'Age', 'Number_of_Dependents', 'With_Minor', 'With_PWD', 'Disability_Category', 'Eligibility']])

print("\n--- Not Eligible records with With_Minor == 'No' ---")
print(no_minor[no_minor['Eligibility'] == 'Not Eligible'][['Full_Name', 'Age', 'Number_of_Dependents', 'With_Minor', 'With_PWD', 'Disability_Category', 'Eligibility']].head(10))
