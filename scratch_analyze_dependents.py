import pandas as pd

df = pd.read_csv('data/solo_parent_dataset.csv')

print("--- Eligibility when With_Minor == 'No' and With_PWD == 'No' ---")
sub1 = df[(df['With_Minor'] == 'No') & (df['With_PWD'] == 'No')]
print(sub1['Eligibility'].value_counts(dropna=False))

print("\n--- Eligibility when With_Minor == 'No' and With_PWD == 'Yes' ---")
sub2 = df[(df['With_Minor'] == 'No') & (df['With_PWD'] == 'Yes')]
print(sub2['Eligibility'].value_counts(dropna=False))

print("\n--- Eligibility when With_Minor == 'Yes' ---")
sub3 = df[df['With_Minor'] == 'Yes']
print(sub3['Eligibility'].value_counts(dropna=False))
