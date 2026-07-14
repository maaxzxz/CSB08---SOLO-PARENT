import pandas as pd
df = pd.read_csv('data/solo_parent_dataset.csv')
sub = df[(df['With_Minor'] == 'No') & (df['Number_of_Dependents'] > 0)]
print(f"Total records with With_Minor == 'No' and Dependents > 0: {len(sub)}")
print(sub['Eligibility'].value_counts(dropna=False))
print("\nLet's print all of them:")
print(sub[['Full_Name', 'Age', 'Number_of_Dependents', 'With_PWD', 'Eligibility']])
