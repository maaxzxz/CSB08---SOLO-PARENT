import pandas as pd
df = pd.read_csv('data/solo_parent_dataset.csv')
not_eligible = df[df['Eligibility'] == 'Not Eligible']
print("--- Number_of_Dependents counts for Not Eligible records ---")
print(not_eligible['Number_of_Dependents'].value_counts(dropna=False))

print("\n--- Let's see the details of the Not Eligible records that have dependents > 0 ---")
print(not_eligible[not_eligible['Number_of_Dependents'] > 0][['Full_Name', 'Age', 'Number_of_Dependents', 'With_Minor', 'With_PWD', 'Eligibility']])
