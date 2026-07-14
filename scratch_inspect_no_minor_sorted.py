import pandas as pd
df = pd.read_csv('data/solo_parent_dataset.csv')
no_minor = df[df['With_Minor'] == 'No'].copy()

# Sort by Monthly_Income
no_minor = no_minor.sort_values('Monthly_Income')
for idx, row in no_minor.iterrows():
    print(f"Index: {idx:3d} | Age: {row['Age']} | PWD: {row['With_PWD']} | Income: {row['Monthly_Income']:5.0f} | Eligible: {row['Eligibility']}")
