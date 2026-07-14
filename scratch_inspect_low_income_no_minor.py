import pandas as pd
df = pd.read_csv('data/solo_parent_dataset.csv')
no_minor_low_income = df[(df['With_Minor'] == 'No') & (df['Monthly_Income'] <= 4000)]
print(no_minor_low_income[['Full_Name', 'Age', 'With_PWD', 'Monthly_Income', 'Eligibility']])
