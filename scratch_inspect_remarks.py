import pandas as pd
df = pd.read_csv('data/solo_parent_dataset.csv')
idxs = [44, 66, 276, 330, 342, 355, 512, 529, 538, 566, 572]
print(df.loc[idxs, ['Full_Name', 'Age', 'Number_of_Dependents', 'With_Minor', 'With_PWD', 'Eligibility', 'Remarks']])
