import pandas as pd
df = pd.read_csv('data/solo_parent_dataset.csv')
idxs = [44, 66, 276, 330, 342, 355, 512, 529, 538, 566, 572]
for idx in idxs:
    row = df.loc[idx]
    print(f"\n--- Index {idx} | Name: {row['Full_Name']} | Eligible: {row['Eligibility']} ---")
    print(f"  Age: {row['Age']} | Dependents: {row['Number_of_Dependents']} | Minor: {row['With_Minor']} | PWD: {row['With_PWD']}")
    print(f"  Remarks: {row['Remarks']}")
