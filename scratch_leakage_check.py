import pandas as pd

df = pd.read_csv('data/solo_parent_dataset.csv')

print("--- Eligibility vs Civil_Status ---")
print(pd.crosstab(df['Civil_Status'], df['Eligibility']))

print("\n--- Eligibility vs Gender (Sex) ---")
print(pd.crosstab(df['Gender'], df['Eligibility']))

print("\n--- Eligibility vs Civil_Status & Gender combined ---")
print(pd.crosstab([df['Civil_Status'], df['Gender']], df['Eligibility']))

print("\n--- Eligibility vs Type_of_Solo_Parent ---")
print(pd.crosstab(df['Type_of_Solo_Parent'], df['Eligibility']))
