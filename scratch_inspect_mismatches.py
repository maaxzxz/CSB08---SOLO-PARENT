import pandas as pd
df = pd.read_csv('data/solo_parent_dataset.csv')
for idx in [512, 579, 582]:
    print(f"\n--- Index {idx} ---")
    row = df.loc[idx]
    for col in df.columns:
        print(f"  {col}: {row[col]}")
