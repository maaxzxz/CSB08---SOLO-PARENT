import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('data/solo_parent_dataset.csv')

# Features we actually use in training + Civil_Status & Gender
feature_cols = [
    'Age', 'Gender', 'Civil_Status', 'Educational_Attainment', 
    'Employment_Status', 'Monthly_Income', 'Number_of_Dependents', 
    'With_Minor', 'With_PWD', 'Type_of_Solo_Parent'
]

X = df[feature_cols].copy()
y = (df['Eligibility'] == 'Eligible').astype(int)

# Encode categoricals using OrdinalEncoder
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

encoder = OrdinalEncoder()
X_encoded = X.copy()
X_encoded[categorical_cols] = encoder.fit_transform(X_encoded[categorical_cols].astype(str))

# Fit decision tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_encoded, y)

print("Decision Tree Rules:")
print(export_text(dt, feature_names=X.columns.tolist()))

for col in categorical_cols:
    idx = categorical_cols.index(col)
    print(f"{col} encoding mapping:")
    for val, code in zip(encoder.categories_[idx], range(len(encoder.categories_[idx]))):
        print(f"  {val} -> {code}")
