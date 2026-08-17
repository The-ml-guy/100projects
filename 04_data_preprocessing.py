#Clean & Scale Data.Handle missing values, encode non-numeric labels, and scale numerical features using scikit-learn.Pythonimport pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Sample data with missing values and text labels
data = {
    'Age': [25, 30, None, 35],
    'Salary': [50000, 60000, 52000, None],
    'Purchased': ['No', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

# 1. Fill missing values with column median
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())

# 2. Encode categorical target to numbers (No -> 0, Yes -> 1)
le = LabelEncoder()
df['Purchased'] = le.fit_transform(df['Purchased'])

# 3. Scale numerical features to mean=0, variance=1
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['Age', 'Salary']])

print("=== Processed & Scaled Data ===")
print(scaled_features)



#3.03_data_splitting.py:
#Train/Test Split.Separate features ($X$) from labels ($y$) and split data into training and evaluation sets.Pythonfrom sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Total samples: {len(X)}")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")
