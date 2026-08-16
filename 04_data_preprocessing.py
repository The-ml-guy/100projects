
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
3.03_data_splitting.py:Train/Test Split.Separate features ($X$) from labels ($y$) and split data into training and evaluation sets.Pythonfrom sklearn.datasets import load_iris
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
4.04_classification_model.py:Train a Classifier.Train a Logistic Regression model on split data and calculate accuracy.Pythonfrom sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load and split
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict & Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Classification Accuracy: {accuracy * 100:.2f}%")
5.05_regression_model.py:Train a Regressor.Predict a continuous value using Linear Regression and evaluate with Mean Squared Error.Pythonfrom sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load regression dataset
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict & Evaluate
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
6.06_model_evaluation.py:Comprehensive Metrics.Generate detailed evaluation reports including Confusion Matrix and Precision/Recall metrics.Pythonfrom sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load binary classification dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, preds))

print("\n=== Classification Report ===")
print(classification_report(y_test, preds))
