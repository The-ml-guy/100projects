# Using the SVM algorithm in the Sklearn Breast cancer Dataset.
# Import necessary libraries
# in
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# 2. Split data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Initialize the Support Vector Classifier
model = SVC(kernel='linear', random_state=42)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Make predictions on the test set
predictions = model.predict(X_test)

# 6. Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=cancer.target_names))
