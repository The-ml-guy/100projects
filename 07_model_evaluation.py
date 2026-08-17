#model_evaluation
#Comprehensive Metrics.
#Generate detailed evaluation reports including Confusion Matrix and Precision/Recall metrics.Pythonfrom sklearn.datasets import load_breast_cancer
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
