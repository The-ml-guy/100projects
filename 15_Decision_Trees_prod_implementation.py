
#The fastest and most robust way to implement a decision tree algorithm in Python is by using the Scikit-Learn Library. 
#It provides highly optimized, production-ready classes for both classification and regression tasks.Below, you will find the standard implementation using scikit-learn,
#followed by a lightweight implementation written entirely from scratch using NumPy to show you how the underlying math works.

#Method 1: Using Scikit-Learn (Production-Ready)This approach handles the heavy lifting like tree pruning, optimal splitting strategy, and evaluation metrics


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# 1. Load sample dataset (Iris flower dataset)
data = load_iris()
X = data.data
y = data.target

# 2. Split dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize the Decision Tree Classifier
# 'entropy' or 'gini' measures the quality of a split. max_depth prevents overfitting.
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)

# 4. Train the model
clf.fit(X_train, y_train)

# 5. Make predictions
y_pred = clf.predict(X_test)

# 6. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 7. Optional: Plot the visual tree layout
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()
