#Using Logistic Regression in the Breastcancer Dataset.
#using train test split befor the Scaler or standardization
# Never use Scaler.fit as try to find mean from test data and
#model can momorize from traiining data canbe overfitting
# Plotting the distribution of y_train labels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()


print(cancer)

x = cancer.data
y = cancer.target

#using train test split befor the Scaler or standardization

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 1000 )

# standardization starts here train X and test x
scaler = StandardScaler()

#Creating a list =to amke easy for logistic regression out 0 or 1
x_train_scaled = scaler.fit_transform(x_train)

# Never use Scaler.fit as try to find mean from test data and
x_test_scaled = scaler.transform(x_test)

model = LogisticRegression()
implementd = model.fit(x_train_scaled, y_train)

#model can momorize from traiining data canbe overfitting
#implementd.score(x_train_scaled, y_train)

#to find real performance
implementd.score(x_test_scaled, y_test)

plt.figure(figsize=(6, 3))
# Plotting the first feature against the second feature
plt.scatter(x_train_scaled[:, 0], x_train_scaled[:, 1], alpha=0.5)
plt.xlabel('Feature 0 (Scaled)')
plt.ylabel('Feature 1 (Scaled)')
plt.title('Scatter Plot of First Two Scaled Features')
plt.grid(True)
plt.ylim(-2, 2)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Plotting the distribution of y_train labels
plt.figure(figsize=(6, 4))
sns.countplot(y=y_train)

plt.show()

print(y_train)
