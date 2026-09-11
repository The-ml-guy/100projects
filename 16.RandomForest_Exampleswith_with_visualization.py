#implemented the Randomforest dataset
#here we are implementing the sample dataset Breast_canmcer dataset 
#the result was good

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestRegressor

#Load the multi-output regressor dataset

lind = load_breast_cancer()
X = lind.data

y = lind.target

#Split the data(using a larger test size fraction since are only rows total)
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=1)

# Initialize the regressor
#We will set a small max_depth and simple_leaf to combat overfitting ont the  20 sample

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=2,
    random_state=1
)

#Now we will train the model
model.fit(X_train, y_train)

#now we will first predict
pred = model.predict(X_test)

print("target values", lind.target_names)
print("R2", r2_score(y_test, pred, multioutput='raw_values'))

# the accuracy score
#target values ['malignant' 'benign']
#R2 [0.82096955]



#used for the visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))


actual_means = np.mean(y_test, axis=0)
pred_means = np.mean(pred, axis=0)

# 5. Simple Bar Chart Plot
x_indexes = np.arange(len(lind.target_names))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x_indexes - width/2, actual_means, width, label='Actual Avg', color='skyblue')
plt.bar(x_indexes + width/2, pred_means, width, label='Predicted Avg', color='salmon')



plt.ylabel('average value')
plt.title('predicted value vs averages vs predicted average')
plt.legend()
plt.grid(axis='y', linestyle='--',alpha=0.7)
plt.tight_layout()
plt.show()
