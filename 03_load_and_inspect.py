#Master loading tabular data with pandas and inspect basic structure/summary statistics.
#we will import the sklearn dataset for these
import pandas as pd

#load the dataset from Skearn
from sklearn.datasets import load_iris

# Load a built-in dataset and convert to pandas DataFrame
iris = load_iris(as_frame=True)
df = iris.frame

print("=== Dataset Shape ===")
print(df.shape)

print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Dataset Summary ===")
print(df.info())
