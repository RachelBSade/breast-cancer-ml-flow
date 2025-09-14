
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_path = "C:/Users/rache/Desktop/machine learning/cancer_train.csv"
test_path  = "C:/Users/rache/Desktop/machine learning/cancer_test.csv"

df_train = pd.read_csv(train_path)
df_test  = pd.read_csv(test_path)

print("Train shape:", df_train.shape, "| Test shape:", df_test.shape)
print("Missing (train):", int(df_train.isna().sum().sum()), "| Missing (test):", int(df_test.isna().sum().sum()))

features = [c for c in df_train.columns if c != "target"]
target = "target"

print(df_train.head())
print(df_train[features].describe().T)

print("Train target distribution:")
print(df_train[target].value_counts())
print("\nTest target distribution:")
print(df_test[target].value_counts())

# Histograms for selected features
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
cols_to_plot = ['mean radius', 'mean texture', 'mean concavity', 'worst radius', 'worst concavity', 'mean area']
for ax, col in zip(axes.ravel(), cols_to_plot):
    ax.hist(df_train[col], bins=30)
    ax.set_title(col)
plt.tight_layout()
plt.show()
