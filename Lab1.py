import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load Dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print(df.head())
print(df.tail())
print(df.shape)

# Data Quality
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())

# Class Distribution
print(df["species"].value_counts())
print(df["species"].value_counts(normalize=True) * 100)

# Summary Statistics
X = df.drop(columns="species")
print(X.describe())
print(df.groupby("species", observed=True).mean(numeric_only=True))

# Highest and Lowest
mean = df.groupby("species", observed=True)["petal length (cm)"].mean()
print("Highest:", mean.idxmax())
print("Lowest:", mean.idxmin())

# Histogram
X.hist()
plt.show()

# One Scatter Plot
sns.scatterplot(data=df, x="petal length (cm)",
                y="petal width (cm)", hue="species")
plt.show()