import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["cultivar"] = wine.target

X, y = df.drop(columns="cultivar"), df["cultivar"]

print(df.head(), df.shape)
print(df.isnull().sum())
print(X.describe())

# Feature Range
print((X.max()-X.min()).agg(["idxmax", "idxmin"]))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Distance
d = (X_train.iloc[0]-X_train.iloc[1])**2
print(np.sqrt(d.sum()), d["proline"]/d.sum()*100)

# Scaling
data = {"Unscaled": (X_train, X_test)}

for s in [StandardScaler(), MinMaxScaler(), RobustScaler()]:
    data[type(s).__name__] = (s.fit_transform(X_train), s.transform(X_test))

# Outliers
q = X_train["magnesium"].quantile([.25, .75])
iqr = q[.75]-q[.25]
print("Outliers:", X_train["magnesium"].between(
    q[.25]-1.5*iqr, q[.75]+1.5*iqr).eq(False).sum())

# One Plot
sns.boxplot(y=RobustScaler().fit_transform(X_train[["magnesium"]]).ravel())
plt.show()

# Model Accuracy
for m in [KNeighborsClassifier(), LogisticRegression(max_iter=5000),
          DecisionTreeClassifier(random_state=42)]:
    for n, (tr, te) in data.items():
        m.fit(tr, y_train)
        print(type(m).__name__, n, accuracy_score(y_test, m.predict(te)))