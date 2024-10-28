import pandas as pd
import matplotlib as plt
import seaborn as sns

df = pd.read_csv("churn.csv")

sns.set_style(style="whitegrid")
plt.figure(figsize=(12,10))

sns.countplot(x="Exited", data=df)
plt.title("Distribution of Churn")

sns.histplot(data=df, x="Age", kde=True)
plt.title("Age Distribution")

sns.scatterplot(data=df, x="CreditScore", y="Age", hue="Exited")
plt.title("Credit Score vs Age")

sns.boxplot(data=df, x="Exited", y="Balance")
plt.title("Balance Distribution by Churn")

sns.boxplot(data=df ,x="Exited", y="CreditScore")
plt.title("Credit Score Distribution by Churn")

# Preprocess of data
features = df.drop('Exited', axis=1)
target = df['Exited']
features = features.drop(["RowNumber",  "CustonerId", "Surname"], axis=1) #remove unnecessary rows

# handle missing values
features = features.dropna()

# one hot encoding (convert text fields to values)
features = pd.get_dummies(features, columns=["Geography", "Gender"])

