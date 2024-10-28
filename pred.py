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

# split training  and test data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# training the models
#  logistic regression model
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)