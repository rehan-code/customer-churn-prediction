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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)

def evaluate_and_save_model(model, X_train, X_test, y_train, y_test, filename):
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
   print(f"{model.__class__.__name__} Accuracy: {accuracy: .4f}")
   print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
   print("-----------")

   with open(filename, "wb") as file:
      pickle.dump(model, file)
   
   print(f"Model saved as {filename}")

# Check and compare models and accuracy
xgb_model = xgb.XGBClassifier(random_state=42)
evaluate_and_save_model(xgb_model, X_train, X_test, y_train, y_test, "xgb_model.pkl")

dt_model = DecisionTreeClassifier(randome_state=42)
evaluate_and_save_model(dt_model, X_train, X_test, y_train, y_test, "dt_model.pkl")

rf_model = RandomForestClassifier(random_state=42)
evaluate_and_save_model(dt_model, X_train, X_test, y_train, y_test, "dt_model.pkl")

nb_model = GaussianNB()
evaluate_and_save_model(nb_model, X_train, X_test, y_train, y_test, "nb_model.pkl")

knn_model = KNeighborsClassifier()
evaluate_and_save_model(knn_model, X_train, X_test, y_train, y_test, "knn_model.pkl")

svm_model = SVC(random_state=42)
evaluate_and_save_model(svm_model, X_train, X_test, y_train, y_test, "svm_model.pkl")
