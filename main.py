import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get('GROQ_API_KEY'),
)


def load_model(filename):
  with open(filename, 'rb') as f:
    return pickle.load(f)


# load models
xgboost_model = load_model('xgb_model.pkl')
naive_bayes_model = load_model('nb_model.pkl')
random_forest_model = load_model('rf_model.pkl')
decision_tree_model = load_model('dt_model.pkl')
svm_model = load_model('svm_model.pkl')
knn_model = load_model('knn_model.pkl')
voting_classifier_model = load_model('voting_clf_model.pkl')
xgboost_SMOTE_model = load_model('xgboost-SMOTE.pkl')
xgboost_featureEngineered_model = load_model('xgboost-featureEngineered.pkl')


def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_products, has_credit_card, is_active_member,
                  estimated_salary):
  input_dict = {
      'CreditScore': credit_score,
      'Location': location,
      'Gender': gender,
      'Age': age,
      'Tenure': tenure,
      'Balance': balance,
      'NumOfProducts': num_products,
      'hasCrCard': has_credit_card,
      'IsActiveMember': is_active_member,
      'EstimatedSalary': estimated_salary,
      'Geography_France': 1 if location == 'France' else 0,
      'Geography_Germany': 1 if location == 'Germany' else 0,
      'Geography_Spain': 1 if location == 'Spain' else 0,
      'Gender_Male': 1 if gender == 'Male' else 0,
      'Genre_Female': 1 if gender == 'Female' else 0,
  }

  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict


def make_predictions(input_df, input_dict):
  probabilities = {
      'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
      'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
      'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
  }
  avg_probability = np.mean(list(probabilities.values()))

  st.markdown("### Model Probabilities")
  for model, prob in probabilities.items():
    st.write(f"{model} {prob}")
  st.write(f"Average Probability: {avg_probability}")

  return avg_probability


def explain_prediction(probability, input_dict, surname):
  prompt = f"""You are an expert data scientist at a bank, where you specialize in interpreting and explaining machine learning models.
  
  Your machine learning model has predicted that a customer name {surname} has a {round(probability*100, 1)}% probability of churning, based on the information provided below
  
  Here is the customer's information:
  {input_dict}
  
  Here are the machine learning model's top 10 most important features for predicting churn:
  
          Feature | Importance
  ------------------------------------
    NumOfProducts | 0.323888
   IsActiveMember | 0.164146
              Age | 0.157531
 Geograpy_Germany | 0.157531
          Balance | 0.157531
 Geography_France | 0.157531
    Gender_Female | 0.157531
  Geography_Spain | 0.157531
      CreditScore | 0.157531
  EstimatedSalary | 0.157531
        HasCrCard | 0.157531
           Tenure | 0.157531
      Gender_Male | 0.157531

  {pd.set_option('display.max_columns', None)}

  Here is a summary statisticts for churned customers:
  {df[df['Exited'] == 1].describe()}


  - If the customer has over a 405 risk of churning, generate a 3 sentence explanation of why they might not be at risk of churning.
  - Your explanation should be based on the customer's information, the summary statistics of churned and non-churned customers, and the feature importance provided.


  Don't mention the probability of churning, or the machine leaning model, or say anything like "Based on the machine learning model's prediction and top 10 most inmportant features", just explain the prediction.
  
  """

  raw_response = client.chat.completions.create(model="llama-3.2-3b-preview",
                                                messages=[{
                                                    "role": "user",
                                                    "content": prompt
                                                }])
  return raw_response.choices[0].message_content


# Streamlit app
st.title("Customer Churn Prediction")

df = pd.read_csv('churn.csv')

customers = [
    f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()
]
selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
  selected_customer_id = int(selected_customer_option.split(" - ")[0])
  print("Selected customer ID:", selected_customer_id)
  selected_surname = selected_customer_option.split(" - ")[1]
  print("Surname:", selected_surname)

  selected_customer = df.loc[df['CustomerId'] == selected_customer_id].iloc[0]
  print("Selected Customer", selected_customer)

  col1, col2 = st.columns(2)

  with col1:

    credit_score = st.number_input("Credit Score",
                                   min_value=300,
                                   max_value=350,
                                   value=int(selected_customer['CreditScore']))

    location = st.selectbox("Location", ["Spain", "France", "germany"],
                            index=["Spain", "France", "germany"
                                   ].index(selected_customer['Geography']))

    gender = st.radio("Gender", ["Male", "Female"],
                      index=0 if selected_customer['Gender'] == 'Male' else 1)

    age = st.number_input("Age",
                          min_value=18,
                          max_value=100,
                          value=int(selected_customer['Age']))

    tenure = st.number_input("Tenure (years)",
                             min_value=0,
                             max_value=50,
                             value=int(selected_customer['Tenure']))

  with col2:
    balance = st.number_input('Balance',
                              min_value=1,
                              max_value=10,
                              value=float(selected_customer['Balance']))

    num_products = st.number_input("Number of Products",
                                   min_value=1,
                                   max_value=10,
                                   value=int(
                                       selected_customer['NumOfProducts']))

    has_credit_card = st.number_input("Has Credit Card",
                                      value=bool(
                                          selected_customer['HasCrCard']))

    is_active_member = st.checkbox("Is Active Member",
                                   value=bool(
                                       selected_customer['IsActiveMember']))

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=float(selected_customer['EstimatedSalary']))

  input_df, input_dict = prepare_input(credit_score, location, gender, age,
                                       tenure, balance, num_products,
                                       has_credit_card, is_active_member,
                                       estimated_salary)

  avg_probaility = make_predictions(input_df, input_dict)
