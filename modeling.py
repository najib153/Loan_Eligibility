import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    df = df.copy()  # Ensure we work on a copy of the DataFrame
    
    # Drop Loan_ID if present
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'], errors='ignore')


    # Fill missing values
    df['Gender'] = df['Gender'].fillna('Male')
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    
    # One-hot encoding
    df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'], dtype=int)
    print(xtrain.dtypes)

    #categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    #df = pd.get_dummies(df, columns=categorical_columns, dtype=int)  
    return df



def split_data(df):
    try:
        X = df.drop('Loan_Approved', axis=1)  # Replace 'TargetColumn' with your target variable's column name
        y = df['Loan_Approved']
        
        # Split the data into training and testing sets (80% train, 20% test)
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return xtrain, xtest, ytrain, ytest
    except Exception as e:
        raise ValueError(f"Error during data splitting: {str(e)}")
    


def train_and_evaluate_models(xtrain, xtest, ytrain, ytest):
    try:
        models = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier()
        }
        
        results = {}
        trained_models = {}  # Store the actual trained models
        
        for model_name, model in models.items():
            model.fit(xtrain, ytrain)
            ypred = model.predict(xtest)
            accuracy = accuracy_score(ytest, ypred)
            results[model_name] = accuracy
            trained_models[model_name] = model  # Store the trained model object
        
        return results, trained_models  # Return both results and models
    except Exception as e:
        raise ValueError(f"Error during model training: {str(e)}")
    
    
def make_prediction(new_data, model):
    # Optionally check input shape
    if hasattr(model, "n_features_in_"):
        if new_data.shape[1] != model.n_features_in_:
            raise ValueError(f"Input has {new_data.shape[1]} features, expected {model.n_features_in_}")
    
    return model.predict(new_data)

def get_user_input():
    """Collect all necessary fields that the model was trained on"""
    # Basic info
    applicant_income = st.number_input("Applicant Income ($)", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0)
    
    # Loan details
    loan_amount = st.number_input("Loan Amount ($)", min_value=0)
    loan_term = st.number_input("Loan Term (months)", min_value=0)
    credit_history = st.selectbox("Credit History", [1, 0])
    
    # Personal details
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    # Create DataFrame with ALL expected columns
    input_data = {
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_history],
        
        # One-hot encoded columns (initialize all to 0)
        'Gender_Female': [0],
        'Gender_Male': [0],
        'Married_No': [0],
        'Married_Yes': [0],
        'Dependents_0': [0],
        'Dependents_1': [0],
        'Dependents_2': [0],
        'Dependents_3+': [0],
        'Education_Graduate': [0],
        'Education_Not Graduate': [0],
        'Self_Employed_No': [0],
        'Self_Employed_Yes': [0],
        'Property_Area_Rural': [0],
        'Property_Area_Semiurban': [0],
        'Property_Area_Urban': [0]
    }

    # Update the relevant one-hot columns
    input_data[f'Gender_{gender}'] = [1]
    input_data[f'Married_{"Yes" if married == "Yes" else "No"}'] = [1]
    input_data[f'Dependents_{dependents.replace("+", "+")}'] = [1]
    input_data[f'Education_{"Not Graduate" if education == "Not Graduate" else "Graduate"}'] = [1]
    input_data[f'Self_Employed_{"Yes" if self_employed == "Yes" else "No"}'] = [1]
    input_data[f'Property_Area_{property_area}'] = [1]

    return pd.DataFrame(input_data)