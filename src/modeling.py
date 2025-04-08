import pandas as pd
import numpy as np
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
    """Split data into training and testing sets."""
    x = df.drop('Loan_Approved', axis=1)
    y = df['Loan_Approved']
    return train_test_split(x, y, test_size=0.2, random_state=123)

def train_and_evaluate_models(xtrain, xtest, ytrain, ytest):
    """Train and evaluate multiple models."""
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }
    
    for name, model in models.items():
        model.fit(xtrain, ytrain)
        predictions = model.predict(xtest)
        accuracy = accuracy_score(ytest, predictions)
        print(f"{name} Accuracy: {accuracy:.4f}")

