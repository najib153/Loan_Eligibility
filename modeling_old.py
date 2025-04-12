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
        # Example models
        models = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier()
        }
        
        results = {}
        
        for model_name, model in models.items():
            model.fit(xtrain, ytrain)
            ypred = model.predict(xtest)
            accuracy = accuracy_score(ytest, ypred)
            results[model_name] = accuracy
        
        return results
    except Exception as e:
        raise ValueError(f"Error during model training: {str(e)}")

def make_prediction(new_data, model):
    """Make a prediction on new data using the trained model."""
    try:
        # Preprocess the new data
        new_data_processed = preprocess_data(new_data)
        
        # Make prediction
        prediction = model.predict(new_data_processed)
        return prediction
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")

def get_user_input():
        """Prompt the user to input data for prediction."""
        print("Please enter the following details for prediction:")

        # Input for Gender (male/female)
        gender = input("Gender (Male/Female): ")
        gender_male = 1 if gender.lower() == 'male' else 0  # One-hot encoding: 1 for Male, 0 for Female

        # Input for Marital Status (Yes/No)
        married = input("Married (Yes/No): ")
        married_yes = 1 if married.lower() == 'yes' else 0  # One-hot encoding: 1 for Yes, 0 for No

        # Input for Dependents (number of dependents)
        dependents = input("Dependents (0, 1, 2+): ")
        dependents_0 = 1 if dependents == '0' else 0  # One-hot encoding for 0 dependents
        dependents_1 = 1 if dependents == '1' else 0  # One-hot encoding for 1 dependent
        dependents_2_plus = 1 if dependents == '2+' else 0  # One-hot encoding for 2+ dependents

        # Input for Education (Graduate/Not Graduate)
        education = input("Education (Graduate/Not Graduate): ")
        education_graduate = 1 if education.lower() == 'graduate' else 0  # One-hot encoding for Graduate

        # Input for Self-Employed (Yes/No)
        self_employed = input("Self-Employed (Yes/No): ")
        self_employed_no = 1 if self_employed.lower() == 'no' else 0  # One-hot encoding for No

        # Input for Property Area (Urban/Semiurban/Rural)
        property_area = input("Property Area (Urban/Semiurban/Rural): ")
        property_area_urban = 1 if property_area.lower() == 'urban' else 0  # One-hot encoding for Urban

        # Input for Loan Amount
        loan_amount = float(input("Loan Amount: "))

        # Input for Loan Amount Term
        loan_term = int(input("Loan Amount Term (months): "))

        # Input for Credit History (1 or 0)
        credit_history = int(input("Credit History (1 for Yes, 0 for No): "))

        # Create a DataFrame from user input
        new_data = pd.DataFrame({
            'Gender_Male': [gender_male],
            'Married_Yes': [married_yes],
            'Dependents_0': [dependents_0],
            'Dependents_1': [dependents_1],
            'Dependents_2+': [dependents_2_plus],
            'Education_Graduate': [education_graduate],
            'Self_Employed_No': [self_employed_no],
            'Property_Area_Urban': [property_area_urban],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_term],
            'Credit_History': [credit_history]
         })

        return new_data

# Load trained model once and store it
@st.cache_resource
def load_trained_model():
        try:
            model = joblib.load("trained_model.pkl")
            return model
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None