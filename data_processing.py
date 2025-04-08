import pandas as pd
import os

def load_data(filepath):
    """Load dataset from a CSV file."""
    file = os.path.join("data", filepath)  # Correct path handling
    return pd.read_csv(file)

def show_data(data, n_rows=5):
    
    #Displays the first few rows of a DataFrame or array
    
    if isinstance(data, pd.DataFrame):
        print("\n" + "="*165)
        print(f"Data Shape: {data.shape}")
        print("First {} rows:".format(n_rows))
        print(data.head(n_rows))
        print("="*165 + "\n")
    else:
        print("\n" + "="*25)
        print("Data Sample:")
        print(data[:n_rows] if hasattr(data, '__len__') else data)
        print("="*25 + "\n")


def preprocess_data(df):
    """Perform data cleaning and feature engineering."""
    # Drop Loan_ID
    df = df.drop('Loan_ID', axis=1)
    
    

    df = df.copy()  # Ensure we work on a copy of the DataFrame
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
    
    return df