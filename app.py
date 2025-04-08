import streamlit as st
import pandas as pd
from data_processing import load_data, preprocess_data
from modeling import split_data, train_and_evaluate_models

# Set page config
st.set_page_config(page_title="Loan Eligibility Predictor", layout="wide")

# Title and description
st.title("Loan Eligibility Prediction System")
st.write("""
This application predicts whether a loan application will be approved based on various features.
Upload your data or use our sample data to get predictions.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select Option:", ["Home", "Data Overview", "Model Training", "Predict"])

# Load data
@st.cache_data
def load_and_preprocess():
    try:
        df = load_data("data\credit.csv")
        df = preprocess_data(df)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

df = load_and_preprocess()

if options == "Home":
    st.subheader("Welcome to the Loan Eligibility Predictor")
    st.image("https://cdn.pixabay.com/photo/2017/08/06/22/01/credit-2596887_1280.jpg", width=600)
    st.write("""
    ### How to use this application:
    1. **Data Overview**: View the dataset and statistics
    2. **Model Training**: Train machine learning models and view their performance
    3. **Predict**: Make predictions on new data
    """)

elif options == "Data Overview":
    st.subheader("Data Overview")
    if df is not None:
        st.write("### Sample Data")
        st.dataframe(df.head())
        
        st.write("### Data Statistics")
        st.dataframe(df.describe())
        
        st.write("### Missing Values")
        st.dataframe(df.isnull().sum().to_frame("Missing Values"))

elif options == "Model Training":
    st.subheader("Model Training and Evaluation")
    
    if st.button("Train Models"):
        if df is not None:
            with st.spinner("Training models..."):
                xtrain, xtest, ytrain, ytest = split_data(df)
                results = train_and_evaluate_models(xtrain, xtest, ytrain, ytest)
                
                st.success("Model training completed!")
                st.write("### Model Performance")
                
                for model, accuracy in results.items():
                    st.metric(label=model, value=f"{accuracy:.2%}")
                
                # Plot results
                st.bar_chart(pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']))
        else:
            st.error("No data available for training")

elif options == "Predict":
    st.subheader("Make Predictions")
    st.write("This feature will be implemented in the next version.")
    st.info("Coming soon: Interactive form for making predictions on new loan applications")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit")