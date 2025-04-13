import streamlit as st
import pandas as pd
import joblib
from data_processing import load_data, preprocess_data, show_data
from modeling import split_data, train_and_evaluate_models, make_prediction, get_user_input


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
                try:
                    xtrain, xtest, ytrain, ytest = split_data(df)
                    results, trained_models = train_and_evaluate_models(xtrain, xtest, ytrain, ytest)
                    
                    if results is not None:
                        st.success("Model training completed!")
                        st.write("### Model Performance")
                        
                        for model, accuracy in results.items():
                            st.metric(label=model, value=f"{accuracy:.2%}")
                        
                        # Get the best model (actual object, not name)
                        best_model_name = max(results, key=results.get)
                        best_model = trained_models[best_model_name]  # Get the trained model object
                        
                        # Save the model and feature columns
                        joblib.dump((best_model, xtrain.columns.tolist()), "trained_model.pkl")
                        
                        st.bar_chart(pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']))
                    else:
                        st.error("Model training failed or returned no results.")
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
        else:
            st.error("No data available for training")

elif options == "Predict":
    st.subheader("Make Predictions")
    trained_model = load_trained_model()
    
    if trained_model is None:
         st.error("Please train a model first!")
         
    
    new_data = get_user_input()
    
    if st.button("Predict"):
        try:
            # Get the feature columns the model expects
            _, feature_columns = joblib.load("trained_model.pkl")
            
            # Ensure all columns exist (fill missing with 0)
            for col in feature_columns:
                if col not in new_data.columns:
                    new_data[col] = 0
            
            # Reorder columns to match training data
            new_data = new_data[feature_columns]
            
            # Make prediction
            prediction = trained_model.predict(new_data)
            result = "Approved" if prediction[0] == 1 else "Not Approved"
            st.success(f"Loan Status: {result}")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
  
# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit")
