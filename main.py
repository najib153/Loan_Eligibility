
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import load_data, preprocess_data, show_data
from modeling import split_data, train_and_evaluate_models

if __name__ == "__main__":
    df = load_data('data\data_file')
    show_data(df)
    df = preprocess_data(df)
    xtrain, xtest, ytrain, ytest = split_data(df)
    print("Data processing complete.")
    show_data(xtrain)
    show_data(xtest)
    train_and_evaluate_models(xtrain, xtest, ytrain, ytest)

