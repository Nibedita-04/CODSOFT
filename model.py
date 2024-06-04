import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import warnings as wg
wg.filterwarnings("ignore")
import os
import streamlit as st

# Loading the dataset to a Pandas DataFrame
flower_data = pd.read_csv("IRIS.csv")

# feature variable and target variable
X  = flower_data.iloc[:,0:4]
Y = flower_data['species']

# training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.30, random_state = 42)

# Using Decision Tree Classifier
dtc = DecisionTreeClassifier(random_state = 1234)

# Fitting X_train, Y_train into model
dtc.fit(X_train, Y_train)

# web app
st.title("Iris Flower Classification Model")
input_df = st.text_input("Enter all required features values : ")
f = input_df.split(",")
data = pd.DataFrame({"sepal_length":[f[0]], "sepal_width":[f[1]], "petal_length":[f[2]],"petal_width":[f[3]]})


submit = st.button("Submit")

if submit:
    prediction = dtc.predict(data)
    
    if prediction == ['Iris-setosa']:
        st.write("IRIS-SETOSA")
    elif prediction == ['Iris-versicolor']:
        st.write("IRIS-VERSICOLOR")
    else:
        st.write("IRIS-VIGINICA")