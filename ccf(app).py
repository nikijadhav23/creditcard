import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load the data
data = pd.read_csv('creditcard.csv')

# Display basic information about the data
st.write(data.head())
st.write(data.info())
st.write(f"Shape of the data: {data.shape}")
st.write(f"Class distribution: {data['Class'].value_counts()}")
st.write(data.describe())

# Separate the data into legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data['Class'] == 1]

st.write(f"Fraudulent transactions: {fraud.shape}")
st.write(f"Legitimate transactions: {legit.shape}")

# Sample legitimate transactions to balance the dataset
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

st.write(f"Balanced class distribution: {data['Class'].value_counts()}")

# Display mean values for each class
st.write(data.groupby('Class').mean())

# Split the data into training and testing sets
X = data.drop('Class', axis=1)
Y = data['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=2)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(max_iter=2000)  # Increased iterations
model.fit(X_train_scaled, Y_train)

# Evaluate model performance
train_acc = accuracy_score(Y_train, model.predict(X_train_scaled))
test_acc = accuracy_score(Y_test, model.predict(X_test_scaled))

st.write(f"Training accuracy: {train_acc:.2f}")
st.write(f"Testing accuracy: {test_acc:.2f}")

# Web app interface
st.title("Credit Card Fraud Detection Model")
input_df = st.text_input('Enter all required feature values separated by commas')

submit = st.button("Submit")

if submit:
    try:
        # Get input feature values
        input_df_lst = list(map(float, input_df.split(',')))
        features = np.array(input_df_lst).reshape(1, -1)
        features_scaled = scaler.transform(features)  # Scale the input features
        
        # Make prediction
        prediction = model.predict(features_scaled)
        
        # Display result
        if prediction[0] == 0:
            st.write("Legitimate transaction")
        else:
            st.write("Fraudulent transaction")
    except ValueError:
        st.write("Please enter valid numeric values separated by commas.")
