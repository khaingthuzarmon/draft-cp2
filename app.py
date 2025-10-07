import streamlit as st
import pickle
import numpy as np

# Load the models
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please make sure they are in the correct directory.")
    st.stop()

# App Title
st.title('Machine Learning Model Deployment')

# Sidebar for user input
st.sidebar.header('User Input Features')

# Function to get user input
def user_input_features():
    # Replace with your actual feature names
    feature1 = st.sidebar.slider('Feature 1', 0.0, 10.0, 5.0)
    feature2 = st.sidebar.slider('Feature 2', 0.0, 10.0, 5.0)
    feature3 = st.sidebar.slider('Feature 3', 0.0, 10.0, 5.0)
    data = {'feature1': feature1,
            'feature2': feature2,
            'feature3': feature3}
    features = np.array([data['feature1'], data['feature2'], data['feature3']]).reshape(1, -1)
    return features

# Get user input
input_df = user_input_features()

# Main panel
st.subheader('User Input parameters')
st.write(input_df)

# Prediction
if st.button('Predict'):
    # Scale the input
    scaled_input = scaler.transform(input_df)

    # K-means prediction
    kmeans_prediction = kmeans.predict(scaled_input)

    # Random Forest prediction
    rf_prediction = rf_model.predict(scaled_input)
    
    st.subheader('Prediction Results')
    st.write(f'K-Means Cluster: {kmeans_prediction[0]}')
    st.write(f'Random Forest Prediction: {rf_prediction[0]}')
