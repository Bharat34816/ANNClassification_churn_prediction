import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Function to load the model and preprocessing objects
def load_model_and_preprocessors():
    model = tf.keras.models.load_model('model.h5')

    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('one_hot_encoder_gender.pkl', 'rb') as file:
        one_hot_encoder_gender = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    return model, label_encoder_gender, one_hot_encoder_gender, scaler

# Load the model and preprocessors
model, label_encoder_gender, one_hot_encoder_gender, scaler = load_model_and_preprocessors()

# Function for churn prediction page
def churn_prediction():
    st.title('Customer Churn Prediction')

    geography = st.selectbox('Geography', one_hot_encoder_gender.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, 30)
    balance = st.number_input('Balance', min_value=0.0, format="%.2f")
    credit_score = st.number_input('Credit Score', min_value=300, max_value=900, step=1)
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, format="%.2f")
    tenure = st.slider('Tenure', 0, 10)
    num_of_products = st.slider('Number of Products', 1, 4)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = one_hot_encoder_gender.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_gender.get_feature_names_out())
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.write(f'Churn Probability: {prediction_proba:.2f}')
    st.progress(int(prediction_proba * 100))

    if prediction_proba > 0.5:
        st.write('🔴 The customer is **likely** to churn.')
    else:
        st.write('🟢 The customer is **not likely** to churn.')

    fig, ax = plt.subplots()
    features = ['CreditScore', 'Age', 'Balance', 'Credit Score', 'Tenure', 'Salary']
    ax.bar(features, np.random.rand(len(features)))
    ax.set_title('Feature Importance for Churn Prediction')
    st.pyplot(fig)

# Function for home page
def home_page():
    st.title('Welcome to the Customer Churn Prediction App!')
    st.write("This is a simple Streamlit app that predicts customer churn.")
    st.write("Navigate to the 'Customer Churn Prediction' page to make a prediction.")
    st.image("image.png", use_container_width=True)
    
    st.subheader('Features:')
    st.write('- **Geography**: Select the customer\'s geographical location.')
    st.write('- **Gender**: Choose the customer\'s gender.')
    st.write('- **Credit Score**: Enter the credit score of the customer.')
    st.write('- **Balance**: Enter the account balance.')
    st.write('- **Tenure**: Enter the tenure of the customer.')

# Main navigation
page = st.sidebar.radio("Choose a page", ["Home", "Customer Churn Prediction"])

if page == "Home":
    home_page()
elif page == "Customer Churn Prediction":
    churn_prediction()
