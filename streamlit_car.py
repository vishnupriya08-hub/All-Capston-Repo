import streamlit as st
import pandas as pd
st.title('Car Dheko Used Car Price Prediction')
st.write("This app predicts the price of a used car based on the input features.")

# Define the user input fields
st.sidebar.header('Car Details:Cardheko')

Fuel_type= st.sidebar.selectbox('Fuel type',('Petrol','Diesel','CNG'))
Kms_Driven = st.sidebar.number_input('Kilometers Driven', value=50000)
transmission = st.sidebar.selectbox('Transmission Type', ('Manual', 'Automatic'))
Ownership = st.sidebar.selectbox('Ownership Type', ('First Owner', 'Second Owner', 'Third Owner'))
Body_type=st.sidebar.selectbox('Body type',('Hatchback','SUV','Sedan','MUV'))
    

