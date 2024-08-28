import streamlit as st
import pandas as pd
import pickle
import numpy as np

#Load and preprocess the dataset
def load_data():
    df = pd.read_csv(
        r"C:\Users\DELL\Desktop\car_details\structured_data\updated_cars_features.csv"
    )

    # Drop unnecessary columns
    df = df.drop([
        'priceSaving', 'priceFixedText', 'trendingText.imgUrl', 'trendingText.heading',
        'trendingText.desc', 'car_links', 'Seats.1', 'Ignition type', 'owner', 'Feature1',
        'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8',
        'Feature9', 'Engine Displacement', 'variantName', 'priceActual', 'Kilometers',
        'model', 'ownerNo', 'centralVariantId', 'Registration Year', 'Fuel Type', 'RTO',
        'Engine Displacement', 'Transmission', 'Year of Manufacture', 'Torque',
        'Head Light', 'Rear View Mirror', 'Air conditioner', 'Power Steering', 'Heater',
        'Child Lock', 'Central Locking', 'Cd Player&Radio', 'Anti Lock Braking System',
        'Fog Lights Front'
    ], axis=1)

    # Convert and clean the 'price' column
    df = df.dropna(subset=['price'])

    #Convert price string to numeric value
    def convert_price(price):
        price = price.replace('â‚¹', '').replace('₹', '').replace(',', '').strip()
        if 'Lakh' in price:
            price = price.replace('Lakh', '').strip()
            return float(price) * 100000
        elif 'Crore' in price:
            price = price.replace('Crore', '').strip()
            return float(price) * 10000000
        return float(price)

    df['price'] = df['price'].apply(convert_price)

    # Clean and preprocess other columns
    # Extract numeric values from the 'Mileage' column and convert them to float
    df['Mileage'] = df['Mileage'].str.extract(r'(\d+\.?\d*)').astype(float)
    # Fill missing values in the 'Mileage' column with the mean mileage
    df['Mileage'] = df['Mileage'].fillna(df['Mileage'].mean())
    # Extract numeric values from the 'Max Power' column and convert them to float
    df['Max Power'] = df['Max Power'].str.extract(r'(\d+\.?\d*)')[0].astype(float).fillna(
        # Fill missing values with the mode (most frequent value) of 'Max Power'
        df['Max Power'].str.extract(r'(\d+\.?\d*)')[0].mode()[0]
    )
    # Clean and convert 'Kms Driven' column to integer
    df['Kms Driven'] = df['Kms Driven'].str.replace(',', '').str.replace('Kms', '').fillna(0).astype(int)
    # Clean and convert 'Engine' column to integer
    df['Engine'] = df['Engine'].str.replace(',', '').str.replace('CC', '').fillna(0).astype(int)
    # Remove 'R' from 'Wheel Size' column and fill missing values with the mode
    df['Wheel Size'] = df['Wheel Size'].str.replace('R', '').fillna(df['Wheel Size'].mode()[0])
    # Clean and convert 'Seats' column to integer
    df['Seats'] = df['Seats'].str.replace('Seats', '').str.strip()
    df['Seats'] = df['Seats'].fillna(df['Seats'].mode()[0]).astype(int)
    # Fill missing values in 'Ownership' column with the mode
    df['Ownership'] = df['Ownership'].fillna(df['Ownership'].mode()[0])

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=[
        'Fuel type', 'Body type', 'transmission', 'oem', 'Insurance Validity', 'City',
        'Power Window Front', 'Ownership'
    ], dtype='int')

    return df

 #Load the trained model
def load_model():
    with open('random_forest_best_model.pkl', mode='rb+') as file:
        model = pickle.load(file)
    return model

#Preprocess input data and align it with the model's expected features
def preprocess_input_data(input_data, feature_columns):
    input_df = pd.DataFrame([input_data])

    # One-hot encode categorical variables
    input_df = pd.get_dummies(input_df, columns=[
        'Fuel type', 'Body type', 'transmission', 'oem', 'Insurance Validity', 'City', 'Ownership'
    ], dtype='int')

    # Handle missing columns in the new data
    missing_cols = set(feature_columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[feature_columns]

    return input_df

#Format price with Lakhs and Crores
def format_price(price):
    if price >= 10000000:  # 1 Crore
        price_in_crores = price / 10000000
        return f'₹{price_in_crores:,.2f} Crores'
    else:
        price_in_lakhs = price / 100000
        return f'₹{price_in_lakhs:,.2f} Lakhs'

# Main function to run the Streamlit app
def main():
    st.title('Car Dheko Used Car Price Prediction')
    st.write("This app predicts the price of a used car based on the input features.")

    # Add and center the image
    st.image(r'C:\Projects\project_3\Cardekho.jpg', caption='Car Dheko Logo', use_column_width=True)

    # Load data and model
    df = load_data()
    model = load_model()
    feature_columns = df.columns[df.columns != 'price']  # Features used by the model

    # Input fields for user
    st.sidebar.header('Car Details')
    Mileage = st.sidebar.number_input('Mileage (kmpl)', min_value=0.0)
    Max_power = st.sidebar.number_input('Max Power (bhp)', min_value=0.0)
    Kms_Driven = st.sidebar.number_input('Kms Driven', min_value=0)
    Engine = st.sidebar.number_input('Engine (CC)', min_value=0)
    Wheel_Size = st.sidebar.number_input('Wheel Size', min_value=0)
    Seats = st.sidebar.number_input('Seats', min_value=1)
    Ownership = st.sidebar.selectbox('Ownership', options=[
        'First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'
    ])
    Fuel_type = st.sidebar.selectbox('Fuel Type', options=['Petrol', 'Diesel', 'CNG', 'LPG'])
    Body_type = st.sidebar.selectbox('Body Type', options=[
        'Hatchback', 'Sedan', 'SUV', 'MUV', 'Convertible', 'Wagon'
    ])
    transmission = st.sidebar.selectbox('Transmission', options=['Manual', 'Automatic'])
    oem = st.sidebar.selectbox('OEM', options=[
        'Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'BMW', 'Mercedes', 'Audi'
    ])
    Insurance_Validity = st.sidebar.selectbox('Insurance Validity', options=['Yes', 'No'])
    car_age=st.sidebar.number_input('car age',min_value=0)
    City = st.sidebar.selectbox('City Name', options=['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Jaipur'])

    # Input data for prediction
    input_data = {
        'Mileage': Mileage,
        'Max Power': Max_power,
        'Kms Driven': Kms_Driven,
        'Engine': Engine,
        'Wheel Size': Wheel_Size,
        'Seats': Seats,
        'Ownership': Ownership,
        'Fuel type': Fuel_type,
        'Body type': Body_type,
        'transmission': transmission,
        'oem': oem,
        'Insurance Validity': Insurance_Validity,
        'City': City,
        'car age':car_age
    }

    # Preprocess input data
    input_df = preprocess_input_data(input_data, feature_columns)

    # Prediction
    if st.button('Predict Price'):
        prediction = model.predict(input_df)
        formatted_price = format_price(prediction[0])
        st.write(f'Predicted Car Price: {formatted_price}')

if __name__ == "__main__":
    main()
