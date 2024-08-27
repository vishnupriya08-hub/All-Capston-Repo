import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle


def clean_and_preprocess_data(file_path):
    
    # Load the dataset
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    columns_to_drop = [
        'priceSaving', 'priceFixedText', 'trendingText.imgUrl', 'trendingText.heading',
        'trendingText.desc', 'car_links', 'Seats.1', 'Ignition type', 'owner', 'Feature1',
        'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8',
        'Feature9', 'Engine Displacement', 'variantName', 'priceActual', 'Kilometers',
        'model', 'ownerNo', 'centralVariantId', 'Registration Year', 'Fuel Type', 'RTO',
        'Engine Displacement', 'Transmission', 'Year of Manufacture', 'Torque',
        'Head Light', 'Rear View Mirror', 'Air conditioner', 'Power Steering', 'Heater',
        'Child Lock', 'Central Locking', 'Cd Player&Radio', 'Anti Lock Braking System',
        'Fog Lights Front'
    ]
    df = df.drop(columns=columns_to_drop)

    # Convert and clean the 'price' column
    df = df.dropna(subset=['price'])

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
    df['Mileage'] = df['Mileage'].str.extract(r'(\d+\.?\d*)').astype(float)
    df['Mileage'] = df['Mileage'].fillna(df['Mileage'].mean())

    df['Max Power'] = df['Max Power'].str.extract(r'(\d+\.?\d*)')[0].astype(float).fillna(
        df['Max Power'].str.extract(r'(\d+\.?\d*)')[0].mode()[0]
    )

    df['Kms Driven'] = df['Kms Driven'].str.replace(',', '').str.replace('Kms', '').fillna(0).astype(int)

    df['Engine'] = df['Engine'].str.replace(',', '').str.replace('CC', '').fillna(0).astype(int)

    df['Wheel Size'] = df['Wheel Size'].str.replace('R', '').fillna(df['Wheel Size'].mode()[0])

    df['Seats'] = df['Seats'].str.replace('Seats', '').str.strip()
    df['Seats'] = df['Seats'].fillna(df['Seats'].mode()[0])
    df['Seats'] = df['Seats'].astype(int)

    df['Ownership'] = df['Ownership'].fillna(df['Ownership'].mode()[0])

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=[
        'Fuel type', 'Body type', 'transmission', 'oem',
        'Insurance Validity', 'City', 'Power Window Front', 'Ownership'
    ], dtype='int')

    X = df.drop(['price'], axis=1)
    y = df['price']

    return X, y


def train_and_evaluate_models(X, y):
    
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the models
    models = {
        'Linear Regression': LinearRegression(),
        'Lasso': Lasso(alpha=1.5),
        'Ridge': Ridge(alpha=1.5),
        'Decision Tree': DecisionTreeRegressor(max_depth=4, min_samples_split=10, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'K-Neighbors Regressor': KNeighborsRegressor(n_neighbors=5)
    }

    # Train and evaluate the models
    for name, model in models.items():
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        print(f"--- {name} ---")
        print(f"MSE Train: {mean_squared_error(y_train, train_pred)}")
        print(f"MSE Test: {mean_squared_error(y_test, test_pred)}")
        print(f"MAE Train: {mean_absolute_error(y_train, train_pred)}")
        print(f"MAE Test: {mean_absolute_error(y_test, test_pred)}")
        print(f"R2 Train: {r2_score(y_train, train_pred)}")
        print(f"R2 Test: {r2_score(y_test, test_pred)}\n")

        # Save the final model
        with open('random_forest.pkl', mode='wb+') as file:
            pickle.dump(model, file)


def main():
    # File path to the dataset
    file_path = r"C:\Users\DELL\Desktop\car_details\structured_data\updated_cars_features.csv"

    # Clean and preprocess the data
    X, y = clean_and_preprocess_data(file_path)

    # Train and evaluate the models
    train_and_evaluate_models(X, y)


if __name__ == "__main__":
    main()
