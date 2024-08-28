import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
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
    df['Seats'] = df['Seats'].fillna(df['Seats'].mode()[0])
    df['Seats'] = df['Seats'].astype(int)

    # Fill missing values in 'Ownership' column with the mode
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

    # Define hyperparameter grids for GridSearchCV
    param_grids = {
        'Linear Regression': {},
        'Lasso': {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]},
        'Ridge': {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]},
        'Decision Tree': {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'K-Neighbors Regressor': {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance']
        }
    }

    best_models = {}

    # Perform hyperparameter tuning and evaluation
    for name, model in models.items():
        if param_grids[name]:
            grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='r2', n_jobs=-1, verbose=2)
            grid_search.fit(x_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"--- {name} ---")
            print(f"Best Parameters: {grid_search.best_params_}")

            # Evaluate on test data
            test_pred = best_model.predict(x_test)
            print(f"MSE Test: {mean_squared_error(y_test, test_pred)}")
            print(f"MAE Test: {mean_absolute_error(y_test, test_pred)}")
            print(f"R2 Test: {r2_score(y_test, test_pred)}\n")

            best_models[name] = best_model
        else:
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
            if name == 'Random Forest':  
                with open('random_forest.pkl', mode='wb+') as file:
                    pickle.dump(model, file)

    # Save the best model from each type
    for name, model in best_models.items():
        with open(f'{name.replace(" ", "_").lower()}_best_model.pkl', mode='wb+') as file:
            pickle.dump(model, file)
        print(f"Best model saved: {name}")

def main():
    # File path to the dataset
    file_path = r"C:\Users\DELL\Desktop\car_details\structured_data\updated_cars_features.csv"

    # Clean and preprocess the data
    X, y = clean_and_preprocess_data(file_path)

    # Define the models
    global models
    models = {
        'Linear Regression': LinearRegression(),
        'Lasso': Lasso(),
        'Ridge': Ridge(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'K-Neighbors Regressor': KNeighborsRegressor()
    }

    # Train and evaluate the models
    train_and_evaluate_models(X, y)

if __name__ == "__main__":
    main()
