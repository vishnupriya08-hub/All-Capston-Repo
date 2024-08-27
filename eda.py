import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

def process_car_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Drop the 'Torque' column if it exists
    if 'Torque' in df.columns:
        df.drop(['Torque'], axis=1, inplace=True)

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=[
        'Fuel Type', 'Body type', 'transmission', 'Insurance Validity', 'oem',
        'Power Window Front', 'City', 'Head Light', 'Rear View Mirror', 'Air conditioner',
        'Power Steering', 'Heater', 'Child Lock', 'Power Window Front',
        'Central Locking', 'Cd Player&Radio', 'Anti Lock Braking System', 'Fog Lights Front'
    ], dtype='int')

    # Label encode the 'RTO' column
    label_encoder = LabelEncoder()
    df['RTO'] = label_encoder.fit_transform(df['RTO'])

    # Display box plots for various features
    features_to_plot = ['price', 'modelYear', 'Kms Driven', 'Year of Manufacture', 
                        'Mileage', 'Engine']

    # Include 'Torque' in the plots if it exists
    if 'Torque' in df.columns:
        features_to_plot.append('Torque')

    for feature in features_to_plot:
        sns.boxplot(df[feature])
        plt.show()

    # Normalize and filter outliers for each feature
    def normalize_and_filter(df, column):
        df[column] = (df[column] - df[column].mean()) / df[column].std()
        return df[(df[column] > -2) & (df[column] < 2)]

    df = normalize_and_filter(df, 'price')
    sns.boxplot(df['price'])
    plt.show()

    df = normalize_and_filter(df, 'modelYear')
    sns.boxplot(df['modelYear'])
    plt.show()

    df = normalize_and_filter(df, 'Kms Driven')
    sns.boxplot(df['Kms Driven'])
    plt.show()

    df = normalize_and_filter(df, 'Year of Manufacture')
    sns.boxplot(df['Year of Manufacture'])
    plt.show()

    df = normalize_and_filter(df, 'Mileage')
    sns.boxplot(df['Mileage'])
    plt.show()

    df = normalize_and_filter(df, 'Engine')
    sns.boxplot(df['Engine'])
    plt.show()

    if 'Torque' in df.columns:
        df = normalize_and_filter(df, 'Torque')
        sns.boxplot(df['Torque'])
        plt.show()

    # Log-transform the 'price' column and plot the histogram
    skew = np.log(df['price'])
    sns.histplot(skew)
    plt.show()

    # Calculate kurtosis and determine the type of kurtosis
    def get_kurtosis_type(value):
        if value > 3:
            return "Leptokurtic"
        elif value < 3:
            return "Platykurtic"
        else:
            return "Mesokurtic"

    data = df['price']
    kurtosis_value = stats.kurtosis(data)
    kurtosis_type = get_kurtosis_type(kurtosis_value)

    print(f"Kurtosis Type: {kurtosis_type}")
    print(f"Kurtosis Value: {kurtosis_value}")

    # Display the correlation heatmap
    plt.figure(figsize=(40, 40))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True)
    plt.show()

    # Return the correlation of features with the 'price' column
    price_correlation = correlation_matrix['price'].sort_values(ascending=False)
    return price_correlation

file_path = r"C:\Users\DELL\Desktop\car_details\structured_data\preprocessed_data.csv"
price_corr = process_car_data(file_path)
print(price_corr)
