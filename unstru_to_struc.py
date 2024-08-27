import pandas as pd
import ast
import os

def process_city_data(city_name, input_file, output_file):
    
    try:
        # Load the Excel file for the specified city
        df1 = pd.read_excel(input_file)
        
        # Process the 'new_car_detail' column to convert string representations of dictionaries into actual dictionaries
        df1['new_car_detail'] = df1['new_car_detail'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # Normalize the JSON data in the 'new_car_detail' column to create individual columns for each detail
        car_details_df = pd.json_normalize(df1['new_car_detail'])
        
        # Drop the columns that are not needed after extracting their details
        df1 = df1.drop(columns=['new_car_detail', 'new_car_feature', 'new_car_overview', 'new_car_specs', 'car_links'])
        
        # Concatenate the normalized car details back into the main DataFrame
        df1 = pd.concat([df1, car_details_df], axis=1)
        
        # Rename columns to more descriptive names
        df1.rename(columns={'it': 'Ignition type', 'ft': 'Fuel type', 'bt': 'Body type', 'km': 'Kilometers'}, inplace=True)
        
        # Process the second column of data (B column in Excel)
        df2 = pd.read_excel(input_file, usecols='B')
        processed_data_2 = []
        
        # Iterate through each row in the second column
        for index, row in df2.iterrows():
            cell_data = row.iloc[0]
            
            # Convert string representations of dictionaries into actual dictionaries
            if isinstance(cell_data, str):
                try:
                    cell_data = ast.literal_eval(cell_data)
                except ValueError:
                    continue
            
            # Extract relevant data if it is a dictionary and contains a 'top' key
            if isinstance(cell_data, dict) and 'top' in cell_data:
                flat_data = {item['key']: item['value'] for item in cell_data['top']}
                processed_data_2.append(flat_data)
        
        # Convert the processed data into a DataFrame
        final_df2 = pd.DataFrame(processed_data_2)
        
        # Process the third column of data (C column in Excel)
        df3 = pd.read_excel(input_file, usecols='C')
        processed_data_3 = []
        
        # Iterate through each row in the third column
        for index, row in df3.iterrows():
            cell_data = row.iloc[0]
            
            # Convert string representations of dictionaries into actual dictionaries
            if isinstance(cell_data, str):
                try:
                    cell_data = ast.literal_eval(cell_data)
                except ValueError:
                    continue
            
            # Extract relevant data if it is a dictionary and contains a 'top' key
            if isinstance(cell_data, dict) and 'top' in cell_data:
                flat_data = {item['value'] for item in cell_data['top']}
                processed_data_3.append(flat_data)
        
        # Convert the processed data into a DataFrame
        final_df3 = pd.DataFrame(processed_data_3)
        
        # Assign names to the columns of the third DataFrame
        final_df3.columns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9']
        
        # Process the fourth column of data (D column in Excel)
        df4 = pd.read_excel(input_file, usecols='D')
        processed_data_4 = []
        
        # Iterate through each row in the fourth column
        for index, row in df4.iterrows():
            cell_data = row.iloc[0]
            
            # Convert string representations of dictionaries into actual dictionaries
            if isinstance(cell_data, str):
                try:
                    cell_data = ast.literal_eval(cell_data)
                except ValueError:
                    continue
            
            # Extract relevant data if it is a dictionary and contains a 'top' key
            if isinstance(cell_data, dict) and 'top' in cell_data:
                flat_data = {item['key']: item['value'] for item in cell_data['top']}
                processed_data_4.append(flat_data)
        
        # Convert the processed data into a DataFrame
        final_df4 = pd.DataFrame(processed_data_4)
        
        # Process the fifth column of data (E column in Excel)
        df5 = pd.read_excel(input_file, usecols='E')
        
        # Concatenate all the processed DataFrames column-wise to form the final structured DataFrame
        city_df = pd.concat([df1, final_df2, final_df3, final_df4, df5], axis=1)
        
        # Add a new column to the DataFrame indicating the city name
        city_df['City'] = city_name
        
        # Save the final DataFrame to a CSV file
        city_df.to_csv(output_file, index=False)
        
        # Display a preview of the processed DataFrame for verification
        print(f"Processed data for {city_name}:")
        print(city_df.head())
        
        return city_df
    
    except Exception as e:
        # Catch any exceptions that occur during the processing and print an error message
        print(f"Error processing data for {city_name}: {e}")
        return None

# Define the input file paths and corresponding cities
cities = {
    'Bangalore': r"C:\Users\DELL\Desktop\car_details\Dataset\bangalore_cars.xlsx",
    'Chennai': r"C:\Users\DELL\Desktop\car_details\Dataset\chennai_cars.xlsx",
    'Delhi': r"C:\Users\DELL\Desktop\car_details\Dataset\delhi_cars.xlsx",
    'Hyderabad': r"C:\Users\DELL\Desktop\car_details\Dataset\hyderabad_cars.xlsx",
    'Jaipur': r"C:\Users\DELL\Desktop\car_details\Dataset\jaipur_cars.xlsx",
    'Kolkata': r"C:\Users\DELL\Desktop\car_details\Dataset\kolkata_cars.xlsx"  
}

# Define the directory where the output CSV files will be saved
output_dir = r"C:\Users\DELL\Desktop\car_details\structured_data"

# Process the data for each city by iterating through the defined cities and file paths
for city, input_file in cities.items():
    # Generate the output file path for each city
    output_file = os.path.join(output_dir, f"structured_data_{city.lower()}.csv")
    
    # Call the function to process the city's data and save it to a CSV file
    process_city_data(city, input_file, output_file)
