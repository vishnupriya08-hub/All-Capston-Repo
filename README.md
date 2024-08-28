Car Dheko - Used Car Price Prediction

Project Overview

The Car Dheko - Used Car Price Prediction project is aimed at predicting the price of used cars based on various features such as mileage, engine size, fuel type, and more. This project utilizes machine learning models to provide accurate price predictions, which can be deployed in a user-friendly Streamlit application.

Skills Takeaway

Throughout this project, the following skills were developed and applied:

•	Data Cleaning and Preprocessing: Handling missing values, converting categorical data, and preparing the dataset for modeling.

•	Exploratory Data Analysis (EDA): Gaining insights from the data using statistical analysis and visualizations.

•	Machine Learning Model Development: Implementing various regression models including Linear Regression, Lasso, Ridge, Decision Tree, Random Forest, Gradient Boosting, and K-Neighbors Regressor.

•	Price Prediction Techniques: Leveraging machine learning algorithms to predict the price of used cars with high accuracy.

•	Model Evaluation and Optimization: Evaluating model performance using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared. Fine-tuning model parameters for better accuracy.

•	Model Deployment: Saving the trained model using pickle and deploying it in a Streamlit application for real-time predictions.

•	Streamlit Application Development: Building an interactive web application where users can input car details to get an estimated price prediction.

•	Documentation and Reporting: Maintaining thorough documentation for the entire project and generating reports to communicate findings and model performance.

Domain

•	Automotive Industry

•	Data Science

•	Machine Learning

Project Structure

The repository contains the following files and directories:

•	data/: Contains the dataset used for training and testing the models.

•	VS Code/: VS Code with detailed steps on data cleaning, preprocessing, EDA, and model training.

•	models/: Contains the saved models in pickle format.

•	app: Streamlit application files for deploying the model.

•	requirements.txt: List of all dependencies required to run the project.

Dataset

The dataset used in this project consists of various features related to used cars, including:

•	Mileage (kmpl)

•	Max Power (bhp)

•	Kms Driven

•	Engine (CC)

•	Wheel Size

•	Seats

•	Ownership

•	Fuel type

•	Body type

•	Transmission

•	OEM (Original Equipment Manufacturer)

•	Insurance Validity

•	City

Model Deployment

The project includes a Streamlit application where users can input car details and get an estimated price prediction. The app is built using the following steps:

1.	Load the trained model: The model is loaded from a pickle file.
   
2.	Preprocess the input data: The user input is preprocessed to match the format used during training.
   
3.	Predict the price: The model predicts the price based on the input data.
   
4.	Display the result: The predicted price is displayed in a user-friendly format (e.g., in Lakhs or Crores).
   
Results

The models were evaluated on their ability to predict the price of used cars. Below are the key metrics:

•	Mean Squared Error (MSE): Indicates the average of the squares of the errors.

•	Mean Absolute Error (MAE): Indicates the average of the absolute differences between predicted and actual values.

•	R-squared (R2): Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
The best-performing model was used in the final deployment, ensuring that users get the most accurate predictions.

Acknowledgments

•	The dataset used in this project is sourced from Car Dheko.

•	The project is inspired by real-world challenges in the automotive industry.

