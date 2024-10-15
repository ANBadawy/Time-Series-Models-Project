# Time Series Forecasting with Random Forest and XGBoost

This project provides a framework for training and saving machine learning models to forecast time series data, utilizing both **Random Forest** and **XGBoost** regressors. The project includes custom feature engineering, data preprocessing, and an ensemble approach for more accurate predictions. It is designed to handle multiple datasets simultaneously and saves trained models for future use, making it ideal for scalable time series prediction tasks.

## Project Overview

The primary goal of this project is to build predictive models for time series data using a combination of tree-based machine learning algorithms. By leveraging **Random Forest** and **XGBoost**, the project aims to capture both simple and complex patterns in time series data, which are then aggregated for enhanced prediction accuracy.

Key components include:
- **Feature Engineering**: Extracts temporal features (e.g., hour, day of the week, month) and lagged values to capture trends and seasonality.
- **Standardization**: Standardizes features to improve model training, especially for models sensitive to feature scales, like XGBoost.
- **Model Training and Saving**: Trains models for each dataset, then saves them as `.joblib` files, making them readily available for deployment.
- **Prediction API**: Includes a prediction function that loads the pre-trained models, applies feature engineering and standardization, and provides real-time predictions based on new input data.

## Features

1. **Data Loading and Preprocessing**:
   - Loads datasets based on unique dataset IDs, allowing you to handle multiple time series datasets with ease.
   - Cleans and processes data, filling missing values and handling anomalies.

2. **Custom Feature Engineering**:
   - Adds features like `minute`, `hour`, `dayofweek`, `month`, and `quarter` to capture time-based patterns.
   - Creates lagged features (e.g., `lag1`, `lag2`, `lag3`, `lag4`, `lag5`) and rolling statistics (mean and standard deviation) for enhanced context on historical values.

3. **Standardization**:
   - Applies standardization using `StandardScaler` from `sklearn` to ensure that all features contribute equally to the model, improving model performance and training stability.

4. **Model Training**:
   - Trains both **Random Forest** and **XGBoost** regressors on the standardized features, capturing a diverse set of patterns in the data.
   - Uses a weighted ensemble approach to combine predictions from both models, improving overall forecasting accuracy.

5. **Model Saving and Loading**:
   - Saves trained models as `.joblib` files with unique names based on the dataset ID and model type.
   - Supports dynamic model loading for quick access during prediction, making the system scalable for various applications.

6. **Prediction Function**:
   - Includes a prediction function that takes in new data, applies the same feature engineering and standardization, and provides predictions using the trained ensemble model.
   - Suitable for integration into a RESTful API, allowing for real-time prediction requests.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/time-series-forecasting.git
   cd time-series-forecasting
