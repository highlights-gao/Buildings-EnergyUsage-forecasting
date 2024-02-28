# Energy Usage Forecasting Project

## Introduction
The goal of this project is to forecast the energy usage of a business building for air conditioning, lighting, and sockets. We aim to compare the performance of various baseline algorithms for time series forecasting using historical energy usage data spanning two years and weather data. Specifically, we will evaluate the effectiveness of the following algorithms: DeepAR, XGBoost, LightGBM Regression, and Recurrent Neural Networks (RNN) with Long Short-Term Memory (LSTM) architecture. The objective is to predict energy usage for the upcoming week based on historical patterns and weather conditions.

## Data Description
### Energy Usage Data
- **Time Period:** Two years
- **Granularity:** Hourly
- **Components:** 
  - Air conditioning
  - Lighting
  - Sockets

### Weather Data
- **Source:** Weather API or historical weather data repository
- **Parameters:** 
  - Temperature
  - Humidity
  - Precipitation
  - Wind Speed
  - Other relevant weather variables

## Methodology
1. **Data Preprocessing:**
   - Clean and preprocess the energy usage data.
   - Merge weather data with energy usage data based on timestamp.
   - Handle missing values and outliers.
   - Normalize or scale the data if necessary.

2. **Model Selection:**
   - Implement baseline algorithms: DeepAR, XGBoost, LightGBM Regression, and RNN (LSTM).
   - Train each model on the preprocessed data.
   - Tune hyperparameters using techniques such as grid search or random search.

3. **Model Evaluation:**
   - Evaluate the performance of each model using appropriate metrics (e.g., RMSE, MAE, MAPE).
   - Compare the accuracy and computational efficiency of the models.
   - Analyze any insights gained from the evaluation process.

4. **Forecasting:**
   - Select the best-performing model based on evaluation results.
   - Generate energy usage forecasts for the upcoming week.
   - Assess the uncertainty of the forecasts and provide confidence intervals if applicable.

## Conclusion
This project aims to provide accurate forecasts of energy usage for a business building by leveraging historical data and weather conditions. By comparing the performance of various baseline algorithms, we aim to identify the most effective approach for forecasting energy consumption. The insights gained from this project can help optimize energy management strategies and improve overall efficiency in commercial buildings.

