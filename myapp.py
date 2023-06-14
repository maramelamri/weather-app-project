# %%
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_numeric_dtype
import sqlite3
# %%
# API URL for weather data
api_url = "https://archive-api.open-meteo.com/v1/archive?latitude=36.81&longitude=10.18&start_date=1998-01-01&end_date=2023-05-24&daily=weathercode,temperature_2m_max,temperature_2m_min,temperature_2m_mean,apparent_temperature_max,apparent_temperature_min,apparent_temperature_mean,sunrise,sunset,shortwave_radiation_sum,precipitation_sum,rain_sum,snowfall_sum,precipitation_hours,windspeed_10m_max,windgusts_10m_max,winddirection_10m_dominant,et0_fao_evapotranspiration&timezone=auto"

# Make the API request
response = requests.get(api_url)

# Extract the JSON response
data = response.json()

# Extract the daily weather data from the API response
daily_data = data['daily']

# Create an empty DataFrame
df = pd.DataFrame(daily_data)
# %%
#data exploration
# Print the column names of the DataFrame
print(df.columns)

#check for missing values 
missing_values = df.isnull().sum()
print(missing_values)

df.head()

# Convert the 'time' column to datetime format
df['time'] = pd.to_datetime(df['time'])
# Set the 'time' column as the index
df.set_index('time', inplace=True)
# %%
#data viz
# Plot the time series of temperature
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['temperature_2m_mean'])
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Time Series')
plt.grid(True)
plt.show()

# Visualize the distribution of precipitation
plt.figure(figsize=(8, 6))
sns.histplot(df['precipitation_sum'], bins=20, kde=True)
plt.xlabel('Precipitation (mm)')
plt.ylabel('Frequency')
plt.title('Distribution of Precipitation')
plt.grid(True)
plt.show()

# Calculate summary statistics
summary_stats = df.describe()
print(summary_stats)

# Calculate the correlation matrix
corr_matrix = df.corr()

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()
# %%
# train the XBoost model using the preprocessed data
# Preprocess the DataFrame
df_preprocessed = df.copy()

# Convert sunrise and sunset columns to datetime
df_preprocessed['sunrise'] = pd.to_datetime(df_preprocessed['sunrise'])
df_preprocessed['sunset'] = pd.to_datetime(df_preprocessed['sunset'])

# Convert numeric columns to float
for col in df_preprocessed.columns:
    if is_numeric_dtype(df_preprocessed[col]):
        df_preprocessed[col] = df_preprocessed[col].astype(float)



# Define the features (X) and target variables (y) based on the DataFrame 'df'
X = df[['weathercode', 'temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_max', 'apparent_temperature_min', 'apparent_temperature_mean', 'sunrise', 'sunset', 'shortwave_radiation_sum', 'precipitation_sum', 'rain_sum', 'snowfall_sum', 'precipitation_hours', 'windspeed_10m_max', 'windgusts_10m_max', 'winddirection_10m_dominant', 'et0_fao_evapotranspiration']]

# Remove non-numeric columns from y
y_numeric = df_preprocessed[['temperature_2m_mean', 'shortwave_radiation_sum', 'precipitation_sum', 'rain_sum', 'snowfall_sum', 'precipitation_hours', 'windspeed_10m_max', 'et0_fao_evapotranspiration']]

# Standardize the numeric features
numeric_cols = ['weathercode', 'temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_max', 'apparent_temperature_min', 'apparent_temperature_mean', 'shortwave_radiation_sum', 'precipitation_sum', 'rain_sum', 'snowfall_sum', 'precipitation_hours', 'windspeed_10m_max', 'windgusts_10m_max', 'winddirection_10m_dominant', 'et0_fao_evapotranspiration']
X_numeric = df_preprocessed[numeric_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Train the XGBoost model using cross-validation
xgb_model = xgb.XGBRegressor()
cv_scores = cross_val_score(xgb_model, X_scaled, y_numeric, cv=5, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)
# Print the cross-validation RMSE scores
print('Cross-Validation RMSE:', cv_rmse_scores)




