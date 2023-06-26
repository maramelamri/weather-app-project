import requests
import random
import json
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_numeric_dtype
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score

#Data retrieval
# API URL for weather data
api_url = "https://archive-api.open-meteo.com/v1/archive?latitude=36.82&longitude=10.17&start_date=2018-06-16&end_date=2023-06-16&daily=weathercode,temperature_2m_max,temperature_2m_min,temperature_2m_mean,sunrise,sunset,shortwave_radiation_sum,precipitation_sum,rain_sum,snowfall_sum,precipitation_hours,windspeed_10m_max,et0_fao_evapotranspiration&timezone=auto&min=&max="

# Make the API request
response = requests.get(api_url)

# Extract the JSON response
data = response.json()

# Extract the daily weather data from the API response
daily_data = data['daily']

# Create an empty DataFrame
df = pd.DataFrame(daily_data)
df = df.drop(['snowfall_sum'], axis=1)
# Convert date/time columns to datetime format
date_columns = ['time', 'sunrise', 'sunset']
df[date_columns] = df[date_columns].apply(pd.to_datetime)

# Handle missing values
df = df.fillna(method='ffill')

# Convert numeric columns to appropriate data types
numeric_columns = ['weathercode', 'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean', 'shortwave_radiation_sum',
                   'precipitation_sum', 'rain_sum', 'precipitation_hours', 'windspeed_10m_max',
                   'et0_fao_evapotranspiration']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

# Print the preprocessed DataFrame
print(df.head())

#Correlation matrix
# Calculate the correlation matrix
corr_matrix = df.corr()

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()


#Plotting temperature_2m_mean and shortwave_radiation_sum
# Set up the figure and axes
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

# Plot temperature_2m_mean
sns.lineplot(data=df, x='time', y='temperature_2m_mean', ax=axes[0])
axes[0].set_title('Temperature 2m Mean')

# Plot shortwave_radiation_sum
sns.lineplot(data=df, x='time', y='shortwave_radiation_sum', ax=axes[1])
axes[1].set_title('Shortwave Radiation Sum')

# Rotate x-axis labels for better visibility
axes[0].tick_params(axis='x', rotation=45)
axes[1].tick_params(axis='x', rotation=45)

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.4)

# Show the plot
plt.show()


# Calculate summary statistics
summary_stats = df.describe()
print(summary_stats)

#Train/Test split
from sklearn.model_selection import train_test_split

# Split the data into X and y
# X = df[['time']]
df['time'] = pd.to_datetime(df['time'])

# Extract relevant features
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day

X = df[['year','month','day']]
# Drop the original 'time' column
df = df.drop('time', axis=1)
df.to_csv('df.csv',index=False)
Y = df[['temperature_2m_mean', 'shortwave_radiation_sum', 'precipitation_sum', 'rain_sum', 'windspeed_10m_max', 'et0_fao_evapotranspiration']]

# Split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


## Model training and eval
# Initialize the XGBoost model
model = xgb.XGBRegressor()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
# Save model
model.save_model('xgboost_model.model')

