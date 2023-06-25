# %%
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
import sqlite3
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import emoji




# %% [markdown]
# #Data retrieval

# %%
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


# %% [markdown]
# #Correlation matrix

# %%
# Calculate the correlation matrix
corr_matrix = df.corr()

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()


# %% [markdown]
# -weathercode has a negative correlation with several variables such as temperature_2m_max, apparent_temperature_max, apparent_temperature_mean, and shortwave_radiation_sum. This suggests that as the weathercode increases, these variables tend to decrease.
# 
# -temperature_2m_max, temperature_2m_min, temperature_2m_mean, apparent_temperature_max, apparent_temperature_min, and apparent_temperature_mean have strong positive correlations among themselves. This indicates that these temperature-related variables are highly correlated and tend to move in the same direction.
# 
# -precipitation_sum, rain_sum, and precipitation_hours have positive correlations with weathercode, indicating that as the weather conditions change, these precipitation-related variables tend to increase.
# 
# -shortwave_radiation_sum shows a moderate positive correlation with et0_fao_evapotranspiration, suggesting that higher shortwave radiation is associated with higher evapotranspiration.
# 
# -windspeed_10m_max and windgusts_10m_max have strong positive correlations, indicating that they are closely related and tend to move together.
# 

# %% [markdown]
# #Plotting temperature_2m_mean and shortwave_radiation_sum

# %%
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


# %%
# Calculate summary statistics
# summary_stats = df.describe()
# print(summary_stats)

# %% [markdown]
# #Train/Test split

# %%
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
Y = df[['temperature_2m_mean', 'shortwave_radiation_sum', 'precipitation_sum', 'rain_sum', 'windspeed_10m_max', 'et0_fao_evapotranspiration']]

# Split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# %% [markdown]
# #Train and eval

# %% [markdown]
# ## Model training and eval
# 
# ---
# 
# 

# %%
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


# %% [markdown]
# #Store the weather data in an SQLite database for future use



# Write the data to a sqlite table
# Write the data to a SQLite table
conn = sqlite3.connect('weather_data.db')
c = conn.cursor()

# Create table if not exists
c.execute('''
    CREATE TABLE IF NOT EXISTS weather(
        time TEXT,
        weathercode REAL,
        temperature_2m_max REAL,
        temperature_2m_min REAL,
        temperature_2m_mean REAL,
        sunrise TEXT,
        sunset TEXT,
        shortwave_radiation_sum REAL,
        precipitation_sum REAL,
        rain_sum REAL,
        precipitation_hours REAL,
        windspeed_10m_max REAL,
        et0_fao_evapotranspiration REAL,
        year INTEGER,
        month INTEGER,
        day INTEGER
    )
''')

# Save data to SQLite database
df.to_sql('weather', conn, if_exists='append', index=False)

# Commit changes and close connection
conn.commit()
conn.close()
#
# %% [predict_weather function]

def predict_weather(input_date, model_path='xgboost_model.model'):
    input_date = pd.to_datetime(input_date)
    input_data = pd.DataFrame({
        'year': [input_date.year],
        'month': [input_date.month],
        'day': [input_date.day]
    })

    labels = ['temperature_2m_mean', 'shortwave_radiation_sum', 'precipitation_sum', 'rain_sum', 'windspeed_10m_max', 'et0_fao_evapotranspiration']
    prediction = {}
    try:
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        prediction_array = model.predict(input_data)
        for i, label in enumerate(labels):
            prediction[label] = prediction_array[0][i]
            if label in ['rain_sum', 'precipitation_sum'] and prediction[label] < 0:
                prediction[label] = 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return {'error': str(e)}
    
    return prediction




 




##Streamlit app

# %%


def main():
    
    # Function to display weather predictions with emojis
    def display_predictions(key, value):
        emojis = {
            "temperature_2m_mean": "☀️",
            "rain_sum": "☔️",
            "precipitation_sum": "💧",
            "shortwave_radiation_sum": "🌞",
            "et0_fao_evapotranspiration": "🌱"
        }
        
        if key in emojis:
            st.write(f"{emojis[key]} {key}: {value:.2f}")
        else:
            st.write(f"{key}: {value:.2f}")


    def plot_data(df):
        # Create a date column
        df['date'] = pd.to_datetime(df[['year','month','day']])

        # Temperature plot
        df_areachart_temp = df[['date', 'temperature_2m_mean']].set_index('date')
        st.area_chart(df_areachart_temp)

        # Histogram
        plt.figure(figsize=(10, 5))
        sns.histplot(data=df, x='shortwave_radiation_sum', kde=False, color='orange', bins=10)
        plt.title('Shortwave Radiation Sum Histogram')
        plt.xlabel('Shortwave Radiation Sum MJ/m²')
        plt.ylabel('Frequency')
        st.pyplot(plt)

        # Shortwave Radiation Sum plot
        df_linechart_radiation = df[['date', 'shortwave_radiation_sum']].set_index('date')
        g = sns.relplot(data=df, x='date', y='shortwave_radiation_sum', kind='line', aspect=2)
        g.set_xticklabels(rotation=45)
        plt.show()

        # Wind speed plot
        df_linechart = df[['date', 'windspeed_10m_max']].set_index('date')
        st.line_chart(df_linechart)

        # Violin plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x='month', y='et0_fao_evapotranspiration')
        plt.title('Monthly et0_fao_evapotranspiration Violin Plots')
        st.pyplot(plt)

        # Precipitation plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='date', y='precipitation_hours')
        plt.yscale('log')
        plt.ylim(bottom=0.1)
        plt.title('Precipitation Hours Over Time')
        plt.xlabel('Date')
        plt.ylabel('Precipitation Hours (log scale)')
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())

    st.title("Tunisia's Atmospheric Oracle: Tempus App")
    st.image(r'all_season_cycle.jpg', width=700)

    st.sidebar.title("About")
    st.sidebar.info("This app uses an XGBoost model trained on weather data to make predictions.")
    
    
    # user interaction code 
    st.title("Weather Buddy: Chatbot and Predictions")
    st.write("Welcome! I'm your weather buddy. How can I assist you today?")
    st.write("Here's what I can do for you:")
    st.write("1. Show prediction of temperature_2m_mean")
    st.write("2. Show rain_sum")
    st.write("3. Show precipitation_sum")
    st.write("4. Show Shortwave Radiation Sum")
    st.write("5. Show et0_fao_evapotranspiration")
    st.write("6. Show all of the above")

    # User interaction options
    option = st.text_input("Please type the number of the option you want:")

    if option:  # check if option is not empty
        if option.isdigit():
            option = int(option)
            if option >= 1 and option <= 6:
                # Get user input for date
                date_input = st.text_input("Please enter a date (YYYY-MM-DD) for the prediction:")

                if date_input:  # check if date_input is not empty
                    # Convert user input date to datetime format
                    try:
                        date = datetime.strptime(date_input, "%Y-%m-%d").date()
                    except ValueError:
                        st.error("Invalid date format. Please enter a date in the format YYYY-MM-DD.")
                        return

                    # Display weather predictions based on user input
                    results = predict_weather(date_input, model_path='xgboost_model.model')
                    if 'error' in results:
                        st.error("An error occurred: " + results['error'])
                    else:
                        # Create mapping between options and keys
                        options = {
                            1: 'temperature_2m_mean',
                            2: 'rain_sum',
                            3: 'precipitation_sum',
                            4: 'shortwave_radiation_sum',
                            5: 'et0_fao_evapotranspiration'
                        }

                        if option == 6:
                            for key, value in results.items():
                                display_predictions(key, value)
                        else:
                            key = options[option]
                            if key in results:
                                display_predictions(key, results[key])
                            else:
                                st.error(f"No prediction available for {key}.")

            else:
                st.error("Invalid option. Please enter a number between 1 and 6.")
        else:
            st.error("Invalid input. Please enter a number.")
    # Assuming df is your DataFrame
    plot_data(df)

if __name__ == "__main__":
    main()






# %% [markdown]












