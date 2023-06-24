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
import datetime
import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment


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


# Create a connection to the database
conn = sqlite3.connect('weather_data.db')

# Write the data to a sqlite table
with sqlite3.connect('weather_data.db') as conn:
    df.to_sql('weather', conn, if_exists='replace', index=False)

# %%
def predict_weather(input_date, model_path='xgboost_model.model'):
    # Load the XGBoost model
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    try:
        # Convert the user date to a dataframe
        input_date = pd.to_datetime(input_date)
        input_date = pd.DataFrame({
            'year': [input_date.year],
            'month': [input_date.month],
            'day': [input_date.day]
        })

        # Make a prediction for the provided date
        prediction = model.predict(input_date)

        # Define labels for the predicted values
        labels = ['temperature_2m_mean', 'shortwave_radiation_sum', 'precipitation_sum', 'rain_sum', 'windspeed_10m_max', 'et0_fao_evapotranspiration']

        # Adjust negative values for rain_sum and precipitation_sum to 0
        for i, label in enumerate(labels):
            if i < len(prediction) and label in ['rain_sum', 'precipitation_sum'] and prediction[i] < 0:
                prediction[i] = 0

        # Prepare a dictionary of predicted values alongside their corresponding labels
        result = dict(zip(labels, prediction))

        return result
    except ValueError:
        print("You entered an invalid date. Please enter the date in the format YYYY-MM-DD.")

 
# %% [markdown]
##Streamlit app

# %%

def main():
    def recognize_audio(audio_file):
        audio = AudioSegment.from_file(audio_file, format='wav')
        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
            text = r.recognize_google(audio)
        return text

    st.title("Tunisia's Atmospheric Oracle:Tempus App")
    st.image(r'all_season_cycle.jpg', width=700)

    df['date'] = pd.to_datetime(df[['year','month','day']])

    df_areachart_temp = df[['date', 'temperature_2m_mean']].set_index('date')
    st.area_chart(df_areachart_temp)

    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='shortwave_radiation_sum', kde=False, color='orange', bins=10)
    plt.title('Shortwave Radiation Sum Histogram')
    plt.xlabel('Shortwave Radiation Sum MJ/m²')
    plt.ylabel('Frequency')
    st.pyplot(plt)

    df_areachart = df[['date', 'windspeed_10m_max']].set_index('date')
    df_linechart = df[['date', 'windspeed_10m_max']].set_index('date')
    st.line_chart(df_linechart)

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='month', y='et0_fao_evapotranspiration')
    plt.title('Monthly et0_fao_evapotranspiration Violin Plots')
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='date', y='precipitation_hours')
    plt.yscale('log')
    plt.ylim(bottom=0.1)
    plt.title('Precipitation Hours Over Time')
    plt.xlabel('Date')
    plt.ylabel('Precipitation Hours (log scale)')
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())

    date_input = st.text_input("Hey there! I'm your weather buddy. Ask and I'll sprinkle some forecasts your way:please Enter a date for weather prediction (YYYY-MM-DD) or 'today':", value="")

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            try:
                results = predict_weather(date_input)
                st.success("Prediction successful!")
                for key, value in results.items():
                    emoji = "☀️" if key == 'temperature_2m_mean' and value > 20 else "⛅️" if key == 'temperature_2m_mean' and value > 10 else "❄️"
                    emoji = "☔" if key in ['rain_sum', 'precipitation_sum'] and value > 0 else emoji
                    st.write(f"{emoji} {key}: {value:.2f}")
            except ValueError:
                st.error("Please enter a valid date.")

    st.sidebar.title("About")
    st.sidebar.info("This app uses an XGBoost model trained on weather data to make predictions.")

    if st.button('Chat'):
        if date_input.lower() == 'today':
            date_input = datetime.date.today()
        else:
            date_input = pd.to_datetime(date_input)

    if st.button('Speech'):
        if st.button("Start Recording"):
            text = transcribe_speech()
            st.write("Transcription: ", text)

            if "today" in text:
                date_input = datetime.date.today()
            else:
                st.write("Sorry, I can't understand the date. Please try again.")
        
        with st.spinner("Predicting..."):
            try:
                results = predict_weather(date_input, r"C:\Users\MARAM\Desktop\GOMYCODE\weather_app_project\xgboost_model.model")
                st.success("Prediction successful!")
                for key, value in results.items():
                    emoji = "☀️" if key == 'temperature_2m_mean' and value > 20 else "⛅️" if key == 'temperature_2m_mean' and value > 10 else "❄️"
                    emoji = "☔" if key in ['rain_sum', 'precipitation_sum'] and value > 0 else emoji
                    st.write(f"{emoji} {key}: {value:.2f}")
            except ValueError:
                st.error("Please enter a valid date.")

        st.sidebar.title("About")
        st.sidebar.info("This app uses an XGBoost model trained on weather data to make predictions.")

if __name__ == "__main__":
    main()




# %% [markdown]












