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
import dateparser
import speech_recognition as sr


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


# Create a connection to the database
conn = sqlite3.connect('weather_data.db')

# Write the data to a sqlite table
df.to_sql('weather', conn, if_exists='replace', index=False)

# Close the connection
conn.close()

# %%
# #Make prediction

# %%
def predict_weather(input_date, model_path):
    # Load the XGBoost model
    model = xgb.Booster()
    model.load_model(model_path)

    try:
        # Convert the user date to a dataframe
        input_date = pd.to_datetime(input_date)
        input_date = pd.DataFrame({
            'year': [input_date.year],
            'month': [input_date.month],
            'day': [input_date.day]
        })

        # Convert the data to DMatrix format
        user_data_dmatrix = xgb.DMatrix(input_date)

        # Make a prediction for the provided date
        prediction = model.predict(user_data_dmatrix)

        # Define labels for the predicted values
        labels = ['temperature_2m_mean', 'shortwave_radiation_sum', 'precipitation_sum', 'rain_sum', 'windspeed_10m_max', 'et0_fao_evapotranspiration']

        # Adjust negative values for rain_sum and precipitation_sum to 0
        for i, label in enumerate(labels):
            if label in ['rain_sum', 'precipitation_sum'] and prediction[0][i] < 0:
                prediction[0][i] = 0

        # Create a dictionary to store the predicted values and their corresponding labels
        results = {}

        # Add the predicted values and their corresponding labels to the dictionary
        for label, value in zip(labels, prediction[0]):
            results[label] = value

        return results

    except ValueError:
        print("You entered an invalid date. Please enter the date in the format YYYY-MM-DD.")
        return {}




# # Example usage
# # user_date_input = input("Please enter a date in the format YYYY-MM-DD: ")
# user_date_input = " 2023-06-19"
# model_name_input = 'xgboost_model.model'
# predict_weather(user_date_input, model_name_input)

# %% [markdown]
# #Streamlit app

# %%




def transcribe_speech():
    # Initialize recognizer class
    r = sr.Recognizer()
    # Reading Microphone as source
    with sr.Microphone() as source:
        st.info("Speak now...")
        # listen for speech and store in audio_text variable
        audio_text = r.listen(source)
        st.info("Transcribing...")

        try:
            # using Google Speech Recognition
            text = r.recognize_google(audio_text)
            return text
        except:
            return "Sorry, I did not get that."

def predict_weather(input_date, model_path='xgboost_model.model'):
    # Load the XGBoost model
    model = xgb.Booster()
    model.load_model(model_path)

    # Convert the user date to a dataframe
    input_date = pd.to_datetime(input_date)
    input_date = pd.DataFrame({
        'year': [input_date.year],
        'month': [input_date.month],
        'day': [input_date.day]
    })

    # Convert the data to DMatrix format
    user_data_dmatrix = xgb.DMatrix(input_date)

    # Make a prediction for the provided date
    prediction = model.predict(user_data_dmatrix)

    # Define labels for the predicted values
    labels = ['temperature_2m_mean', 'shortwave_radiation_sum', 'precipitation_sum', 'rain_sum', 'windspeed_10m_max', 'et0_fao_evapotranspiration']

    # Adjust negative values for rain_sum and precipitation_sum to 0
    for i, label in enumerate(labels):
        if label in ['rain_sum', 'precipitation_sum'] and prediction[0][i] < 0:
            prediction[0][i] = 0

    # Return a dictionary of predictions
    return dict(zip(labels, prediction[0]))
def main():
    st.title("Tunisia's Atmospheric Oracle:Tempus App")
    st.image(r'all_season_cycle.jpg', width=700)
    # Create a date column
    df['date'] = pd.to_datetime(df[['year','month','day']])

    # Temperature plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='date', y='temperature_2m_mean')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    # Shortwave Radiation Sum Histogram
    st.title('Shortwave Radiation Sum Histogram')
    plt.figure(figsize=(10, 6))
    plt.hist(df['shortwave_radiation_sum'], bins=30, edgecolor='black')
    plt.xlabel('Shortwave Radiation Sum (W/m²)')
    plt.ylabel('Frequency')
    st.pyplot(plt)
    # Create a new dataframe for the area chart
    df_areachart = df[['date', 'windspeed_10m_max']].set_index('date')

    # Plot the area chart
    # Create a new dataframe for the line chart
    df_linechart = df[['date', 'windspeed_10m_max']].set_index('date')

    # Plot the line chart
    st.line_chart(df_linechart)

    date_input = st.text_input("Hey there! I'm your weather buddy. Ask and I'll sprinkle some forecasts your way:please Enter a date for weather prediction (YYYY-MM-DD) or 'today':", value="")

    
    if st.button('Chat'):
        if date_input.lower() == 'today':
            date_input = datetime.date.today()
        
        else:
            date_input = pd.to_datetime(date_input)
    # elif add_selectbox == 'Speech':
    if st.button('Speech'):
        if st.button("Start Recording"):
            text = transcribe_speech()
            st.write("Transcription: ", text)

            if "today" in text:
                date_input = datetime.date.today()
            elif "tomorrow" in text:
                date_input = datetime.date.today() + datetime.timedelta(days=1)
            else:
                st.write("Sorry, I can't understand the date. Please try again.")
        else:
            return

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
# #Chatbot

# Function to generate a random response


# Call the chatbot function
##chatbot('xgboost_model.model')


# %%
# Function to generate a random response


# Function to handle user input and provide responses



# %%



