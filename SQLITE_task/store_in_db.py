import sqlite3
import requests
import pandas as pd

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
