import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_RECORDS = 5000
DUPLICATE_RATE = 0.02  # 2% duplicates
NULL_RATE = 0.03  # 3% nulls per column
OUTLIER_RATE = 0.02  # 2% outliers

# Helper function to introduce nulls
def add_nulls(series, rate=NULL_RATE):
    mask = np.random.random(len(series)) < rate
    series[mask] = None
    return series

# Generate base data

# 1. Weather IDs (sequential)
weather_ids = list(range(5001, 5001 + NUM_RECORDS))

# 2. Generate dates (spanning full year 2024)
start_date = datetime(2024, 1, 1)
date_times = []
for i in range(NUM_RECORDS):
    days_offset = random.randint(0, 364)
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    date_times.append(start_date + timedelta(days=days_offset, hours=hour, minutes=minute))

# 3. City (all London, with some nulls)
cities = ['London'] * NUM_RECORDS

# 4. Season based on month
def get_season(date):
    if date is None or pd.isna(date):
        return None
    month = date.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

seasons = [get_season(dt) for dt in date_times]

# 5. Temperature (realistic by season with outliers)
def generate_temperature(season):
    if season == 'Winter':
        return np.random.normal(5, 5)
    elif season == 'Spring':
        return np.random.normal(12, 6)
    elif season == 'Summer':
        return np.random.normal(22, 5)
    elif season == 'Autumn':
        return np.random.normal(13, 5)
    else:
        return np.random.normal(12, 8)

temperatures = [generate_temperature(s) for s in seasons]

# Add temperature outliers
outlier_indices = np.random.choice(NUM_RECORDS, int(NUM_RECORDS * OUTLIER_RATE), replace=False)
for idx in outlier_indices[:len(outlier_indices)//2]:
    temperatures[idx] = random.choice([-30, -25, 60, 55])

# 6. Humidity (20-100 with outliers)
humidity = np.random.randint(30, 95, NUM_RECORDS).astype(float)
outlier_indices = np.random.choice(NUM_RECORDS, int(NUM_RECORDS * OUTLIER_RATE), replace=False)
for idx in outlier_indices:
    humidity[idx] = random.choice([-10, 0, 150, 120])

# 7. Rainfall (0-50mm typical, some extreme)
rain_mm = np.random.exponential(5, NUM_RECORDS)
rain_mm = np.clip(rain_mm, 0, 50)
outlier_indices = np.random.choice(NUM_RECORDS, int(NUM_RECORDS * OUTLIER_RATE), replace=False)
for idx in outlier_indices:
    rain_mm[idx] = random.uniform(120, 200)

# 8. Wind speed (0-80 typical, some extreme)
wind_speed = np.random.gamma(2, 10, NUM_RECORDS)
wind_speed = np.clip(wind_speed, 0, 80)
outlier_indices = np.random.choice(NUM_RECORDS, int(NUM_RECORDS * OUTLIER_RATE), replace=False)
for idx in outlier_indices:
    wind_speed[idx] = random.uniform(200, 300)

# 9. Visibility (50-10000m typical)
visibility = np.random.normal(5000, 3000, NUM_RECORDS)
visibility = np.clip(visibility, 50, 10000).astype(int)
outlier_indices = np.random.choice(NUM_RECORDS, int(NUM_RECORDS * OUTLIER_RATE), replace=False)
for idx in outlier_indices:
    visibility[idx] = 50000

# 10. Weather condition
weather_conditions = np.random.choice(
    ['Clear', 'Rain', 'Fog', 'Storm', 'Snow'],
    NUM_RECORDS,
    p=[0.4, 0.25, 0.15, 0.1, 0.1]
)

# 11. Air pressure (950-1050 typical)
air_pressure = np.random.normal(1013, 15, NUM_RECORDS)
air_pressure = np.clip(air_pressure, 950, 1050)

# Create DataFrame
df = pd.DataFrame({
    'weather_id': weather_ids,
    'date_time': date_times,
    'city': cities,
    'season': seasons,
    'temperature_c': temperatures,
    'humidity': humidity,
    'rain_mm': rain_mm,
    'wind_speed_kmh': wind_speed,
    'visibility_m': visibility,
    'weather_condition': weather_conditions,
    'air_pressure_hpa': air_pressure
})

# Format datetime YYYY-MM-DD HH:MM
df['date_time'] = df['date_time'].apply(
    lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notna(x) else x
)

# 1. Add duplicates
num_duplicates = int(NUM_RECORDS * DUPLICATE_RATE)
duplicate_indices = np.random.choice(df.index, num_duplicates, replace=False)
duplicates = df.loc[duplicate_indices].copy()
df = pd.concat([df, duplicates], ignore_index=True)

# 2. Add NULL values to weather_id
null_indices = np.random.choice(df.index, int(len(df) * NULL_RATE), replace=False)
df.loc[null_indices, 'weather_id'] = None

# 3. Mess up some date_time formats
date_format_messy_indices = np.random.choice(df.index, int(len(df) * 0.05), replace=False)

# Format 1: DD/MM/YYYY HH:MM format
for idx in date_format_messy_indices[:len(date_format_messy_indices)//4]:
    if pd.notna(df.loc[idx, 'date_time']):
        dt_str = df.loc[idx, 'date_time']
        try:
            dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M')
            df.loc[idx, 'date_time'] = f"{dt.day:02d}/{dt.month:02d}/{dt.year} {dt.hour:02d}:{dt.minute:02d}"
        except:
            pass

# Format 2: DD/MM/YYYY with AM/PM (like 15/01/2024 2PM)
for idx in date_format_messy_indices[len(date_format_messy_indices)//4:len(date_format_messy_indices)//2]:
    if pd.notna(df.loc[idx, 'date_time']):
        dt_str = df.loc[idx, 'date_time']
        try:
            dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M')
            hour_12 = dt.hour % 12 if dt.hour % 12 != 0 else 12
            am_pm = "AM" if dt.hour < 12 else "PM"
            df.loc[idx, 'date_time'] = f"{dt.day}/{dt.month}/{dt.year} {hour_12}{am_pm}"
        except:
            pass

# Format 3: ISO format with Z (like 2024-01-15T14:00Z)
for idx in date_format_messy_indices[len(date_format_messy_indices)//2:3*len(date_format_messy_indices)//4]:
    if pd.notna(df.loc[idx, 'date_time']):
        dt_str = df.loc[idx, 'date_time']
        try:
            dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M')
            df.loc[idx, 'date_time'] = f"{dt.year}-{dt.month:02d}-{dt.day:02d}T{dt.hour:02d}:{dt.minute:02d}Z"
        except:
            pass

# Invalid/garbage dates
for idx in date_format_messy_indices[3*len(date_format_messy_indices)//4:]:
    df.loc[idx, 'date_time'] = random.choice([
        "2099-13-40 25:61",
        "Unknown",
        "Invalid Date",
        "2024-00-00 99:99"
    ])

# Add some NULL date_times
null_dt_indices = np.random.choice(df.index, int(len(df) * NULL_RATE), replace=False)
df.loc[null_dt_indices, 'date_time'] = None

# 4. Add NULL cities
df['city'] = add_nulls(df['city'].copy(), NULL_RATE)

# 5. Add NULL seasons
df['season'] = add_nulls(df['season'].copy(), NULL_RATE)

# 6. Add NULLs to numeric columns
df['temperature_c'] = add_nulls(df['temperature_c'].copy(), NULL_RATE)
df['humidity'] = add_nulls(df['humidity'].copy(), NULL_RATE)
df['rain_mm'] = add_nulls(df['rain_mm'].copy(), NULL_RATE)
df['wind_speed_kmh'] = add_nulls(df['wind_speed_kmh'].copy(), NULL_RATE)

# Add some non-numeric strings to visibility
visibility_messy_indices = np.random.choice(df.index, 10, replace=False)
for idx in visibility_messy_indices:
    df.loc[idx, 'visibility_m'] = "unclear"
df['visibility_m'] = add_nulls(df['visibility_m'].copy(), NULL_RATE)

# 7. Add NULLs to weather_condition
df['weather_condition'] = add_nulls(df['weather_condition'].copy(), NULL_RATE)

# 8. Add NULLs to air_pressure
df['air_pressure_hpa'] = add_nulls(df['air_pressure_hpa'].copy(), NULL_RATE)

# Shuffle the data, but keep weather_id sequential
df_data = df.drop('weather_id', axis=1).sample(frac=1).reset_index(drop=True)
df_data.insert(0, 'weather_id', range(5001, 5001 + len(df_data)))

# Save to CSV
output_file = 'synthetic_weather_data.csv'
df_data.to_csv(output_file, index=False)
print(f"Dataset saved to '{output_file}'")