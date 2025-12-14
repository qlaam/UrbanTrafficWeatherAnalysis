import pandas as pd
import numpy as np
import os

# --- Configuration ---
TRAFFIC_PATH = r"data\cleaned\traffic_cleaned\traffic_cleaned.csv"
WEATHER_PATH = r"data\cleaned\Weather_cleaned\weather_cleaned.csv"
OUTPUT_DIR = 'data/gold'
OUTPUT_FILE = 'traffic_weather_merged.parquet'
OUTPUT_CSV = 'traffic_weather_merged.csv'

def load_data():
    """Loads cleaned datasets from the Silver Layer."""
    print("Loading Silver Layer data...")
    df_traffic = pd.read_csv(TRAFFIC_PATH)
    df_weather = pd.read_csv(WEATHER_PATH)
    print(f"Traffic Data Loaded: {df_traffic.shape}")
    print(f"Weather Data Loaded: {df_weather.shape}")
    return df_traffic, df_weather

def align_timestamps(df, time_col='date_time'):
    """
    Rounds timestamps to the nearest hour to ensure Traffic and Weather data 
    can be joined correctly (since sensors might differ by minutes/seconds).
    """
    # Create a temporary key for merging
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df['merge_key'] = df[time_col].dt.round('h')
    return df

def feature_engineering(df):
    """Adds analytical features for the Gold Layer."""
    # 1. Day of Week (e.g., Monday, Tuesday)
    df['day_of_week'] = df['date_time'].dt.day_name()
    
    # 2. Is Weekend? (True if Saturday or Sunday)
    df['is_weekend'] = df['date_time'].dt.weekday >= 5
    
    # 3. Hour of Day
    df['hour'] = df['date_time'].dt.hour
    
    # 4. Rush Hour Flag (Morning: 7-9 AM, Evening: 4-7 PM)
    df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0)
    
    return df

def run_pipeline():
    # 1. Load Data
    traffic, weather = load_data()
    if traffic is None or weather is None: return

    # 2. Temporal Alignment
    traffic = align_timestamps(traffic)
    weather = align_timestamps(weather)

    # 3. Merging (Inner Join)
    print("Merging datasets...")
    # Join on City and the rounded Time (merge_key)
    merged_df = pd.merge(
        traffic, 
        weather, 
        left_on=['merge_key', 'city'], 
        right_on=['merge_key', 'city'], 
        how='inner',
        suffixes=('_traffic', '_weather')
    )


    # Clean up redundant columns after merge
    if 'date_time_weather' in merged_df.columns:
        merged_df.drop(columns=['date_time_weather'], inplace=True)
    
    merged_df.rename(columns={'date_time_traffic': 'date_time'}, inplace=True)
    merged_df.drop(columns=['merge_key'], inplace=True)

    # 4. Feature Engineering
    merged_df = feature_engineering(merged_df)

    # 5. Final Audit
    print("\n--- Gold Layer Audit ---")
    print(f"Total Merged Records: {len(merged_df)}")
    print(f"Missing Values: {merged_df.isnull().sum().sum()}")
    print(f"Columns: {list(merged_df.columns)}")
    
    # 6. Save to Gold Layer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged_df.to_parquet(os.path.join(OUTPUT_DIR, OUTPUT_FILE))
    merged_df.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_CSV), index=False)
    print(f"\nSuccess! Data saved to {OUTPUT_DIR}/{OUTPUT_FILE}")

    print("---"*20)
    print()
    print()
    print(f"Merged data: {merged_df.info()}")
    print()
    print()
    print("---"*20) 




if __name__ == "__main__":
    run_pipeline()
    



