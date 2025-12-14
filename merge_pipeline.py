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
    print("Loading Silver Layer data...")
    
    df_traffic = pd.read_csv(TRAFFIC_PATH)
    df_weather = pd.read_csv(WEATHER_PATH)

    df_traffic.columns = df_traffic.columns.str.strip()
    df_weather.columns = df_weather.columns.str.strip()

    for df, name in [(df_traffic, "Traffic"), (df_weather, "Weather")]:
        for col in ['date_time', 'city']:
            if col not in df.columns:
                raise ValueError(f"{name} missing required column: {col}")

    print(f"Traffic Data Loaded: {df_traffic.shape}")
    print(f"Weather Data Loaded: {df_weather.shape}")

    return df_traffic, df_weather



def check_time_format(df1, df2, time_col='date_time'):
    """
    Ensures both dataframes have the same datetime format
    before merging.
    """
    for name, df in [('Traffic', df1), ('Weather', df2)]:
        if time_col not in df.columns:
            raise ValueError(f"{name} dataframe is missing '{time_col}' column")

        # Convert to datetime
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

        # Fail fast if conversion failed
        if df[time_col].isna().any():
            bad_rows = df[df[time_col].isna()].shape[0]
            raise ValueError(
                f"{name} dataframe has {bad_rows} invalid datetime values in '{time_col}'"
            )

        # Ensure timezone consistency (naive)
        if df[time_col].dt.tz is not None:
            df[time_col] = df[time_col].dt.tz_localize(None)

    # Final safety check
    if df1[time_col].dtype != df2[time_col].dtype:
        raise TypeError("Datetime formats do not match after conversion")

    print("Time format check passed for both datasets")
    return df1, df2


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
    if traffic is None or weather is None:
        return

    # 2. Ensure consistent time format
    traffic, weather = check_time_format(traffic, weather, time_col='date_time')

    # 3. Temporal Alignment
    traffic = align_timestamps(traffic, time_col='date_time')
    weather = align_timestamps(weather, time_col='date_time')

    # 4. Merging
    print("Merging datasets...")
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

    print("Merged data info:")
    merged_df.info()
   




if __name__ == "__main__":
    run_pipeline()
    



