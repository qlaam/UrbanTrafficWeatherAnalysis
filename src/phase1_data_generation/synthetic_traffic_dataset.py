"""
synthetic_traffic_dataset.py

I make this script to generate a synthetic traffic dataset consistent with Big Data
project requirements. The generated dataset includes realistic conditions,
controlled anomalies, duplicate records, missing values, and messy formatting.

"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime

# First i make configuration to make parameters global so ican use them accross my code
CONFIG = {
    "RANDOM_SEED": 42,
    "N_TRAFFIC": 5000,  
    "TRAFFIC_ID_START": 9001,
    "CITY_NAME": "London",
    "AREAS": ["Camden", "Chelsea", "Islington", "Southwark","Kensington", "Greenwich", "Hackney", "Hammersmith"],
    "OUTPUT_DIR": "./outputs",
    
    #probabilities as required by PDF
    "date_invalid_prob": 0.02,    
    "date_null_prob": 0.01,       
    "text_in_numeric_prob": 0.005, 
    "null_value_prob": 0.01,      
    "area_null_prob": 0.06,       
    "duplicate_prob": 0.01,       
}

# Second iniatialization
RND = CONFIG["RANDOM_SEED"]
random.seed(RND)
np.random.seed(RND)
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

# helper functions to use 
def rand_datetimes(n, start_dt=datetime(2024,1,1,0,0),end_dt=datetime(2024,12,31,23,59)):
    """
    Generate random timestamps within the required date range.
    Parameters: n:int (Number of timestamps to generate)
    Returns: list of datetime (Randomized timestamps within the given boundaries)
    """
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())
    ts = np.random.randint(start_ts, end_ts+1, size=n)
    return [datetime.fromtimestamp(int(t)) for t in ts]

def mix_date_formats(dt_list, prob_invalid=0.02, prob_null=0.01):
    """
    i apply date formats as specified to:
    1. Standard: 2024-01-15 08:00
    2. Variation: 15/01/2024 8AM
    3. Variation: 2024-01-15T08:00Z (note: image.png shows T08:002 which is typo)
    and invalid values and NULLs.
    """
    fmts = [
        lambda d: d.strftime("%Y-%m-%d %H:%M"),      
        lambda d: d.strftime("%d/%m/%Y %I%p"),       
        lambda d: d.strftime("%Y-%m-%dT%H:%MZ"),     
    ]
    garbage = ["2099-13-40 25:61", "TBD", "2099-00-00 99:99", 
               "Unknown", "??/??/????", "2024-02-31 12:00"]
    
    out = []
    for d in dt_list:
        r = random.random()
        if r < prob_null:
            out.append(None)
        elif r < prob_invalid + prob_null:
            out.append(random.choice(garbage))
        else:
            out.append(random.choice(fmts)(d))
    return out

def inject_text_values(series, text_values=["N/A", "unknown", "error", "-"], 
                       prob=0.01):
    # Inject messy text-based invalid values into numeric columns.
    # Example: "N/A", "unknown",.....
    return [random.choice(text_values) if random.random() < prob else v 
            for v in series]

def introduce_nulls(series, prob=0.01):
    #Randomly inject None values into any given list.
    return [None if random.random() < prob else v for v in series]

#Third traffic data generation engin
def generate_traffic(config=CONFIG):
    """
    I then generate a synthetic traffic dataset following the exact PDF specification.
    Returned DataFrame columns:
        traffic_id, date_time, city, area, vehicle_count,
        avg_speed_kmh, accident_count, congestion_level,
        road_condition, visibility_m
    """
    n = config["N_TRAFFIC"]
    
    # 1. timestamp coloumn
    dt_list = rand_datetimes(n)
    date_col = mix_date_formats(
        dt_list, 
        prob_invalid=config["date_invalid_prob"],
        prob_null=config["date_null_prob"]
    )
    
    # 2.traffic ids with null ids allowed
    ids = list(range(config["TRAFFIC_ID_START"], 
                     config["TRAFFIC_ID_START"] + n))

    for i in random.sample(range(n), k=max(1, int(n * 0.01))):
        ids[i] = None
    
    # 3.city coloumn
    city_col = [config["CITY_NAME"] if random.random() > 0.01 
                else None for _ in range(n)]
    
    # 4.area coloumns with controllowed vaules
    area_col = [random.choice(config["AREAS"]) for _ in range(n)]
    for i in random.sample(range(n), k=max(1, int(n * config["area_null_prob"]))):
        area_col[i] = None
    
    # 5.vehicle counts with allowed outlears
    vehicle_count = np.random.poisson(lam=200, size=n).tolist()
    vehicle_count = [min(v, 5000) for v in vehicle_count]

    for i in random.sample(range(n), k=max(1, int(n * 0.005))):
        vehicle_count[i] = random.choice([20000, 50000])

    vehicle_count = introduce_nulls(vehicle_count, prob=0.02)
    vehicle_count = inject_text_values(vehicle_count, text_values=["many", "N/A"], prob=config["text_in_numeric_prob"])
    
    # 6.avarage speed with allowed negative values
    avg_speed = []
    for _ in range(n):
        base_speed = np.random.normal(loc=45, scale=12)
        # Ensure normal range 3-120
        if base_speed < 3:
            base_speed = 3
        elif base_speed > 120:
            base_speed = 120
        avg_speed.append(round(base_speed, 1))
    
    for i in random.sample(range(n), k=max(1, int(n * 0.004))):
        avg_speed[i] = random.choice([-5.0, -20.0, -50.0])
    
    avg_speed = introduce_nulls(avg_speed, prob=config["null_value_prob"])
    avg_speed = inject_text_values(avg_speed, 
                                   text_values=["stopped", "N/A"], 
                                   prob=config["text_in_numeric_prob"])
    
    # 7.accident count
    accident_count = np.random.poisson(lam=0.2, size=n).tolist()
    accident_count = [min(v, 10) for v in accident_count]

    for i in random.sample(range(n), k=max(1, int(n * 0.002))):
        accident_count[i] = random.choice([50, 99])
    
    accident_count = introduce_nulls(accident_count, 
                                     prob=config["null_value_prob"])
    accident_count = inject_text_values(accident_count, 
                                        text_values=["unknown", "N/A"], 
                                        prob=config["text_in_numeric_prob"])
    
    # 8.congestion level with low , medium , high and messy
    congestion = [random.choice(["Low", "Medium", "High"]) for _ in range(n)]

    for i in random.sample(range(n), k=max(1, int(n * 0.03))):
        congestion[i] = random.choice([None, "", " "])
    
    # 9.road condition (Dry, Wet, Snowy, Damaged) with messy values
    road_condition = [random.choice(["Dry", "Wet", "Snowy", "Damaged"]) 
                      for _ in range(n)]

    for i in random.sample(range(n), k=max(1, int(n * 0.015))):
        road_condition[i] = None
    
    # 10.visability from 50–10000 and allowed extreme 50000
    visibility = np.random.randint(50, 10001, size=n).tolist()

    for i in random.sample(range(n), k=max(1, int(n * 0.003))):
        visibility[i] = random.choice([50000, 999999])
    
    visibility = introduce_nulls(visibility, prob=0.005)
    visibility = inject_text_values(visibility, text_values=["unknown", "clear", "N/A"],prob=config["text_in_numeric_prob"])
    
    #then i build data frame
    df = pd.DataFrame({
        "traffic_id": ids,
        "date_time": date_col,
        "city": city_col,
        "area": area_col,
        "vehicle_count": vehicle_count,
        "avg_speed_kmh": avg_speed,
        "accident_count": accident_count,
        "congestion_level": congestion,
        "road_condition": road_condition,
        "visibility_m": visibility
    })
    
    # 11.insert duplicates
    dup_n = max(1, int(n * config["duplicate_prob"]))
    dup_rows = df.sample(n=dup_n, replace=True, random_state=RND)
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    # 12. Add duplicate traffic_id with same value
    valid_ids = df['traffic_id'].dropna().unique()
    if len(valid_ids) > 5:
        dup_ids = random.sample(list(valid_ids), 5)
        for dup_id in dup_ids:
            original_row = df[df['traffic_id'] == dup_id].iloc[0].copy()
            
            # Create duplicate with same traffic_id
            new_row = original_row.copy()
            new_row['traffic_id'] = dup_id  

            # insert int the data frame
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=RND).reset_index(drop=True)
    
# Print statistics
    print("SYNTHETIC TRAFFIC DATASET GENERATION COMPLETE")
    print(f"Total records: {len(df)}")
    print(f"Expected: 5000 + duplicates")
    print(f"NULL values per column:\n{df.isnull().sum()}")
    print(f"\nDuplicate rows (exact duplicates): {df.duplicated().sum()}")
    print(f"Duplicate traffic_id values: {df['traffic_id'].duplicated().sum()}")
    
    print("VALUE RANGES (numeric values only):")
    numeric_vehicle = pd.to_numeric(df['vehicle_count'], errors='coerce')
    print(f"vehicle_count: {numeric_vehicle.min():.0f} to {numeric_vehicle.max():.0f}")
    print(f"  (Non-numeric values: {(df['vehicle_count'].apply(lambda x: isinstance(x, str))).sum()})")
    
    numeric_speed = pd.to_numeric(df['avg_speed_kmh'], errors='coerce')
    print(f"avg_speed_kmh: {numeric_speed.min():.1f} to {numeric_speed.max():.1f}")
    print(f"  (Non-numeric values: {(df['avg_speed_kmh'].apply(lambda x: isinstance(x, str))).sum()})")
    
    numeric_accident = pd.to_numeric(df['accident_count'], errors='coerce')
    print(f"accident_count: {numeric_accident.min():.0f} to {numeric_accident.max():.0f}")
    print(f"  (Non-numeric values: {(df['accident_count'].apply(lambda x: isinstance(x, str))).sum()})")
    
    numeric_visibility = pd.to_numeric(df['visibility_m'], errors='coerce')
    print(f"visibility_m: {numeric_visibility.min():.0f} to {numeric_visibility.max():.0f}")
    print(f"  (Non-numeric values: {(df['visibility_m'].apply(lambda x: isinstance(x, str))).sum()})")
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    return df

#finally main excution
def main():
    print("Generating messy synthetic dataset...")
    raw_traffic = generate_traffic(CONFIG)
    traffic_csv_path = os.path.join(CONFIG["OUTPUT_DIR"], 'synthetic_traffic_dataset.csv')
    
    raw_traffic.to_csv(traffic_csv_path, index=False)
    print("Saved raw files to:", traffic_csv_path)


if __name__ == "__main__":
    main()



