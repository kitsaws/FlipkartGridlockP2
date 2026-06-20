import pandas as pd
import numpy as np
import h3
import os

def build_time_series(input_file, output_file):
    print(f"Loading cleaned dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Parse datetime and extract hour floor
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], format='mixed')
    
    print("Generating H3 Spatial Grid from raw coordinates...")
    df = df.dropna(subset=['latitude', 'longitude'])
    def get_h3(row):
        try:
            return h3.latlng_to_cell(row['latitude'], row['longitude'], 8)
        except:
            return None
    df['h3_cell_id'] = df.apply(get_h3, axis=1)
    df = df.dropna(subset=['h3_cell_id'])
    
    print("Stage 10.1: Grouping chronologically (Date + Hour)...")
    df['timestamp_hour'] = df['created_datetime'].dt.floor('h')
    
    # Check bounds to ensure we don't blow up memory with a 10-year outlier
    min_date = df['timestamp_hour'].min()
    max_date = df['timestamp_hour'].max()
    print(f"Dataset spans from {min_date} to {max_date}")
    
    # Aggregate violations by cell and exact hour
    hourly_counts = df.groupby(['h3_cell_id', 'timestamp_hour']).size().reset_index(name='violation_count')
    
    print("Stage 10.2: Performing Zero-Inflation (Padding missing hours)...")
    # Pivot so time is index and cells are columns
    pivot_df = hourly_counts.pivot(index='timestamp_hour', columns='h3_cell_id', values='violation_count')
    
    # Resample to 1 hour frequency, filling empty slots with 0 (Zero-Inflation)
    pivot_df = pivot_df.resample('1h').asfreq().fillna(0)
    
    print("Stage 10.3: Calculating Historical & Trend Features...")
    # Shift operations on the pivoted dataframe apply to every cell simultaneously
    last_week_df = pivot_df.shift(168)  # 168 hours = 7 days
    yesterday_df = pivot_df.shift(24)   # 24 hours = 1 day
    
    # Rolling Means
    rolling_3d_mean = pivot_df.rolling(window=72, min_periods=1).mean()
    rolling_7d_mean = pivot_df.rolling(window=168, min_periods=1).mean()
    
    print("Stage 10.4: Melting back to Long Format...")
    # Melt helper
    def melt_df(df_to_melt, val_name):
        return df_to_melt.reset_index().melt(id_vars=['timestamp_hour'], value_name=val_name)
        
    ts_df = melt_df(pivot_df, 'violation_count')
    ts_df['count_last_week'] = melt_df(last_week_df, 'v')['v']
    ts_df['count_yesterday'] = melt_df(yesterday_df, 'v')['v']
    ts_df['rolling_3d_mean'] = melt_df(rolling_3d_mean, 'v')['v']
    ts_df['rolling_7d_mean'] = melt_df(rolling_7d_mean, 'v')['v']
    
    print("Stage 10.5: Engineering Standard Temporal Features & Labels...")
    ts_df['hour'] = ts_df['timestamp_hour'].dt.hour
    ts_df['day_of_week'] = ts_df['timestamp_hour'].dt.dayofweek
    ts_df['is_weekend'] = ts_df['day_of_week'].isin([5, 6]).astype(int)
    ts_df['month'] = ts_df['timestamp_hour'].dt.month
    
    # Drop rows where 'count_last_week' is NaN (the first 7 days of our timeline)
    ts_df = ts_df.dropna(subset=['count_last_week'])
    
    # Label Generation (Is this hour a Hotspot?)
    # We define a hotspot event as an hour where violations exceed the 95th percentile of typical active hours
    active_hours = ts_df[ts_df['violation_count'] > 0]['violation_count']
    threshold = active_hours.quantile(0.95)
    
    # Ensure threshold isn't trivially small (e.g. at least 5 tickets)
    threshold = max(threshold, 5)
    print(f"Risk Threshold set to: >= {threshold:.1f} violations in an hour")
    
    ts_df['is_hotspot'] = (ts_df['violation_count'] >= threshold).astype(int)
    
    print(f"Final Predictive Dataset constructed: {ts_df.shape[0]} rows, {ts_df.shape[1]} columns.")
    
    os.makedirs('features', exist_ok=True)
    ts_df.to_csv(output_file, index=False)
    print(f"Saved successfully to {output_file}")

if __name__ == "__main__":
    INPUT_PATH = "cleaned/cleaned_data.csv"
    OUTPUT_PATH = "features/predictive_time_series.csv"
    
    print("=== Starting Task 10: Predictive Feature Engineering ===")
    build_time_series(INPUT_PATH, OUTPUT_PATH)
