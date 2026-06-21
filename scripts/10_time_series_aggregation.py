import pandas as pd
import numpy as np
import h3
import os

def get_violation_weight(v_str):
    """
    Parses the list string of violations and assigns a severity weight.
    Dangerous violations carry more weight than standard parking violations.
    """
    if not isinstance(v_str, str): return 1.0
    v_str = v_str.upper()
    
    # Critical Safety Hazards (Weight 3.0)
    if any(k in v_str for k in ['DOUBLE PARKING', 'ZEBRA CROSS', 'TRAFFIC LIGHT', 'FOOTPATH', 'CROSSING']):
        return 3.0
    # Major Obstructions (Weight 2.0)
    elif any(k in v_str for k in ['MAIN ROAD', 'BUS', 'SCHOOL', 'HOSPITAL', 'OPPOSITE']):
        return 2.0
    # Standard Parking Violations (Weight 1.0)
    return 1.0

def build_time_series(input_file, output_file):
    print(f"Loading cleaned dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Parse datetime
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], format='mixed')
    
    print("Stage 10.1: Assigning Violation Weights...")
    df['violation_weight'] = df['violation_type'].apply(get_violation_weight)
    
    print("Stage 10.2: Generating H3 Spatial Grid from raw coordinates...")
    df = df.dropna(subset=['latitude', 'longitude'])
    def get_h3(row):
        try:
            return h3.latlng_to_cell(row['latitude'], row['longitude'], 8)
        except:
            return None
    df['h3_cell_id'] = df.apply(get_h3, axis=1)
    df = df.dropna(subset=['h3_cell_id'])
    
    print("Stage 10.3: Grouping chronologically (Date + Hour)...")
    df['date'] = df['created_datetime'].dt.date
    df['hour'] = df['created_datetime'].dt.hour
    
    # The Violation-Aware Aggregation
    hourly = df.groupby(['h3_cell_id', 'date', 'hour']).agg(
        violation_count=('created_datetime', 'count'),
        weighted_severity_score=('violation_weight', 'sum')
    ).reset_index()
    
    # To maintain chronology for time-series momentum, we perform Zero-Inflation
    print("Stage 10.4: Performing Zero-Inflation (Padding missing hours)...")
    hourly['datetime'] = pd.to_datetime(hourly['date'].astype(str) + ' ' + hourly['hour'].astype(str) + ':00:00')
    
    min_time = hourly['datetime'].min()
    max_time = hourly['datetime'].max()
    print(f"Dataset spans from {min_time} to {max_time}")
    
    full_time_range = pd.date_range(start=min_time, end=max_time, freq='h')
    all_cells = hourly['h3_cell_id'].unique()
    
    multi_idx = pd.MultiIndex.from_product([all_cells, full_time_range], names=['h3_cell_id', 'datetime'])
    full_grid = pd.DataFrame(index=multi_idx).reset_index()
    
    merged = pd.merge(full_grid, hourly, on=['h3_cell_id', 'datetime'], how='left')
    
    # Zeros for missing hours
    merged['violation_count'] = merged['violation_count'].fillna(0)
    merged['weighted_severity_score'] = merged['weighted_severity_score'].fillna(0)
    
    merged['date'] = merged['datetime'].dt.date
    merged['hour'] = merged['datetime'].dt.hour
    merged['day_of_week'] = merged['datetime'].dt.dayofweek
    merged['is_weekend'] = merged['day_of_week'].isin([5, 6]).astype(int)
    merged['month'] = merged['datetime'].dt.month
    
    merged = merged.sort_values(by=['h3_cell_id', 'datetime']).reset_index(drop=True)
    
    print("Stage 10.5: Calculating Historical & Trend Features (Momentum)...")
    merged['severity_last_week'] = merged.groupby('h3_cell_id')['weighted_severity_score'].shift(24 * 7)
    merged['severity_yesterday'] = merged.groupby('h3_cell_id')['weighted_severity_score'].shift(24)
    
    shifted = merged.groupby('h3_cell_id')['weighted_severity_score'].shift(1)
    merged['rolling_3d_severity'] = shifted.groupby(merged['h3_cell_id']).rolling(window=24 * 3, min_periods=1).mean().reset_index(level=0, drop=True)
    merged['rolling_7d_severity'] = shifted.groupby(merged['h3_cell_id']).rolling(window=24 * 7, min_periods=1).mean().reset_index(level=0, drop=True)
    
    final_df = merged.dropna(subset=['severity_last_week']).copy() 
    
    print("Stage 10.6: Generating Risk Classes (Low/Medium/High)...")
    # Identify percentiles only on ACTIVE hours to find the true baseline of congestion
    active = final_df[final_df['weighted_severity_score'] > 0]
    p75 = active['weighted_severity_score'].quantile(0.75)
    
    def get_risk_class(score):
        if score == 0:
            return 'Low'
        elif score <= p75:
            return 'Medium'
        else:
            return 'High'
            
    final_df['Risk_Class'] = final_df['weighted_severity_score'].apply(get_risk_class)
    
    print(f"Risk Class Distribution:\n{final_df['Risk_Class'].value_counts(normalize=True)*100}")
    
    print("Stage 10.7: Historical Profiler (Injecting Primary Violations)...")
    import ast
    def safe_parse(val):
        try:
            return ast.literal_eval(val)
        except:
            return [str(val)]
            
    df['parsed_violations'] = df['violation_type'].apply(safe_parse)
    exploded = df.explode('parsed_violations')
    
    # Calculate percentage of each violation per cell
    cell_violations = exploded.groupby('h3_cell_id')['parsed_violations'].value_counts(normalize=True).reset_index(name='primary_violation_pct')
    top_violations = cell_violations.groupby('h3_cell_id').first().reset_index()
    top_violations = top_violations.rename(columns={'parsed_violations': 'primary_violation'})
    top_violations['primary_violation_pct'] = (top_violations['primary_violation_pct'] * 100).round(1)
    
    final_df = pd.merge(final_df, top_violations, on='h3_cell_id', how='left')
    final_df['primary_violation'] = final_df['primary_violation'].fillna("None")
    final_df['primary_violation_pct'] = final_df['primary_violation_pct'].fillna(0.0)
    
    os.makedirs('../features', exist_ok=True)
    final_df.to_csv(output_file, index=False)
    print(f"Final Predictive Dataset constructed: {len(final_df)} rows.")
    print(f"Saved successfully to {output_file}")

if __name__ == "__main__":
    INPUT_PATH = "../cleaned/cleaned_data.csv"
    OUTPUT_PATH = "../features/predictive_time_series.csv"
    
    print("=== Starting Task 10: V2 Violation-Aware Feature Engineering ===")
    build_time_series(INPUT_PATH, OUTPUT_PATH)
