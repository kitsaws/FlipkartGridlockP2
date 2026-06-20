import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer

def perform_aggregation(input_file, output_file):
    print(f"Loading engineered dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    print("Stage 5: Violation Feature Engineering...")
    # The violation_type column contains stringified JSON arrays like '["WRONG PARKING"]'
    def parse_violations(val):
        try:
            return json.loads(val)
        except:
            return [val]

    df['violation_list'] = df['violation_type'].apply(parse_violations)
    
    # Multi-hot encode the violation types
    mlb = MultiLabelBinarizer()
    violation_encoded = mlb.fit_transform(df['violation_list'])
    
    # Clean column names
    violation_cols = [f"is_{str(c).lower().replace(' ', '_').replace('/', '_')}" for c in mlb.classes_]
    encoded_df = pd.DataFrame(violation_encoded, columns=violation_cols)
    
    # Merge back
    df = pd.concat([df, encoded_df], axis=1)
    
    print("Stage 6: Zone-Time Aggregation...")
    # Ensure datetime format for active days/weeks
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], format='mixed')
    
    # Drop rows without an H3 cell (if any)
    df = df.dropna(subset=['h3_cell_id'])
    
    # We will aggregate by h3_cell_id and hour
    group = df.groupby(['h3_cell_id', 'hour'])
    
    print("Aggregating metrics...")
    agg_df = group.agg(
        violation_count=('id', 'count'),
        unique_vehicle_count=('vehicle_number', 'nunique'),
        violation_diversity=('violation_type', 'nunique'),
        vehicle_type_diversity=('vehicle_type', 'nunique'),
        days_active=('created_datetime', lambda x: x.dt.date.nunique()),
        weeks_active=('created_datetime', lambda x: x.dt.isocalendar().week.nunique()),
        night_violation_count=('is_night', 'sum'),
        peak_hour_violation_count=('is_peak_hour', 'sum')
    ).reset_index()
    
    # Also sum the multi-hot violation columns to know the severity breakdown per zone-hour
    violation_sums = df[['h3_cell_id', 'hour'] + violation_cols].groupby(['h3_cell_id', 'hour']).sum().reset_index()
    agg_df = pd.merge(agg_df, violation_sums, on=['h3_cell_id', 'hour'])
    
    print("Calculating Ratios...")
    agg_df['repeat_offender_ratio'] = 1.0 - (agg_df['unique_vehicle_count'] / agg_df['violation_count'])
    agg_df['night_violation_ratio'] = agg_df['night_violation_count'] / agg_df['violation_count']
    agg_df['peak_hour_ratio'] = agg_df['peak_hour_violation_count'] / agg_df['violation_count']
    
    # Rename weeks_active to weekly_persistence to match README terminology
    agg_df.rename(columns={'weeks_active': 'weekly_persistence'}, inplace=True)
    
    print(f"Final aggregated dataset shape: {agg_df.shape[0]} rows, {agg_df.shape[1]} columns")
    print(f"Saving to {output_file}...")
    agg_df.to_csv(output_file, index=False)
    print("Task 4 completed successfully!")

if __name__ == "__main__":
    INPUT_PATH = "features/engineered_data.csv"
    OUTPUT_PATH = "features/zone_time_aggregated.csv"
    
    print("=== Starting Task 4: Aggregation ===")
    perform_aggregation(INPUT_PATH, OUTPUT_PATH)
