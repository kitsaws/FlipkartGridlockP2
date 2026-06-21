import pandas as pd
import numpy as np
import datetime

def clean_data(input_file, output_file, report_file):
    print(f"Loading raw dataset from {input_file}...")
    df = pd.read_csv(input_file)
    initial_shape = df.shape
    
    with open(report_file, 'w') as f:
        f.write("=== Data Cleaning Report ===\n")
        f.write(f"Initial row count: {initial_shape[0]}\n")
        f.write(f"Initial column count: {initial_shape[1]}\n\n")

    # 2.1 Remove Leakage Features
    leakage_cols = [
        'updated_vehicle_number', 'updated_vehicle_type', 
        'validation_status', 'validation_timestamp', 
        'data_sent_to_scita_timestamp'
    ]
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    
    # 2.2 Remove Empty Columns (100% missing)
    empty_cols = ['description', 'closed_datetime', 'action_taken_timestamp']
    cols_to_drop.extend([c for c in empty_cols if c in df.columns])
    
    df.drop(columns=cols_to_drop, inplace=True)
    
    with open(report_file, 'a') as f:
        f.write(f"Dropped {len(cols_to_drop)} columns (leakage & empty).\n")
        f.write(f"Columns removed: {', '.join(cols_to_drop)}\n\n")

    # 2.3 Duplicate Handling
    # Identify duplicate rows based on event-defining columns
    event_cols = ['created_datetime', 'vehicle_number', 'latitude', 'longitude', 'violation_type']
    # Check if these exist just in case
    event_cols = [c for c in event_cols if c in df.columns]
    
    initial_rows = len(df)
    duplicates = df.duplicated(subset=event_cols, keep='first')
    df = df[~duplicates]
    dropped_dupes = initial_rows - len(df)
    
    with open(report_file, 'a') as f:
        f.write(f"Found and removed {dropped_dupes} duplicate rows.\n")
        f.write(f"Duplicates defined by matching: {', '.join(event_cols)}\n\n")

    # 2.4 Missing Value Handling
    # Spatial & Temporal critical drops
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = df.dropna(subset=['latitude', 'longitude'])
    if 'created_datetime' in df.columns:
        df = df.dropna(subset=['created_datetime'])
        
    with open(report_file, 'a') as f:
        f.write(f"Rows after dropping missing coordinates/timestamps: {len(df)}\n")

    # Categorical imputation
    # We saw location and center_code had a few missing
    cat_cols_with_missing = ['location', 'center_code']
    for col in cat_cols_with_missing:
        if col in df.columns:
            # fill with 'Unknown'
            # center_code is float, so converting it to string first prevents mixed types
            df[col] = df[col].astype(str).fillna('Unknown')
            df[col] = df[col].replace('nan', 'Unknown')
    
    # 2.5 Data Validation (Flagging)
    print("Performing data validation...")
    
    # Coordinates validity
    df['is_valid_coordinates'] = True
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['is_valid_coordinates'] = (
            (df['latitude'] >= -90) & (df['latitude'] <= 90) &
            (df['longitude'] >= -180) & (df['longitude'] <= 180)
        )
    
    invalid_coords = (~df['is_valid_coordinates']).sum()
    
    # Temporal validity
    df['is_valid_temporal'] = True
    if 'created_datetime' in df.columns and 'modified_datetime' in df.columns:
        try:
            created_dt = pd.to_datetime(df['created_datetime'], utc=True, errors='coerce')
            modified_dt = pd.to_datetime(df['modified_datetime'], utc=True, errors='coerce')
            df['is_valid_temporal'] = (modified_dt >= created_dt) | (modified_dt.isna())
        except Exception as e:
            print("Temporal parse issue:", e)
    
    invalid_temporal = (~df['is_valid_temporal']).sum()

    with open(report_file, 'a') as f:
        f.write("=== Data Quality Checks ===\n")
        f.write(f"Rows flagged with invalid coordinates: {invalid_coords}\n")
        f.write(f"Rows flagged with invalid temporal relations: {invalid_temporal}\n\n")
        f.write(f"Final dataset shape: {df.shape[0]} rows, {df.shape[1]} columns.\n")
        
    # Save the cleaned data
    print(f"Saving cleaned dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done. Review the cleaning report!")

if __name__ == "__main__":
    INPUT_PATH = "../DATASET/jan to may police violation_anonymized791b166.csv"
    OUTPUT_PATH = "../cleaned/cleaned_data.csv"
    REPORT_PATH = "../reports/cleaning_report.txt"
    
    print("=== Starting Task 2: Data Cleaning ===")
    clean_data(INPUT_PATH, OUTPUT_PATH, REPORT_PATH)
