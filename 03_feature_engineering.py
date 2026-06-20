import pandas as pd
import h3

def perform_feature_engineering(input_file, output_file):
    print(f"Loading cleaned dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Ensure created_datetime is a datetime object
    print("Extracting Temporal Features...")
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], utc=True, format='mixed')
    
    # 1. Temporal Feature Engineering (Stage 3)
    df['hour'] = df['created_datetime'].dt.hour
    df['day_of_week'] = df['created_datetime'].dt.dayofweek # 0=Monday, 6=Sunday
    df['month'] = df['created_datetime'].dt.month
    
    # is_weekend: Saturday (5) and Sunday (6)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Morning Peak = 7-10, Evening Peak = 17-20
    # Inclusive: 7, 8, 9, 10 and 17, 18, 19, 20
    df['is_peak_hour'] = (
        ((df['hour'] >= 7) & (df['hour'] <= 10)) | 
        ((df['hour'] >= 17) & (df['hour'] <= 20))
    ).astype(int)
    
    # Night = 22-5
    # Inclusive: 22, 23, 0, 1, 2, 3, 4, 5
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    
    # 2. Spatial Feature Engineering (Stage 4)
    print("Extracting Spatial Features (H3 Indexing)...")
    H3_RESOLUTION = 8
    
    def get_h3_cell(row):
        try:
            # For h3 package >= 4.0, we use latlng_to_cell
            return h3.latlng_to_cell(row['latitude'], row['longitude'], H3_RESOLUTION)
        except Exception:
            return None

    # Apply H3 indexing to all valid coordinates
    df['h3_cell_id'] = df.apply(get_h3_cell, axis=1)
    
    # Validate how many failed to convert (should be 0 since we cleaned them)
    missing_h3 = df['h3_cell_id'].isnull().sum()
    print(f"H3 Cells missing/failed: {missing_h3}")

    print(f"Saving engineered dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done! Feature engineering completed.")

if __name__ == "__main__":
    INPUT_PATH = "cleaned/cleaned_data.csv"
    OUTPUT_PATH = "features/engineered_data.csv"
    
    print("=== Starting Task 3: Feature Engineering ===")
    perform_feature_engineering(INPUT_PATH, OUTPUT_PATH)
