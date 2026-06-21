import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def compute_priority_scores(input_file, output_file):
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # We only want to deploy enforcement officers to actual Hotspots. 
    # Therefore, we filter out noise (cluster_id == -1)
    df = df[df['cluster_id'] != -1].copy()
    print(f"Processing {len(df)} Zone-Time blocks belonging to identified hotspots...")
    
    print("Stage 9: Computing Enforcement Priority Scores...")
    
    # 1. Density: 'violation_count'
    # 2. Persistence: 'weekly_persistence'
    # 3. Repeat Offender: 'repeat_offender_ratio'
    # 4. Severity: We need to calculate an average severity score based on the violation types present
    
    # Defining weights for severe violations (baseline is 1.0)
    severity_weights = {
        'is_wrong_parking': 1.0,
        'is_no_parking': 1.0,
        'is_parking_on_footpath': 2.0,
        'is_double_parking': 2.0,
        'is_parking_near_road_crossing': 2.0,
        'is_parking_near_bustop_school_hospital_etc': 3.0,
        'is_parking_near_traffic_light_or_zebra_cross': 3.0,
        'is_obstructing_driver': 3.0,
        'is_against_one_way_no_entry': 3.0,
        'is_jumping_traffic_signal': 3.0
    }
    
    # Calculate the total severity points in the bucket
    severity_score_col = pd.Series(0.0, index=df.index)
    for col, weight in severity_weights.items():
        if col in df.columns:
            severity_score_col += df[col] * weight
            
    # Calculate the average severity per violation in this bucket
    df['severity_score'] = severity_score_col / df['violation_count']
    df['severity_score'] = df['severity_score'].fillna(1.0)
    
    # Normalize the 4 components to a [0, 1] scale so they can be combined equitably
    scaler = MinMaxScaler()
    components = ['violation_count', 'weekly_persistence', 'severity_score', 'repeat_offender_ratio']
    normalized_cols = [f"{c}_norm" for c in components]
    
    df[normalized_cols] = scaler.fit_transform(df[components])
    
    # The mathematical formula for Enforcement Priority
    # Heavy emphasis on density (40%) and persistence (30%), with severity (20%) and repeat offenders (10%) as tie-breakers.
    df['priority_score'] = (
        0.40 * df['violation_count_norm'] +
        0.30 * df['weekly_persistence_norm'] +
        0.20 * df['severity_score_norm'] +
        0.10 * df['repeat_offender_ratio_norm']
    )
    
    # Scale to 0-100 for easier human interpretation
    df['priority_score'] = (df['priority_score'] * 100).round(2)
    
    # Sort the table to put the most critical enforcement targets at the absolute top
    df = df.sort_values(by='priority_score', ascending=False)
    
    print("\n--- Top 5 Enforcement Targets ---")
    print(df[['h3_cell_id', 'hour', 'cluster_id', 'priority_score', 'violation_count', 'severity_score']].head(5).to_string(index=False))
    print("---------------------------------\n")
    
    df.to_csv(output_file, index=False)
    print(f"Saved prioritized zones to {output_file}")
    print("Task 6 completed successfully!")

if __name__ == "__main__":
    INPUT = "../features/zone_time_aggregated_clusters.csv"
    OUTPUT = "../features/enforcement_priority.csv"
    
    print("=== Starting Task 6: Enforcement Priority Engine ===")
    compute_priority_scores(INPUT, OUTPUT)
