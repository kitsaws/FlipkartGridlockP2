import pandas as pd
import numpy as np
import h3
import hdbscan
import pickle
import json
import warnings

# Suppress pandas chained assignment warnings
warnings.filterwarnings('ignore')

def discover_hotspots(agg_input, eng_input, hotspot_output):
    print("Loading aggregated data to get unique spatial zones...")
    agg_df = pd.read_csv(agg_input)
    
    unique_cells = agg_df['h3_cell_id'].unique()
    print(f"Found {len(unique_cells)} unique H3 cells.")
    
    # Get centroids for each H3 cell
    cell_data = []
    for cell in unique_cells:
        try:
            lat, lng = h3.cell_to_latlng(cell)
            cell_data.append({'h3_cell_id': cell, 'latitude': lat, 'longitude': lng})
        except Exception:
            pass
            
    cell_df = pd.DataFrame(cell_data)
    
    print("Stage 7: Running HDBSCAN for Spatial Hotspot Discovery...")
    
    # We must filter out low-density "bridge" cells so HDBSCAN can properly find distinct dense islands
    cell_violations = agg_df.groupby('h3_cell_id')['violation_count'].sum().reset_index()
    cell_df = pd.merge(cell_df, cell_violations, on='h3_cell_id', how='left')
    
    # Only cluster cells that have significant violation volume (e.g., > 50 violations)
    high_density_cells = cell_df[cell_df['violation_count'] >= 50].copy()
    low_density_cells = cell_df[cell_df['violation_count'] < 50].copy()
    
    # Convert coordinates to radians for the haversine distance metric
    coords = np.radians(high_density_cells[['latitude', 'longitude']])
    
    # HDBSCAN parameters: min_cluster_size=2 means a hotspot must have at least 2 nearby dense H3 cells
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='haversine')
    clusterer.fit(coords)
    
    high_density_cells['cluster_id'] = clusterer.labels_
    
    # Low density cells are automatically labeled as noise (-1)
    low_density_cells['cluster_id'] = -1
    
    # Recombine the dataset
    cell_df = pd.concat([high_density_cells, low_density_cells])
    
    # Save the clustering model
    with open('../models/hdbscan_model.pkl', 'wb') as f:
        pickle.dump(clusterer, f)
        
    num_clusters = cell_df['cluster_id'].nunique() - (1 if -1 in cell_df['cluster_id'].values else 0)
    print(f"Discovered {num_clusters} distinct spatial hotspots (excluding noise).")
    
    print("Loading full engineered data for temporal pattern discovery...")
    eng_df = pd.read_csv(eng_input)
    
    # Merge cluster_id back into the row-level data to analyze time patterns
    eng_df = pd.merge(eng_df, cell_df[['h3_cell_id', 'cluster_id']], on='h3_cell_id', how='inner')
    
    # Exclude noise points (cluster_id == -1)
    hotspot_df = eng_df[eng_df['cluster_id'] != -1]
    
    print("Stage 8: Computing Temporal Patterns for every hotspot...")
    hotspot_patterns = []
    
    for c_id, group in hotspot_df.groupby('cluster_id'):
        peak_hour = int(group['hour'].value_counts().idxmax())
        peak_day = int(group['day_of_week'].value_counts().idxmax())
        
        # Convert to datetime to extract accurate days and weeks
        datetime_col = pd.to_datetime(group['created_datetime'], format='mixed')
        unique_days = datetime_col.dt.date.nunique()
        total_violations = len(group)
        
        avg_daily = total_violations / unique_days if unique_days > 0 else 0
        weekly_persistence = datetime_col.dt.isocalendar().week.nunique()
        
        hourly_dist = group['hour'].value_counts().to_dict()
        
        hotspot_patterns.append({
            'cluster_id': c_id,
            'cluster_size_violations': total_violations,
            'unique_cells': group['h3_cell_id'].nunique(),
            'peak_violation_hour': peak_hour,
            'peak_violation_day': peak_day,
            'average_daily_violations': round(avg_daily, 2),
            'weekly_persistence': weekly_persistence,
            'hourly_distribution': json.dumps(hourly_dist)
        })
        
    pattern_df = pd.DataFrame(hotspot_patterns)
    pattern_df.to_csv(hotspot_output, index=False)
    
    # We also need to map the cluster_ids back to our zone_time_aggregated table for Stage 9
    agg_df = pd.merge(agg_df, cell_df[['h3_cell_id', 'cluster_id']], on='h3_cell_id', how='left')
    output_agg = agg_input.replace('.csv', '_clusters.csv')
    agg_df.to_csv(output_agg, index=False)
    
    print(f"Saved hotspot patterns to {hotspot_output}")
    print("Updated aggregated data with cluster IDs.")
    print("Task 5 completed successfully!")

if __name__ == "__main__":
    AGG_INPUT = "../features/zone_time_aggregated.csv"
    ENG_INPUT = "../features/engineered_data.csv"
    HOTSPOT_OUTPUT = "../features/hotspot_patterns.csv"
    
    print("=== Starting Task 5: Hotspot & Temporal Discovery ===")
    discover_hotspots(AGG_INPUT, ENG_INPUT, HOTSPOT_OUTPUT)
