import pandas as pd
import folium
import h3
import os
import matplotlib.pyplot as plt
import seaborn as sns

def create_deliverables(input_file):
    print(f"Loading final prioritized data from {input_file}...")
    df = pd.read_csv(input_file)
    
    os.makedirs('outputs', exist_ok=True)
    
    # Deliverable 2: Top N Enforcement Targets
    print("\nStage 12: Generating Top 50 Enforcement Targets...")
    # The Priority Engine outputs "Zone-Time" blocks. So the same physical zone might be in the Top 50 multiple times (e.g., at 2 AM, 3 AM, 4 AM).
    # To give authorities 50 UNIQUE physical locations, we will sort by priority and drop duplicate zones, keeping the worst hour.
    unique_zones = df.sort_values('priority_score', ascending=False).drop_duplicates(subset=['h3_cell_id'])
    top_50 = unique_zones.head(50).copy()
    
    top_50.to_csv('outputs/actionable_enforcement_targets.csv', index=False)
    print("Saved top targets to outputs/actionable_enforcement_targets.csv")
    
    # Deliverable 1: Dynamic Hotspot Heatmap
    print("\nGenerating Geographic Heatmap (Folium)...")
    
    # We want to center the map on the average coordinate of our top targets
    lats, lngs = [], []
    for cell in top_50['h3_cell_id'].unique():
        try:
            lat, lng = h3.cell_to_latlng(cell)
            lats.append(lat)
            lngs.append(lng)
        except:
            continue
            
    if lats and lngs:
        center_lat, center_lng = sum(lats)/len(lats), sum(lngs)/len(lngs)
    else:
        # Fallback coordinate if something goes wrong
        center_lat, center_lng = 12.9716, 77.5946 # Default Bangalore
    
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13, tiles='CartoDB dark_matter')
    
    # To draw the heatmap, we will color the H3 Hexagons by their Priority Score
    max_score = df['priority_score'].max()
    
    # We group by the cell to get the MAX priority score across all hours
    # This shows the "worst-case" scenario for each geographical zone
    cell_priority = df.groupby('h3_cell_id').agg({
        'priority_score': 'max',
        'violation_count': 'sum'
    }).reset_index()
    
    # Only plot the top 500 cells to prevent the HTML file from becoming too large/slow
    cell_priority = cell_priority.sort_values(by='priority_score', ascending=False).head(500)
    
    for _, row in cell_priority.iterrows():
        cell = row['h3_cell_id']
        score = row['priority_score']
        v_count = row['violation_count']
        
        try:
            # Get hexagon boundary coordinates
            boundary = h3.cell_to_boundary(cell)
            
            # Color mapping: Intensity scales with the priority score
            intensity = score / max_score
            
            # Create the polygon
            folium.Polygon(
                locations=boundary,
                color='#ff4b4b', # Red outline
                weight=1,
                fill=True,
                fill_color='#ff0000', # Solid red fill
                fill_opacity=intensity * 0.8, # Opacity scales with priority!
                tooltip=f"<b>H3 Cell:</b> {cell}<br><b>Max Priority:</b> {score:.1f}<br><b>Total Violations:</b> {v_count}"
            ).add_to(m)
        except:
            continue
            
    m.save('outputs/hotspot_heatmap.html')
    print("Saved Interactive Geographic Heatmap to outputs/hotspot_heatmap.html")
    
    # Deliverable 3: Temporal Analytics
    print("\nGenerating Temporal Analytics Chart...")
    plt.figure(figsize=(12, 6))
    
    # Group the total violations by hour
    hourly_violations = df.groupby('hour')['violation_count'].sum().reset_index()
    
    sns.barplot(data=hourly_violations, x='hour', y='violation_count', color='#ff4b4b', alpha=0.8)
    
    plt.title('Total Violations by Hour Across All Hotspots', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Hour of Day (0-23)', fontsize=12, fontweight='bold')
    plt.ylabel('Total Violations', fontsize=12, fontweight='bold')
    
    # Add a subtle grid
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Remove top and right borders for a cleaner look
    sns.despine()
    
    plt.tight_layout()
    plt.savefig('outputs/hourly_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved Hourly Trends Chart to outputs/hourly_trends.png")
    print("\nTask 8 completed successfully! All deliverables are in the 'outputs/' folder.")

if __name__ == "__main__":
    INPUT = "features/enforcement_priority.csv"
    create_deliverables(INPUT)
