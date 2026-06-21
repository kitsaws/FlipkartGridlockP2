import pandas as pd
import folium
import h3

def generate_verification_map():
    print("Loading raw, unaggregated violation data...")
    raw_df = pd.read_csv('../cleaned/cleaned_data.csv')
    
    # Take a random sample of 10,000 actual violation points so we don't crash the browser
    sample_points = raw_df.dropna(subset=['latitude', 'longitude']).sample(10000, random_state=42)
    
    print("Loading our Top 50 Computed Hotspots...")
    top_50 = pd.read_csv('../outputs/actionable_enforcement_targets.csv')
    
    lats, lngs = [], []
    for cell in top_50['h3_cell_id'].unique():
        try:
            lat, lng = h3.cell_to_latlng(cell)
            lats.append(lat)
            lngs.append(lng)
        except:
            continue
            
    # Fallback to Bangalore coordinates if we can't get center
    if not lats:
        lats, lngs = [12.9716], [77.5946]
        
    center_lat, center_lng = sum(lats)/len(lats), sum(lngs)/len(lngs)
    
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13, tiles='CartoDB dark_matter')
    
    # 1. Draw our Mathematical H3 Cells (Red Hexagons)
    for _, row in top_50.iterrows():
        cell = row['h3_cell_id']
        boundary = h3.cell_to_boundary(cell)
        folium.Polygon(
            locations=boundary,
            color='#ff4b4b',
            weight=2,
            fill=True,
            fill_color='#ff0000',
            fill_opacity=0.3
        ).add_to(m)
        
    # 2. Draw the literal raw data coordinates (Blue Dots)
    print("Plotting raw data points over the map...")
    for _, row in sample_points.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=1, # Very tiny dot
            color='#00d4ff', # Neon blue
            fill=True,
            fill_opacity=1.0
        ).add_to(m)
        
    m.save('../outputs/verification_map.html')
    print("Saved Verification Map to outputs/verification_map.html")

if __name__ == "__main__":
    print("=== Generating Geographic Output Verification Map ===")
    generate_verification_map()
