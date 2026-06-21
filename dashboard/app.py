import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib
import shap
import h3
import os

# 1. Page Configuration
st.set_page_config(page_title="GridLock AI Dashboard", layout="wide")

# Paths logic to load from parent directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'features', 'predictive_time_series.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'risk_classifier.pkl')

st.title("Spatio-Temporal Enforcement Priority Intelligence")
st.markdown("Predictive AI Engine mapping future parking gridlocks before they occur.")

# 2. Loading Data & Models (Cached for performance)
@st.cache_data(show_spinner="Loading 6 months of spatio-temporal data...")
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Pre-calculate spatial target encoding for speed
    cell_avg = df.groupby('h3_cell_id')['weighted_severity_score'].mean().reset_index(name='historical_avg_severity')
    df = pd.merge(df, cell_avg, on='h3_cell_id', how='left')
    return df

@st.cache_resource(show_spinner="Loading Risk Classifier AI...")
def load_model():
    return joblib.load(MODEL_PATH)

try:
    df = load_data()
    model = load_model()
except FileNotFoundError:
    st.error("Model or Data files not found. Ensure you have run Phase 2 scripts.")
    st.stop()

# Build the SHAP Explainer once for performance
explainer = shap.TreeExplainer(model)
high_idx = list(model.classes_).index('High')

# 3. Sidebar Controls (Time Travel Engine)
st.sidebar.header("Time Travel Engine")
st.sidebar.markdown("Slide forward in time to project future gridlocks.")

available_dates = sorted(df['date'].unique())
# Hackathon Trick: Treat the last 7 days of the dataset as the "Future Forecast"
forecast_dates = available_dates[-7:]
# Create user-friendly labels mapping the dataset date to a "Future" label
date_labels = {
    forecast_dates[0]: "Today (Live Baseline)",
    forecast_dates[1]: "Tomorrow",
    forecast_dates[2]: "In 2 Days",
    forecast_dates[3]: "In 3 Days",
    forecast_dates[4]: "In 4 Days",
    forecast_dates[5]: "In 5 Days",
    forecast_dates[6]: "In 6 Days",
}
selected_date = st.sidebar.select_slider(
    "Select Future Date", 
    options=forecast_dates,
    format_func=lambda x: date_labels[x]
)
selected_hour = st.sidebar.slider("Select Target Hour", min_value=0, max_value=23, value=12)

st.sidebar.markdown("---")
st.sidebar.markdown("### Commander Control Panel")
st.sidebar.markdown("Dynamically scale AI predictions by adjusting the priority of specific violations.")

# Get top 4 most common primary violations from the dataset for the sliders
top_vios = df['primary_violation'].value_counts().head(4).index.tolist()
if "None" in top_vios: top_vios.remove("None")

user_weights = {}
with st.sidebar.form("commander_form"):
    for vio in top_vios:
        # Default is 3.0 (neutral scale)
        user_weights[vio] = st.slider(f"Priority: {vio.title()}", min_value=0.0, max_value=5.0, value=3.0, step=0.5)
    submit_button = st.form_submit_button("Apply Filters")

st.sidebar.markdown("---")
st.sidebar.markdown("### Risk Legend")
st.sidebar.markdown("**High Risk:** Severe gridlock predicted (Action Required).")
st.sidebar.markdown("**Medium Risk:** Emerging congestion.")
st.sidebar.markdown("**Low Risk:** Clear roads (Hidden from map).")

# 4. Inference Engine
# Filter data for the exact hour requested
hour_data = df[(df['date'] == selected_date) & (df['hour'] == selected_hour)].copy()

if hour_data.empty:
    st.warning("No data available for the selected time.")
else:
    features = [
        'hour', 'day_of_week', 'is_weekend', 'month',
        'severity_last_week', 'severity_yesterday', 
        'rolling_3d_severity', 'rolling_7d_severity',
        'historical_avg_severity'
    ]
    
    # Fill NAs for safety
    X = hour_data[features].fillna(0)
    
    # Run the AI Base Prediction
    hour_data['Predicted_Risk'] = model.predict(X)
    probs = model.predict_proba(X)
    base_high_prob = probs[:, high_idx] * 100
    
    # Apply Dynamic UI Scaling (The Commander Control Panel)
    def scale_prob(row, base_prob):
        vio = row['primary_violation']
        if vio in user_weights:
            scale_factor = user_weights[vio] / 3.0 # 3.0 is neutral
            return base_prob * scale_factor
        return base_prob # If it's a rare violation not in sliders, keep base probability
        
    # Scale probabilities and update Risk Class
    hour_data['High_Prob'] = [scale_prob(row, p) for (_, row), p in zip(hour_data.iterrows(), base_high_prob)]
    hour_data['Predicted_Risk'] = hour_data.apply(lambda r: 'High' if r['High_Prob'] >= 50 else ('Medium' if r['High_Prob'] >= 25 else 'Low'), axis=1)
    
    # Calculate Display Metrics
    high_count = len(hour_data[hour_data['Predicted_Risk'] == 'High'])
    med_count = len(hour_data[hour_data['Predicted_Risk'] == 'Medium'])
    safe_count = len(hour_data) - high_count - med_count
    
    c1, c2, c3 = st.columns(3)
    c1.metric("High Risk Zones", high_count)
    c2.metric("Medium Risk Zones", med_count)
    c3.metric("Safe Zones", safe_count)
    
    st.markdown("---")
    
    # 5. Map Rendering
    if len(hour_data) > 0:
        # Default center map to Bangalore
        center_lat, center_lng = 12.9716, 77.5946
        m = folium.Map(location=[center_lat, center_lng], zoom_start=11, tiles="CartoDB dark_matter")
        
        # Filter to only show actionable intelligence (High/Med)
        visible_cells = hour_data[hour_data['Predicted_Risk'].isin(['High', 'Medium'])]
        
        for _, row in visible_cells.iterrows():
            cell_id = row['h3_cell_id']
            try:
                poly = h3.cell_to_boundary(cell_id)
            except:
                continue
                
            is_high = row['Predicted_Risk'] == 'High'
            color = "#FF0000" if is_high else "#FFA500" # Red or Orange
            fill_opacity = 0.6 if is_high else 0.3
            
            # 6. SHAP Explainability Engine
            popup_html = ""
            if is_high:
                # Dynamically calculate SHAP explanation for this exact cell on the fly
                x_val = row[features].to_frame().T
                shap_values = explainer.shap_values(x_val)
                raw_shap = shap_values.values if hasattr(shap_values, 'values') else shap_values
                
                if isinstance(raw_shap, list):
                    demo_shap = raw_shap[high_idx][0]
                elif len(raw_shap.shape) == 3:
                    demo_shap = raw_shap[0, :, high_idx]
                else:
                    demo_shap = raw_shap[0]
                    
                contribs = list(zip(features, demo_shap))
                contribs.sort(key=lambda x: abs(x[1]), reverse=True)
                
                # Format the popup
                popup_html = f"<div style='width:250px;'><b>HIGH RISK ({row['High_Prob']:.1f}%)</b><br><hr style='margin:5px 0;'/>"
                
                # Show Historical Profiler Output
                vio_name = str(row['primary_violation']).title()
                vio_pct = row['primary_violation_pct']
                if vio_name != "None":
                    popup_html += f"<b>Most Likely Cause:</b><br><span style='color:#FF5555'><b>{vio_name}</b></span><br>"
                    popup_html += f"<i>(Accounts for {vio_pct}% of historical incidents here)</i><br><br>"
                
                popup_html += "<b>AI Reason for Flagging:</b><br>"
                human_text = {
                    'rolling_3d_severity': 'Recent surge in dangerous parking (3-Day Trend)',
                    'rolling_7d_severity': 'Sustained escalation in violations (7-Day Trend)',
                    'historical_avg_severity': 'Location is a historically known gridlock zone',
                    'severity_yesterday': 'High volume of illegal parking recorded yesterday',
                    'severity_last_week': 'Recurring weekly pattern detected',
                    'hour': 'This specific time of day is historically dangerous here',
                    'is_weekend': 'Weekend traffic behavior',
                    'day_of_week': 'Weekday traffic behavior',
                    'month': 'Seasonal congestion patterns'
                }
                
                for f, imp in contribs[:3]:
                    # Only show factors that INCREASED the risk (positive SHAP impact)
                    if imp > 0:
                        popup_html += f"- {human_text.get(f, f)}<br>"
                popup_html += "</div>"
            else:
                vio_name = str(row['primary_violation']).title()
                vio_pct = row['primary_violation_pct']
                popup_html = "<div style='width:200px;'><b>Medium Risk</b><br>Emerging congestion.<hr style='margin:5px 0;'/>"
                if vio_name != "None":
                    popup_html += f"<b>Most Likely Cause:</b><br><span style='color:#FFA500'><b>{vio_name}</b></span><br>"
                    popup_html += f"<i>(Accounts for {vio_pct}% of historical incidents here)</i><br>"
                popup_html += "</div>"
                
            # Draw the Hexagon
            folium.Polygon(
                locations=poly,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=fill_opacity,
                weight=1,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)
            
        # Display in Streamlit (returned_objects=[] prevents jittery re-runs on map zoom)
        st_folium(m, width=1200, height=600, returned_objects=[])
