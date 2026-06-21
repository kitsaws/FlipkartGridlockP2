from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import h3
import os

app = FastAPI()

# Enable CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'features', 'predictive_time_series.parquet')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'risk_classifier.pkl')

print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)
cell_avg = df.groupby('h3_cell_id')['weighted_severity_score'].mean().reset_index(name='historical_avg_severity')
df = pd.merge(df, cell_avg, on='h3_cell_id', how='left')

print("Loading model...")
model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)
high_idx = list(model.classes_).index('High')

# Extract top primary violations
top_vios = df['primary_violation'].value_counts().head(4).index.tolist()
if "None" in top_vios: top_vios.remove("None")
available_dates = sorted(df['date'].unique())
forecast_dates = available_dates[-7:]

class PredictRequest(BaseModel):
    date: str
    hour: int
    user_weights: dict

@app.get("/api/config")
def get_config():
    return {
        "dates": forecast_dates,
        "top_violations": top_vios
    }

@app.post("/api/predict")
def predict_hotspots(req: PredictRequest):
    hour_data = df[(df['date'] == req.date) & (df['hour'] == req.hour)].copy()
    if hour_data.empty:
        return {"polygons": []}

    features = [
        'hour', 'day_of_week', 'is_weekend', 'month',
        'severity_last_week', 'severity_yesterday', 
        'rolling_3d_severity', 'rolling_7d_severity',
        'historical_avg_severity'
    ]
    
    X = hour_data[features].fillna(0)
    probs = model.predict_proba(X)
    base_high_prob = probs[:, high_idx] * 100
    
    polygons = []
    
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
    
    for i, (idx, row) in enumerate(hour_data.iterrows()):
        vio = row['primary_violation']
        base_prob = base_high_prob[i]
        
        # Scale probability
        scale_factor = 1.0
        if vio in req.user_weights:
            scale_factor = req.user_weights[vio] / 3.0
            
        adj_prob = base_prob * scale_factor
        
        if adj_prob < 25:
            continue # Low risk
            
        risk_class = 'High' if adj_prob >= 50 else 'Medium'
        
        # Calculate SHAP if High
        reasons = []
        if risk_class == 'High':
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
            
            for f, imp in contribs[:3]:
                if imp > 0:
                    reasons.append(human_text.get(f, f))
        
        try:
            poly = h3.cell_to_boundary(row['h3_cell_id'])
        except:
            continue
            
        polygons.append({
            "id": row['h3_cell_id'],
            "coordinates": poly,
            "risk": risk_class,
            "probability": round(adj_prob, 1),
            "primary_violation": str(vio).title(),
            "primary_violation_pct": row['primary_violation_pct'],
            "reasons": reasons
        })
        
    return {"polygons": polygons}
