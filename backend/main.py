from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import h3
import os
import subprocess
import sys

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
DATA_PATH = os.path.join(BASE_DIR, 'features', 'predictive_time_series_minimized.parquet')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'risk_classifier.pkl')

print("Loading minimized dataset...")
df = pd.read_parquet(DATA_PATH)

print("Loading model...")
model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)
high_idx = list(model.classes_).index('High')

# Extract top primary violations
top_vios = df['primary_violation'].value_counts().head(4).index.tolist()
if "None" in top_vios: top_vios.remove("None")
available_dates = sorted(df['date'].unique())
forecast_dates = available_dates[-7:]

training_status = {"status": "idle", "step": ""}

def run_training_pipeline():
    global df, model, explainer, high_idx, top_vios, available_dates, forecast_dates, training_status
    training_status["status"] = "running"
    scripts_to_run = [
        ("01_dataset_inspection.py", "Data Inspection"),
        ("02_data_cleaning.py", "Data Cleaning"),
        ("03_feature_engineering.py", "Feature Engineering"),
        ("04_aggregation.py", "Basic Aggregation"),
        ("10_time_series_aggregation.py", "Time Series Zero-Inflation"),
        ("11_risk_classification_model.py", "Training Random Forest"),
    ]
    scripts_dir = os.path.join(BASE_DIR, "scripts")
    
    try:
        for script, step_name in scripts_to_run:
            training_status["step"] = step_name
            script_path = os.path.join(scripts_dir, script)
            subprocess.run([sys.executable, script_path], cwd=scripts_dir, check=True)
            
        training_status["step"] = "Minimizing Dataset"
        minimize_script = os.path.join(BASE_DIR, "minimize_data.py")
        subprocess.run([sys.executable, minimize_script], cwd=BASE_DIR, check=True)
        
        training_status["step"] = "Reloading Models"
        df = pd.read_parquet(DATA_PATH)
        model = joblib.load(MODEL_PATH)
        explainer = shap.TreeExplainer(model)
        high_idx = list(model.classes_).index('High')
        
        new_top_vios = df['primary_violation'].value_counts().head(4).index.tolist()
        if "None" in new_top_vios: new_top_vios.remove("None")
        top_vios = new_top_vios
        
        available_dates = sorted(df['date'].unique())
        forecast_dates = available_dates[-7:]
        
        training_status["status"] = "idle"
        training_status["step"] = "Complete"
    except Exception as e:
        training_status["status"] = "error"
        training_status["step"] = f"Failed at {training_status['step']}: {str(e)}"

class PredictRequest(BaseModel):
    date: str
    hour: int
    user_weights: dict

@app.get("/api/config")
def get_config():
    return {
        "dates": forecast_dates,
        "top_violations": top_vios,
        "status": training_status
    }

@app.get("/api/retrain/status")
def get_retrain_status():
    return training_status

@app.post("/api/retrain")
async def start_retrain(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if training_status["status"] == "running":
        return {"message": "Training already in progress."}
        
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "raw_data.csv")
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    background_tasks.add_task(run_training_pipeline)
    return {"message": "Pipeline started successfully"}

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
