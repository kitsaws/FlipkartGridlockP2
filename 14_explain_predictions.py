import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_explanations(data_file, model_file):
    print(f"Loading Predictive Dataset: {data_file}")
    df = pd.read_csv(data_file)
    
    print(f"Loading V2 Multi-Class Classifier: {model_file}")
    model = joblib.load(model_file)
    
    cell_avg = df.groupby('h3_cell_id')['weighted_severity_score'].mean().reset_index(name='historical_avg_severity')
    df = pd.merge(df, cell_avg, on='h3_cell_id', how='left')
    
    features = [
        'hour', 'day_of_week', 'is_weekend', 'month',
        'severity_last_week', 'severity_yesterday', 
        'rolling_3d_severity', 'rolling_7d_severity',
        'historical_avg_severity'
    ]
    
    # Isolate true HIGH RISK hotspots to explain WHY they happen
    hotspots = df[df['Risk_Class'] == 'High'].copy()
    if len(hotspots) == 0:
        print("No High Risk cells found to explain!")
        return
        
    X_hotspots = hotspots[features]
    
    # Take a sample for SHAP
    sample_size = min(500, len(X_hotspots))
    X_sample = X_hotspots.sample(n=sample_size, random_state=42)
    
    print("\nStage 13.1: Running SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    demo_idx = 0
    demo_row = X_sample.iloc[demo_idx]
    
    # Find which index maps to 'High' class
    high_class_idx = list(model.classes_).index('High')
    
    # Safely extract raw numpy arrays (newer SHAP versions return 'Explanation' objects)
    raw_shap = shap_values.values if hasattr(shap_values, 'values') else shap_values
    
    # Handle multi-class SHAP dimensions
    if isinstance(raw_shap, list):
        shap_values_high = raw_shap[high_class_idx]
        demo_shap = shap_values_high[demo_idx]
    elif len(raw_shap.shape) == 3:
        shap_values_high = raw_shap[:, :, high_class_idx]
        demo_shap = raw_shap[demo_idx, :, high_class_idx]
    else:
        shap_values_high = raw_shap
        demo_shap = raw_shap[demo_idx]
        
    os.makedirs('outputs/explainability', exist_ok=True)
    
    # Generate Summary Plot for High Risk
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_high, X_sample, show=False)
    plt.title("What Drives HIGH Hotspot Risk? (SHAP Explainability)")
    plt.tight_layout()
    plt.savefig('outputs/explainability/v2_shap_summary.png')
    plt.close()
    
    print("Saved SHAP summary to outputs/explainability/v2_shap_summary.png")
    
    # Output the required 'Query 3' text format from instructions
    print("\n" + "="*50)
    print("QUERY 3 DEMO: Why was this prediction made?")
    print("="*50)
    
    # Calculate Risk Probability for the High class
    prob = model.predict_proba(demo_row.to_frame().T)[0][high_class_idx] * 100
    
    print(f"Location: {hotspots.loc[X_sample.index[demo_idx]]['h3_cell_id']}")
    print(f"Time: Day {int(demo_row['day_of_week'])}, Hour {int(demo_row['hour'])}")
    print(f"\nHIGH Hotspot Risk: {prob:.1f}%")
    print("\nMain Contributors (SHAP):")
    
    # Match features with their specific SHAP contribution for this single prediction
    contributions = list(zip(features, demo_shap))
    # Sort by absolute impact
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for feat, impact in contributions[:3]: # Top 3 contributors
        direction = "+" if impact > 0 else "-"
        print(f"[{direction}] {feat} (Value: {demo_row[feat]:.1f})")

if __name__ == "__main__":
    DATA = "features/predictive_time_series.csv"
    MODEL = "models/risk_classifier.pkl"
    
    print("=== Starting Task 13: Multi-Class Predictive Explainability ===")
    generate_explanations(DATA, MODEL)
