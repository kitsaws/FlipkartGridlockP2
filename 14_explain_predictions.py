import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

def generate_explanations(data_file, model_file):
    print(f"Loading Predictive Dataset: {data_file}")
    df = pd.read_csv(data_file)
    
    print(f"Loading Hotspot Risk Classifier: {model_file}")
    model = joblib.load(model_file)
    
    # We need to recreate the exact feature set the model was trained on
    cell_avg = df.groupby('h3_cell_id')['violation_count'].mean().reset_index(name='historical_avg_occupancy')
    df = pd.merge(df, cell_avg, on='h3_cell_id', how='left')
    
    features = [
        'hour', 'day_of_week', 'is_weekend', 'month',
        'count_last_week', 'count_yesterday', 
        'rolling_3d_mean', 'rolling_7d_mean',
        'historical_avg_occupancy'
    ]
    
    # Isolate true hotspots to explain WHY they happen
    hotspots = df[df['is_hotspot'] == 1].copy()
    X_hotspots = hotspots[features]
    
    # Take a sample for SHAP (it is computationally heavy)
    sample_size = min(500, len(X_hotspots))
    X_sample = X_hotspots.sample(n=sample_size, random_state=42)
    
    print("\nStage 13.1: Running SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    demo_idx = 0
    
    # Safely extract raw numpy arrays (newer SHAP versions return 'Explanation' objects)
    raw_shap = shap_values.values if hasattr(shap_values, 'values') else shap_values
    
    # Handle different SHAP versions returning different numpy array shapes for Classifiers
    if isinstance(raw_shap, list):
        shap_values_hotspot = raw_shap[1]
        demo_shap = shap_values_hotspot[demo_idx]
    elif len(raw_shap.shape) == 3:
        shap_values_hotspot = raw_shap[:, :, 1]
        demo_shap = raw_shap[demo_idx, :, 1]
    else:
        shap_values_hotspot = raw_shap
        demo_shap = raw_shap[demo_idx]
        
    os.makedirs('outputs/explainability', exist_ok=True)
    
    # Generate Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_hotspot, X_sample, show=False)
    plt.title("What Drives Hotspot Risk? (SHAP Explainability)")
    plt.tight_layout()
    plt.savefig('outputs/explainability/predictive_shap_summary.png')
    plt.close()
    
    print("Saved SHAP summary to outputs/explainability/predictive_shap_summary.png")
    
    # Output the required 'Query 3' text format from instructions
    print("\n" + "="*50)
    print("QUERY 3 DEMO: Why was this prediction made?")
    print("="*50)
    
    demo_row = X_sample.iloc[demo_idx]
    
    # Calculate Risk Probability
    prob = model.predict_proba(demo_row.to_frame().T)[0][1] * 100
    
    print(f"Location: {hotspots.loc[X_sample.index[demo_idx]]['h3_cell_id']}")
    print(f"Time: Day {int(demo_row['day_of_week'])}, Hour {int(demo_row['hour'])}")
    print(f"\nHotspot Risk: {prob:.1f}%")
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
    
    print("=== Starting Task 13: Predictive Explainability ===")
    generate_explanations(DATA, MODEL)
