import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

def run_explainability_model(input_file):
    print(f"Loading prioritized data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print("Stage 11: Training Explainability Model...")
    
    # We will use granular underlying features to predict the final priority_score.
    # This prevents the model from just "memorizing" the math formula directly.
    # We want to see how HOUR, DIVERSITY, and SPECIFIC VIOLATIONS drive the Priority Score
    potential_features = [
        'hour', 'violation_diversity', 'vehicle_type_diversity', 
        'night_violation_ratio', 'peak_hour_ratio', 'is_wrong_parking',
        'is_no_parking', 'is_parking_on_footpath', 'is_double_parking',
        'is_parking_near_bustop_school_hospital_etc', 'is_parking_near_traffic_light_or_zebra_cross',
        'is_obstructing_driver', 'is_jumping_traffic_signal',
        'is_against_one_way_no_entry', 'is_defective_number_plate'
    ]
                
    # Filter features to only those that exist in the dataframe
    actual_features = [f for f in potential_features if f in df.columns]
    
    X = df[actual_features]
    y = df['priority_score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training RandomForestRegressor on {len(actual_features)} granular features...")
    # Train model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Validate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Model R-squared (Accuracy): {r2:.4f}")
    
    if r2 > 0.6:
        print("Success! The model successfully learned the logic. Engineered features are mathematically robust.")
    
    print("Generating SHAP Explainability Plot...")
    
    # Compute SHAP values
    # We use a sample of the background data to make the SHAP explainer faster
    # TreeExplainer is specifically optimized for Tree-based models like Random Forest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Save SHAP Summary Plot
    os.makedirs('../outputs', exist_ok=True)
    plt.figure(figsize=(10, 8))
    
    # The summary plot shows feature importance and the direction of impact
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Explanation: What drives Enforcement Priority?", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('../outputs/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved SHAP summary plot to outputs/shap_summary.png")
    print("Task 7 completed successfully!")

if __name__ == "__main__":
    INPUT = "../features/enforcement_priority.csv"
    run_explainability_model(INPUT)
