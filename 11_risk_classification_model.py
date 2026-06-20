import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

def train_classifier(input_file, model_output_path):
    print(f"Loading predictive time-series dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    print("Stage 11.1: Feature Engineering (Spatial Target Encoding)...")
    # Calculate historical average occupancy per cell to serve as a core Spatial Feature
    cell_avg = df.groupby('h3_cell_id')['violation_count'].mean().reset_index(name='historical_avg_occupancy')
    df = pd.merge(df, cell_avg, on='h3_cell_id', how='left')
    
    features = [
        'hour', 'day_of_week', 'is_weekend', 'month',
        'count_last_week', 'count_yesterday', 
        'rolling_3d_mean', 'rolling_7d_mean',
        'historical_avg_occupancy'
    ]
    
    X = df[features]
    y = df['is_hotspot']
    
    # Check class imbalance
    hotspot_ratio = y.mean() * 100
    print(f"Class Distribution: {hotspot_ratio:.2f}% Hotspots, {100-hotspot_ratio:.2f}% Normal")
    
    print("Stage 11.2: Balancing Dataset (Prioritizing Recall)...")
    # The dataset is massive (2.6M) and heavily imbalanced (mostly zeros due to zero-inflation).
    # We will undersample the negative class to train faster and heavily bias the model toward high Recall.
    hotspots = df[df['is_hotspot'] == 1]
    
    # If there are extremely few hotspots, just use all data, otherwise sample 3:1
    if len(hotspots) == 0:
        print("ERROR: No hotspots detected. Threshold may be too high.")
        return
        
    normals = df[df['is_hotspot'] == 0].sample(n=len(hotspots) * 3, random_state=42) # 3:1 ratio
    
    balanced_df = pd.concat([hotspots, normals])
    X_bal = balanced_df[features]
    y_bal = balanced_df['is_hotspot']
    
    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal)
    
    print(f"Stage 11.3: Training RandomForestClassifier on {len(X_train)} samples...")
    # class_weight='balanced' pushes the model to prioritize the minority class, increasing Recall
    model = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("\n=== Model Evaluation (Test Set) ===")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred, target_names=["Normal", "Hotspot"]))
    
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc:.4f}")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"\nTask 11 Completed: Saved Hotspot Risk Classifier to {model_output_path}")

if __name__ == "__main__":
    INPUT = "features/predictive_time_series.csv"
    OUTPUT = "models/risk_classifier.pkl"
    
    print("=== Starting Task 11: Hotspot Risk Classification ===")
    train_classifier(INPUT, OUTPUT)
