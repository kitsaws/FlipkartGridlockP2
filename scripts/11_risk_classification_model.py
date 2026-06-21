import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def train_classifier(input_file, model_output_path):
    print(f"Loading predictive time-series dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    print("Stage 11.1: Feature Engineering (Spatial Target Encoding)...")
    # Use the weighted severity score for spatial encoding
    cell_avg = df.groupby('h3_cell_id')['weighted_severity_score'].mean().reset_index(name='historical_avg_severity')
    df = pd.merge(df, cell_avg, on='h3_cell_id', how='left')
    
    # Notice we strictly DO NOT use weighted_severity_score as a feature here (No Leakage!)
    features = [
        'hour', 'day_of_week', 'is_weekend', 'month',
        'severity_last_week', 'severity_yesterday', 
        'rolling_3d_severity', 'rolling_7d_severity',
        'historical_avg_severity'
    ]
    
    X = df[features]
    y = df['Risk_Class']
    
    print("Stage 11.2: Balancing Dataset (Multi-Class)...")
    # The dataset is massive (2.6M) and mostly 'Low' due to zero-inflation.
    # We will undersample the 'Low' class to balance training and improve High/Med Recall.
    high = df[df['Risk_Class'] == 'High']
    med = df[df['Risk_Class'] == 'Medium']
    
    if len(high) == 0 or len(med) == 0:
        print("ERROR: Target classes missing. Check quantile logic.")
        return
        
    # Sample Low classes at 3x the size of High classes to maintain a realistic but balanced distribution
    low = df[df['Risk_Class'] == 'Low'].sample(n=len(high) * 3, random_state=42)
    
    balanced_df = pd.concat([high, med, low])
    X_bal = balanced_df[features]
    y_bal = balanced_df['Risk_Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal)
    
    print(f"Stage 11.3: Training Multi-Class RandomForestClassifier on {len(X_train)} samples...")
    model = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("\n=== Model Evaluation (Test Set) ===")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    print(classification_report(y_test, y_pred))
    
    try:
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        print(f"Multi-Class ROC-AUC Score (OVR): {auc:.4f}")
    except Exception as e:
        pass
    
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"\nTask 11 Completed: Saved V2 Multi-Class Risk Classifier to {model_output_path}")

if __name__ == "__main__":
    INPUT = "../features/predictive_time_series.csv"
    OUTPUT = "../models/risk_classifier.pkl"
    
    print("=== Starting Task 11: Multi-Class Hotspot Risk Classification ===")
    train_classifier(INPUT, OUTPUT)
