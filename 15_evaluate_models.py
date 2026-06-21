import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def evaluate_all():
    print("Loading Dataset...")
    df = pd.read_csv('features/predictive_time_series.csv')
    
    # Preprocessing
    cell_avg = df.groupby('h3_cell_id')['weighted_severity_score'].mean().reset_index(name='historical_avg_severity')
    df = pd.merge(df, cell_avg, on='h3_cell_id', how='left')
    
    print("\n" + "="*60)
    print("MODEL 1: V2 MULTI-CLASS HOTSPOT RISK CLASSIFIER")
    print("="*60)
    
    features_clf = [
        'hour', 'day_of_week', 'is_weekend', 'month',
        'severity_last_week', 'severity_yesterday', 
        'rolling_3d_severity', 'rolling_7d_severity',
        'historical_avg_severity'
    ]
    
    high = df[df['Risk_Class'] == 'High']
    med = df[df['Risk_Class'] == 'Medium']
    
    if len(high) == 0 or len(med) == 0:
        print("Classes missing.")
        return
        
    low = df[df['Risk_Class'] == 'Low'].sample(n=len(high) * 3, random_state=42)
    bal_df = pd.concat([high, med, low])
    
    X_clf = bal_df[features_clf]
    y_clf = bal_df['Risk_Class']
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    clf = RandomForestClassifier(n_estimators=50, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_train_c, y_train_c)
    
    y_pred_c = clf.predict(X_test_c)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test_c, y_pred_c))
    
    print("\n--- Confusion Matrix ---")
    # Pretty print confusion matrix
    cm = pd.crosstab(y_test_c, y_pred_c, rownames=['Actual'], colnames=['Predicted'])
    print(cm)
    
    try:
        auc = roc_auc_score(y_test_c, clf.predict_proba(X_test_c), multi_class='ovr')
        print(f"\nMulti-Class ROC-AUC Score (OVR): {auc:.4f}")
    except Exception as e:
        pass
        
    print("\n" + "="*60)
    print("MODEL 2: SEVERITY REGRESSOR")
    print("="*60)
    print("Status: DEPRECATED")
    print("Reason: Replaced entirely by the Multi-Class Risk prediction per V2 architecture.")
    print("\n")

if __name__ == '__main__':
    evaluate_all()
