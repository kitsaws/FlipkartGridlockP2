import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def evaluate_all():
    print("Loading Dataset...")
    df = pd.read_csv('features/predictive_time_series.csv')
    
    # Preprocessing
    cell_avg = df.groupby('h3_cell_id')['violation_count'].mean().reset_index(name='historical_avg_occupancy')
    df = pd.merge(df, cell_avg, on='h3_cell_id', how='left')
    
    print("\n" + "="*60)
    print("MODEL 1: HOTSPOT RISK CLASSIFIER (Upgrade 2)")
    print("="*60)
    
    features_clf = [
        'hour', 'day_of_week', 'is_weekend', 'month',
        'count_last_week', 'count_yesterday', 
        'rolling_3d_mean', 'rolling_7d_mean',
        'historical_avg_occupancy'
    ]
    
    hotspots = df[df['is_hotspot'] == 1]
    normals = df[df['is_hotspot'] == 0].sample(n=len(hotspots) * 3, random_state=42)
    bal_df = pd.concat([hotspots, normals])
    
    X_clf = bal_df[features_clf]
    y_clf = bal_df['is_hotspot']
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    clf = RandomForestClassifier(n_estimators=50, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_train_c, y_train_c)
    
    y_pred_c = clf.predict(X_test_c)
    print(classification_report(y_test_c, y_pred_c, target_names=["Normal", "Hotspot"]))
    print(f"ROC-AUC Score: {roc_auc_score(y_test_c, clf.predict_proba(X_test_c)[:, 1]):.4f}")
    
    
    print("\n" + "="*60)
    print("MODEL 2A: SEVERITY REGRESSOR (Strict Rules: Loc + Time Only)")
    print("="*60)
    
    active = df[df['violation_count'] > 0].copy()
    loc_enc = active.groupby('h3_cell_id')['violation_count'].mean().reset_index(name='location_encoding')
    active = pd.merge(active, loc_enc, on='h3_cell_id', how='left')
    
    features_reg_strict = ['location_encoding', 'hour', 'day_of_week', 'is_weekend', 'month']
    X_reg1 = active[features_reg_strict]
    y_reg = np.log1p(active['violation_count'])
    
    X_train_r1, X_test_r1, y_train_r, y_test_r = train_test_split(X_reg1, y_reg, test_size=0.2, random_state=42)
    reg1 = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    reg1.fit(X_train_r1, y_train_r)
    
    y_pred_r1 = reg1.predict(X_test_r1)
    print(f"Log R-Squared  : {r2_score(y_test_r, y_pred_r1):.4f}")
    print(f"MAE (Vehicles) : {mean_absolute_error(np.expm1(y_test_r), np.expm1(y_pred_r1)):.2f}")
    
    print("\n" + "="*60)
    print("MODEL 2B: SEVERITY REGRESSOR (With Momentum Features Added)")
    print("="*60)
    
    features_reg_full = ['location_encoding', 'hour', 'day_of_week', 'is_weekend', 'month', 
                         'count_last_week', 'count_yesterday', 'rolling_3d_mean', 'rolling_7d_mean']
    X_reg2 = active[features_reg_full]
    
    X_train_r2, X_test_r2, _, _ = train_test_split(X_reg2, y_reg, test_size=0.2, random_state=42)
    reg2 = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    reg2.fit(X_train_r2, y_train_r)
    
    y_pred_r2 = reg2.predict(X_test_r2)
    print(f"Log R-Squared  : {r2_score(y_test_r, y_pred_r2):.4f}")
    print(f"MAE (Vehicles) : {mean_absolute_error(np.expm1(y_test_r), np.expm1(y_pred_r2)):.2f}")
    print("\n")
    
if __name__ == '__main__':
    evaluate_all()
