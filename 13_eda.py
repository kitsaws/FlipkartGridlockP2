import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(input_file):
    print(f"Loading predictive time-series dataset from {input_file}...")
    df = pd.read_csv(input_file)
    active_events = df[df['violation_count'] > 0].copy()
    
    print("\n" + "="*50)
    print("1. FEATURE CORRELATION ANALYSIS (Signal vs Noise)")
    print("="*50)
    
    # We will compute the correlation of every numeric feature against the violation_count (Severity)
    # and against the binary is_hotspot label.
    
    features = [
        'violation_count', 'is_hotspot', 'hour', 'day_of_week', 'is_weekend', 'month',
        'count_last_week', 'count_yesterday', 'rolling_3d_mean', 'rolling_7d_mean'
    ]
    
    corr_matrix = active_events[features].corr()
    
    print("\n--- Correlation with Raw Severity (Violation Count) ---")
    print(corr_matrix['violation_count'].sort_values(ascending=False))
    
    print("\n--- Correlation with Hotspot Probability ---")
    print(corr_matrix['is_hotspot'].sort_values(ascending=False))
    
    print("\n" + "="*50)
    print("2. TEMPORAL DISTRIBUTIONS")
    print("="*50)
    
    # Hotspots by hour
    hotspots = df[df['is_hotspot'] == 1]
    print("\n--- Top 5 Worst Hours for Hotspots ---")
    print(hotspots['hour'].value_counts().head(5))
    
    print("\n--- Top 3 Worst Days for Hotspots (0=Mon, 6=Sun) ---")
    print(hotspots['day_of_week'].value_counts().head(3))
    
    os.makedirs('outputs/eda', exist_ok=True)
    
    # 1. Severity Distribution (Why R-Squared is failing)
    plt.figure(figsize=(10, 6))
    sns.histplot(active_events['violation_count'], bins=50, kde=True, color='red')
    plt.title("Distribution of Violation Counts (Severity)")
    plt.xlabel("Number of Violations in an Hour")
    plt.ylabel("Frequency")
    plt.savefig('outputs/eda/1_severity_distribution.png')
    plt.close()
    
    # 2. Hotspots by Hour
    plt.figure(figsize=(10, 6))
    sns.countplot(data=hotspots, x='hour', palette='Reds_r')
    plt.title("Hotspot Occurrences by Hour of Day")
    plt.savefig('outputs/eda/2_hotspots_by_hour.png')
    plt.close()
    
    # 3. Hotspots by Day of Week
    plt.figure(figsize=(10, 6))
    sns.countplot(data=hotspots, x='day_of_week', palette='Oranges_r')
    plt.title("Hotspot Occurrences by Day of Week (0=Mon, 6=Sun)")
    plt.savefig('outputs/eda/3_hotspots_by_day.png')
    plt.close()
    
    # 4. Correlation Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.savefig('outputs/eda/4_correlation_heatmap.png')
    plt.close()
    
    print("\nTask Completed: EDA Charts saved to outputs/eda/")

if __name__ == "__main__":
    INPUT = "features/predictive_time_series.csv"
    run_eda(INPUT)
