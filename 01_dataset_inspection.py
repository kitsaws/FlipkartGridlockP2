import os
import pandas as pd
import numpy as np

def create_project_structure():
    """Creates the necessary folder structure for the project."""
    folders = [
        "data", "reports", "cleaned", "features", 
        "models", "outputs", "dashboard"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Ensured folder exists: {folder}/")

def inspect_dataset(file_path, output_report_path):
    """Loads the dataset, inspects features, and generates a summary report."""
    print(f"\nLoading dataset from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    except FileNotFoundError:
        print(f"Error: Could not find the dataset at {file_path}.")
        return

    print("\nGenerating dataset summary...")
    
    summary_data = []
    
    for col in df.columns:
        dtype = df[col].dtype
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        unique_count = df[col].nunique()
        
        # Grab a sample value if possible for context
        sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
        
        summary_data.append({
            "column_name": col,
            "datatype": str(dtype),
            "percentage_missing": round(missing_pct, 2),
            "unique_value_count": unique_count,
            "sample_value": sample_val
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save the summary as a CSV for easy viewing
    summary_df.to_csv(output_report_path, index=False)
    print(f"\nDataset summary report saved to {output_report_path}")
    
    # Print a text-based preview
    print("\n=== Dataset Summary Preview ===")
    print(summary_df.to_string())
    print("===============================\n")

if __name__ == "__main__":
    # Define paths
    DATASET_PATH = "DATASET/jan to may police violation_anonymized791b166.csv"
    REPORT_PATH = "reports/dataset_summary.csv"
    
    print("=== Starting Task 0 & 1: Project Initialization & Dataset Inspection ===")
    # Task 0: Create project structure
    create_project_structure()
    
    # Task 1: Inspect dataset
    inspect_dataset(DATASET_PATH, REPORT_PATH)
    print("\nNext step: Please review the 'reports/dataset_summary.csv' to decide on data cleaning steps.")
