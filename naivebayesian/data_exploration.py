import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

def explore_data():
    # Find all data files in the data directory
    data_files = glob('data/*')
    
    print(f"Found {len(data_files)} files in the data directory:")
    for file in data_files:
        print(f"- {file}")
        # Try to read the file and display basic info
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.endswith('.xlsx'):
                df = pd.read_excel(file)
            elif file.endswith('.json'):
                df = pd.read_json(file)
            else:
                print(f"  Skipping file with unsupported format: {file}")
                continue
                
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")
            print(f"  Data types:\n{df.dtypes}")
            print(f"  Missing values: {df.isnull().sum().sum()}")
            print(f"  Sample data:\n{df.head()}")
            
            # Enhanced exploration
            visualize_data(df, os.path.basename(file).split('.')[0])
            
        except Exception as e:
            print(f"  Error reading file: {str(e)}")

def visualize_data(df, filename):
    """Generate visualizations for the dataset."""
    print("\n=== Generating visualizations ===")
    
    # Create output directory for visualizations
    os.makedirs('visualizations', exist_ok=True)
    
    # Summary statistics
    print("Summary statistics:")
    print(df.describe())
    
    # Check for class distribution if last column is categorical
    try:
        last_col = df.columns[-1]
        if df[last_col].dtype == 'object' or len(df[last_col].unique()) < 10:
            print(f"\nClass distribution for '{last_col}':")
            class_dist = df[last_col].value_counts()
            print(class_dist)
            
            # Plot class distribution
            plt.figure(figsize=(10, 6))
            sns.countplot(y=df[last_col], order=class_dist.index)
            plt.title(f'Class Distribution - {filename}')
            plt.tight_layout()
            plt.savefig(f'visualizations/{filename}_class_dist.png')
            plt.close()
    except:
        print("Could not analyze class distribution")
    
    # Correlation heatmap for numerical features
    try:
        numerical_df = df.select_dtypes(include=[np.number])
        if not numerical_df.empty:
            plt.figure(figsize=(12, 10))
            sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
            plt.title(f'Correlation Heatmap - {filename}')
            plt.tight_layout()
            plt.savefig(f'visualizations/{filename}_corr_heatmap.png')
            plt.close()
    except:
        print("Could not generate correlation heatmap")
    
    # Histograms of numerical features
    try:
        numerical_df = df.select_dtypes(include=[np.number])
        if not numerical_df.empty:
            numerical_df.hist(figsize=(14, 12), bins=20)
            plt.suptitle(f'Feature Distributions - {filename}')
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.savefig(f'visualizations/{filename}_histograms.png')
            plt.close()
    except:
        print("Could not generate histograms")

if __name__ == "__main__":
    explore_data()
