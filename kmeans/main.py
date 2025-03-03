import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.decomposition import PCA

# Read the dataset with proper delimiter
df = pd.read_csv('e-shop-clothing-2008.csv', delimiter=';')

# Basic information about the dataset
print(f"Dataset Shape: {df.shape} (rows, columns)")
print(f"Dataset Size: {df.size} elements")
print(f"Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB")

# Display first rows
print("\nFirst 5 rows:")
print(df.head())

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicates: {duplicates} rows")

# Column information
print("\nColumn Information:")
column_info = pd.DataFrame({
    'Data Type': df.dtypes,
    'Non-Null Count': df.count(),
    'Null Count': df.isnull().sum(),
    'Unique Values': [df[col].nunique() for col in df.columns]
})
print(column_info)

# Basic statistics
print("\nNumerical Statistics:")
print(df.describe())

# Check for missing values
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print("\nMissing Values:")
    print(missing_values[missing_values > 0])

# ----------------------
# Data Preprocessing for K-means Clustering
# ----------------------

print("\n\n--- K-means Clustering Analysis ---\n")

# For session-based analysis, we want to group data by session ID
session_df = df.copy()

# Feature Engineering: Creating session-level features
print("Creating session-level features...")

# Group by session ID and create aggregated features
session_features = session_df.groupby('session ID').agg(
    session_duration=('order', 'max'),
    total_pages_viewed=('page', 'nunique'),
    main_category_count=('page 1 (main category)', 'nunique'),
    clothing_models_count=('page 2 (clothing model)', 'nunique'),
    avg_price=('price', 'mean'),
    max_price=('price', 'max'),
    min_price=('price', 'min'),
    price_range=('price', lambda x: x.max() - x.min()),
    country=('country', 'first')
)

# Select the numerical features for clustering
numerical_features = [
    'session_duration', 'total_pages_viewed', 'main_category_count', 
    'clothing_models_count', 'avg_price', 'max_price', 'min_price', 'price_range'
]

# Add country as a categorical feature (might be useful)
categorical_features = ['country']

# Print some information about our features
print(f"\nSession features shape: {session_features.shape}")
print("\nFeature statistics:")
print(session_features[numerical_features].describe())

# Handle NaN values if any are present
if session_features[numerical_features].isnull().sum().sum() > 0:
    session_features[numerical_features] = session_features[numerical_features].fillna(0)

# ----------------------
# Finding Optimal K
# ----------------------

# Function to determine optimal number of clusters using Elbow method and Silhouette score
def find_optimal_clusters(data, max_k):
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # To store the results
    inertia_values = []
    silhouette_values = []
    
    # Try different numbers of clusters
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        
        # Calculate inertia (sum of squared distances to closest centroid)
        inertia_values.append(kmeans.inertia_)
        
        # Calculate silhouette score
        cluster_labels = kmeans.labels_
        silhouette_avg = silhouette_score(scaled_data, cluster_labels)
        silhouette_values.append(silhouette_avg)
        
        print(f"For k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette_avg:.4f}")
        
    return inertia_values, silhouette_values

print("\nDetermining optimal number of clusters...")
max_clusters = 10
inertia, silhouette = find_optimal_clusters(session_features[numerical_features], max_clusters)

# Plot the Elbow Method result
plt.figure(figsize=(16, 6))

# Plot inertia (Elbow method)
plt.subplot(1, 2, 1)
plt.plot(range(2, max_clusters + 1), inertia, marker='o', linestyle='-')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)

# Plot silhouette scores
plt.subplot(1, 2, 2)
plt.plot(range(2, max_clusters + 1), silhouette, marker='o', linestyle='-')
plt.title('Silhouette Score for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)

plt.tight_layout()
plt.savefig('kmeans_optimal_k.png')
print("Saved optimal k analysis to 'kmeans_optimal_k.png'")

# Based on the analysis, determine optimal k
# (The actual optimal k would be determined by examining the plots)
# For this implementation, we'll use k=4 but this should be adjusted after examining the plots
optimal_k = 4

# ----------------------
# K-means Clustering
# ----------------------

# Prepare the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features)
    ]
)

# Create the K-means pipeline
kmeans_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=optimal_k, random_state=42, n_init=10))
])

# Fit the pipeline to the data
kmeans_pipeline.fit(session_features[numerical_features])

# Get the cluster assignments
session_features['cluster'] = kmeans_pipeline.named_steps['kmeans'].labels_

# Display cluster distribution
cluster_counts = session_features['cluster'].value_counts().sort_index()
print("\nCluster distribution:")
print(cluster_counts)

# Calculate percentage distribution properly
for cluster_id, count in cluster_counts.items():
    percentage = (count / len(session_features)) * 100
    print(f"Cluster {cluster_id}: {count} sessions ({percentage:.2f}%)")

# ----------------------
# Cluster Analysis
# ----------------------

# Analyzing cluster characteristics
cluster_analysis = session_features.groupby('cluster')[numerical_features].mean()
print("\nCluster Centers (average values for each feature):")
print(cluster_analysis)

# Visualize the clusters (using PCA for dimensionality reduction)
print("\nVisualizing clusters using PCA...")

# Scale the numerical data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(session_features[numerical_features])

# Apply PCA to reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

# Create a DataFrame with the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
pca_df['cluster'] = session_features['cluster']

# Plot the PCA results
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='cluster', palette='viridis', data=pca_df, s=50)
plt.title('Session Clusters Visualization using PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.savefig('kmeans_clusters_pca.png')
print("Saved cluster visualization to 'kmeans_clusters_pca.png'")

# ----------------------
# Feature Importance
# ----------------------

# Calculate feature importance by examining the distance of features from cluster centers
feature_importance = pd.DataFrame(
    np.abs(kmeans_pipeline.named_steps['kmeans'].cluster_centers_),
    columns=numerical_features
)

print("\nFeature importance based on cluster centers:")
print(feature_importance)

# Plot feature importance for each cluster
plt.figure(figsize=(14, 8))
feature_importance.T.plot(kind='bar', ax=plt.gca())
plt.title('Feature Importance by Cluster')
plt.xlabel('Feature')
plt.ylabel('Absolute Centroid Coordinate Value')
plt.legend(title='Cluster')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('kmeans_feature_importance.png')
print("Saved feature importance visualization to 'kmeans_feature_importance.png'")

# ----------------------
# Cluster Interpretation
# ----------------------

print("\nCluster Interpretation:")
for i in range(optimal_k):
    print(f"\nCluster {i} characteristics:")
    # For each numerical feature, compute the difference from the overall mean
    for feature in numerical_features:
        overall_mean = session_features[feature].mean()
        cluster_mean = session_features[session_features['cluster'] == i][feature].mean()
        diff_percent = ((cluster_mean - overall_mean) / overall_mean * 100) if overall_mean != 0 else 0
        
        importance = "average"
        if abs(diff_percent) > 50:
            importance = "much higher" if diff_percent > 0 else "much lower"
        elif abs(diff_percent) > 20:
            importance = "higher" if diff_percent > 0 else "lower"
            
        if importance == "average":
            print(f"  - {feature}: {cluster_mean:.2f} (average, {diff_percent:.1f}% difference)")
        else:
            print(f"  - {feature}: {cluster_mean:.2f} ({importance} than average, {diff_percent:.1f}% difference)")
        
# Save the clustered data
session_features.to_csv('kmeans_clustered_sessions.csv')
print("\nSaved clustered data to 'kmeans_clustered_sessions.csv'")

print("\nK-means clustering analysis completed!")