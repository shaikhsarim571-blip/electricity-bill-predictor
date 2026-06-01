import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings

warnings.filterwarnings('ignore')


class ConsumerSegmentation:
    def __init__(self, config):
        self.config = config
        self.kmeans = None
        self.scaler = StandardScaler()
        self.pca = None
        self.cluster_labels = None
        self.scaled_data = None
        self.pca_data = None
        
    def prepare_clustering_features(self, df):
        """Prepare features for clustering"""
        features = []
        
        if 'unit_consumption_kwh' in df.columns:
            features.append('unit_consumption_kwh')
        elif 'consumption' in df.columns:
            features.append('consumption')
        
        if 'temperature' in df.columns:
            features.append('temperature')
        if 'humidity' in df.columns:
            features.append('humidity')
        
        return features
    
    def fit(self, X, n_clusters=None):
        """Fit K-Means clustering"""
        if n_clusters is None:
            n_clusters = self.config.OPTIMAL_CLUSTERS
        
        self.scaled_data = self.scaler.fit_transform(X)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(self.scaled_data)
        
        return self.cluster_labels
    
    def apply_pca(self, n_components=2):
        """Apply PCA for 2D/3D visualization"""
        self.pca = PCA(n_components=n_components)
        self.pca_data = self.pca.fit_transform(self.scaled_data)
        
        return self.pca_data
    
    def get_silhouette_score(self):
        """Calculate silhouette score"""
        if self.cluster_labels is None:
            return None
        return silhouette_score(self.scaled_data, self.cluster_labels)
    
    def get_davies_bouldin_score(self):
        """Calculate Davies-Bouldin Index"""
        if self.cluster_labels is None:
            return None
        return davies_bouldin_score(self.scaled_data, self.cluster_labels)
    
    def assign_segments(self, df):
        """Assign consumer segment names based on clusters"""
        df_copy = df.copy()
        df_copy['cluster'] = self.cluster_labels
        
        avg_consumption = df_copy.groupby('cluster')['unit_consumption_kwh'].mean().sort_values()
        
        segment_names = [
            "Low-Income/Protected",
            "Middle-Class/Slab-Breacher",
            "Heavy Commercial/AC-Heavy",
            "Industrial/Extreme"
        ]
        
        cluster_to_segment = {}
        for idx, cluster_id in enumerate(sorted(df_copy['cluster'].unique())):
            if idx < len(segment_names):
                cluster_to_segment[cluster_id] = segment_names[idx]
            else:
                cluster_to_segment[cluster_id] = f"Segment {idx}"
        
        df_copy['segment'] = df_copy['cluster'].map(cluster_to_segment)
        return df_copy
    
    def get_cluster_profiles(self, df):
        """Get summary statistics for each cluster"""
        df_copy = df.copy()
        df_copy['cluster'] = self.cluster_labels
        
        profiles = df_copy.groupby('cluster').agg({
            'unit_consumption_kwh': ['mean', 'std', 'min', 'max', 'count'],
            'temperature': 'mean',
            'humidity': 'mean'
        }).round(2)
        
        return profiles
    
    def get_optimal_clusters(self, X, max_clusters=8):
        """Find optimal number of clusters using elbow method"""
        inertias = []
        silhouettes = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaler.fit_transform(X))
            
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(self.scaler.fit_transform(X), labels))
        
        return {
            'k_range': list(range(2, max_clusters + 1)),
            'inertias': inertias,
            'silhouettes': silhouettes
        }
    
    def get_pca_variance_explained(self):
        """Get PCA variance explained"""
        if self.pca is None:
            return None
        
        return {
            'ratio': self.pca.explained_variance_ratio_.tolist(),
            'cumulative': np.cumsum(self.pca.explained_variance_ratio_).tolist()
        }
