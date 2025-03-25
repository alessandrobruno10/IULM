import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.cm as cm

# Load the data
df = pd.read_csv('market_segmentation_geography.csv')
print(f"Loaded data with {df.shape[0]} regions and {df.shape[1]} features")

# Select features for clustering
features = ['gdp_per_capita', 'population_density', 'avg_household_income', 
           'unemployment_rate', 'internet_penetration', 'urban_percentage',
           'electronics_sales_index', 'clothing_sales_index', 
           'furniture_sales_index', 'food_sales_index']

# Scale the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Create a DataFrame for the scaled data with labels
df_scaled = pd.DataFrame(df_scaled, columns=features)

# Compute the linkage matrix (choosing ward method to minimize variance)
Z = linkage(df_scaled, method='ward')

# Plot the dendrogram to visualize the hierarchy
plt.figure(figsize=(16, 10))
plt.title('Hierarchical Clustering Dendrogram for Market Segmentation', fontsize=18)
plt.xlabel('States/Regions', fontsize=14)
plt.ylabel('Distance', fontsize=14)

# Plot with truncation to make it readable
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=10.,  # font size for the x axis labels
    labels=df['state'].values,
    color_threshold=7.5,  # sets the color threshold for coloring the branches
)

# Add a horizontal line to suggest where to cut the tree
plt.axhline(y=7.5, color='r', linestyle='--', label='Suggested cluster cutoff')
plt.legend()
plt.tight_layout()
plt.savefig('geographic_clustering_dendrogram.png', dpi=300)

# Based on the dendrogram analysis, determine an appropriate number of clusters
# Let's try 5 clusters which seems reasonable
n_clusters = 5
hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
df['cluster'] = hc.fit_predict(df_scaled)

# Add more descriptive cluster names based on characteristics
# This will be populated after analyzing the cluster profiles
cluster_names = {}

# Analyze the clusters
cluster_profiles = df.groupby('cluster')[features].mean().round(2)
print("\nCluster Profiles:")
print(cluster_profiles)

# Add size of each cluster
cluster_sizes = df.groupby('cluster').size()
print("\nCluster Sizes:")
print(cluster_sizes)

# Create a function to determine cluster names based on their profiles
def name_clusters(profiles):
    names = {}
    for cluster in profiles.index:
        # Get this cluster's profile
        profile = profiles.loc[cluster]
        
        # Check dominant characteristics
        if profile['gdp_per_capita'] > profiles['gdp_per_capita'].mean() and profile['urban_percentage'] > profiles['urban_percentage'].mean():
            if profile['population_density'] > profiles['population_density'].mean() * 1.2:
                names[cluster] = "Urban Affluent Markets"
            else:
                names[cluster] = "Suburban Wealthy Markets"
        elif profile['rural_percentage'] if 'rural_percentage' in profile else (100 - profile['urban_percentage']) > 40:
            if profile['avg_household_income'] < profiles['avg_household_income'].mean():
                names[cluster] = "Rural Value Markets"
            else:
                names[cluster] = "Rural Affluent Markets"
        elif profile['electronics_sales_index'] > profiles['electronics_sales_index'].mean() and profile['internet_penetration'] > profiles['internet_penetration'].mean():
            names[cluster] = "Tech-Savvy Markets"
        elif profile['unemployment_rate'] > profiles['unemployment_rate'].mean() * 1.1:
            names[cluster] = "Economically Challenged Markets"
        else:
            names[cluster] = f"General Market {cluster+1}"
    
    return names

# Generate cluster names
cluster_names = name_clusters(cluster_profiles)
print("\nCluster Names:")
for cluster, name in cluster_names.items():
    print(f"Cluster {cluster}: {name}")

# Add cluster names to the dataframe
df['cluster_name'] = df['cluster'].map(cluster_names)

# Visualize the clusters using a scatter plot matrix of key variables
plt.figure(figsize=(20, 20))
sns.pairplot(df, hue='cluster_name', 
             vars=['gdp_per_capita', 'population_density', 'avg_household_income', 'urban_percentage'], 
             palette='viridis', diag_kind='kde', 
             plot_kws={'alpha': 0.6, 's': 80})
plt.suptitle('Market Clusters by Key Economic Indicators', y=1.02, fontsize=24)
plt.tight_layout()
plt.savefig('geographic_clustering_pairplot.png', dpi=300)

# Create a heatmap of the cluster profiles
plt.figure(figsize=(14, 8))
sns.heatmap(cluster_profiles, annot=True, cmap='YlGnBu', fmt='.1f', linewidths=.5)
plt.title('Market Segment Characteristics by Cluster', fontsize=18)
plt.tight_layout()
plt.savefig('geographic_clustering_heatmap.png', dpi=300)

# Create regional distribution of clusters
region_cluster_distribution = pd.crosstab