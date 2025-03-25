import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv('customer_data.csv')

# Ask user for number of clusters
n_clusters = int(input("Enter the number of clusters (K): "))

# Prepare the data
# Assuming your CSV has numerical features that you want to use for clustering
X = df.select_dtypes(include=[np.number])  # Select only numerical columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the results
# We'll create a scatter plot using the first two numerical features
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], 
                     c=df['Cluster'], 
                     cmap='viridis')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title('Customer Clusters')
plt.colorbar(scatter, label='Cluster')
plt.show()

# Print summary of clusters
print("\nCluster Summary:")
print(df.groupby('Cluster').size())