import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate synthetic data
X = np.random.rand(100, 2)

# Create a KMeans clustering model
model = KMeans(n_clusters=3)

# Fit the model to the data
model.fit(X)

# Get cluster assignments for each data point
labels = model.labels_

# Get cluster centroids
centroids = model.cluster_centers_

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=100, color='black')
plt.show()
