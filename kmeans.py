import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class KMeansFromScratch:
    """
    K-Means clustering implemented from scratch using NumPy.
    Supports 2D visualization if data is 2D or reduced to 2D.
    """

    def __init__(self, n_clusters: int = 3, max_iters: int = 300, tol: float = 1e-4, random_state: int = 42):
        """
        Initialize K-Means parameters.
        
        Args:
            n_clusters (int): Number of clusters.
            max_iters (int): Maximum number of iterations.
            tol (float): Tolerance for convergence (relative change in centroids).
            random_state (int): Seed for reproducibility.
        """
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None  # Sum of squared distances to nearest centroid

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids by randomly selecting data points."""
        np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[indices].astype(np.float64)

    def _compute_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distances from each point to each centroid.
        
        Returns:
            distances: shape (n_samples, n_clusters)
        """
        # Efficient broadcasting: (n, d) vs (k, d) → (n, k)
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (n, k, d)
        return np.linalg.norm(diff, axis=2)  # (n, k)

    def _assign_labels(self, distances: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest centroid."""
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute new centroids as mean of assigned points."""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = cluster_points.mean(axis=0)
            else:
                # Reinitialize empty centroid (fallback)
                centroids[i] = X[np.random.choice(X.shape[0])]
        return centroids

    def _has_converged(self, old_centroids: np.ndarray, new_centroids: np.ndarray) -> bool:
        """Check convergence based on relative change in centroids."""
        # Use Frobenius norm of difference
        shift = np.linalg.norm(new_centroids - old_centroids)
        norm_old = np.linalg.norm(old_centroids)
        if norm_old == 0:
            return shift < self.tol
        return (shift / norm_old) < self.tol

    def fit(self, X: np.ndarray) -> 'KMeansFromScratch':
        """
        Fit K-Means clustering on data X.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Input X must be 2D array.")

        self.centroids_ = self._initialize_centroids(X)

        for _ in range(self.max_iters):
            distances = self._compute_distances(X, self.centroids_)
            self.labels_ = self._assign_labels(distances)
            old_centroids = self.centroids_.copy()
            self.centroids_ = self._update_centroids(X, self.labels_)

            if self._has_converged(old_centroids, self.centroids_):
                break

        # Compute inertia (within-cluster sum of squares)
        final_distances = self._compute_distances(X, self.centroids_)
        self.inertia_ = np.sum(np.min(final_distances, axis=1) ** 2)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        if self.centroids_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        X = np.asarray(X, dtype=np.float64)
        distances = self._compute_distances(X, self.centroids_)
        return self._assign_labels(distances)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        return self.fit(X).labels_


def load_iris_from_tfds() -> Tuple[np.ndarray, np.ndarray]:
    """Load Iris dataset using TensorFlow Datasets."""
    ds, info = tfds.load('iris', split='train', with_info=True, as_supervised=False)
    features = []
    labels = []
    for example in tfds.as_numpy(ds):
        features.append(example['features'])
        labels.append(example['label'])
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y


def main():
    # Load data
    X, y_true = load_iris_from_tfds()
    print(f"Loaded Iris dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Apply K-Means
    kmeans = KMeansFromScratch(n_clusters=3, random_state=42)
    labels_pred = kmeans.fit_predict(X)

    print(f"Inertia (WCSS): {kmeans.inertia_:.4f}")

    # Evaluate against true labels (note: label mismatch is expected due to permutation)
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(y_true, labels_pred)
    print(f"Adjusted Rand Index (vs true labels): {ari:.4f}")

    # Visualize if 2D (use first two features)
    if X.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels_pred, cmap='viridis', alpha=0.7, s=50)
        plt.scatter(kmeans.centroids_[:, 0], kmeans.centroids_[:, 1],
                    c='red', marker='x', s=200, linewidths=3, label='Centroids')
        plt.title('K-Means Clustering on Iris Dataset (First 2 Features)')
        plt.xlabel('Sepal Length')
        plt.ylabel('Sepal Width')
        plt.legend()
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()