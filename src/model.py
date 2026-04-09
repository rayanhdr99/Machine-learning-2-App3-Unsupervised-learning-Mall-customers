# model.py - handles KMeans training and hyperparameter evaluation (elbow + silhouette)
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


# calculate the within-cluster sum of squares for different k values (for the elbow method)
def compute_elbow(features: pd.DataFrame, k_range=range(3, 9)) -> pd.DataFrame:
    logger.info("Computing Elbow curve for k in %s", list(k_range))
    results = []

    # try different values of k and record the WCSS (inertia) for each
    for k in k_range:
        model = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
        model.fit(features)
        results.append({"k": k, "WCSS": model.inertia_})

    return pd.DataFrame(results)


# compute silhouette score for each k to find the best number of clusters
def compute_silhouette(features: pd.DataFrame, k_range=range(3, 9)) -> pd.DataFrame:
    logger.info("Computing Silhouette scores for k in %s", list(k_range))
    results = []

    # for each k, fit the model and calculate how well-separated the clusters are
    for k in k_range:
        model = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
        labels = model.fit_predict(features)  # fit and get cluster labels in one step
        score = silhouette_score(features, labels)  # higher score = better separation
        results.append({"k": k, "Silhouette_Score": score})

    return pd.DataFrame(results)


# train the final KMeans model with the chosen number of clusters
def train_kmeans(features: pd.DataFrame, n_clusters: int = 5) -> KMeans:
    logger.info("Training KMeans with n_clusters=%d", n_clusters)
    try:
        # use k-means++ initialization for better starting centroids
        model = KMeans(n_clusters=n_clusters, init='k-means++', n_init='auto', random_state=42)
        model.fit(features)
    except Exception as e:
        logger.error("KMeans training failed: %s", e)
        raise
    logger.info("KMeans training complete. Inertia: %.2f", model.inertia_)
    return model
