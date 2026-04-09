# evaluator.py - evaluates how good our clustering results are
import logging
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


# compute inertia and silhouette score for a fitted KMeans model
def evaluate_clustering(model, features) -> dict:
    # get the cluster labels and inertia from the trained model
    labels = model.labels_
    inertia = model.inertia_

    # try computing silhouette score (might fail if there's only 1 cluster)
    try:
        sil = silhouette_score(features, labels)
    except Exception as e:
        logger.warning("Silhouette score computation failed: %s", e)
        sil = None

    logger.info("Inertia: %.2f  Silhouette: %s", inertia, sil)

    # return everything as a dictionary so we can display it in the app
    return {"inertia": inertia, "silhouette_score": sil, "n_clusters": model.n_clusters}
